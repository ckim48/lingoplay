import os
import re
import json
import sqlite3
import random
from datetime import datetime
from functools import wraps
from urllib.parse import urlparse, urljoin


from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, g, session, jsonify
)
from werkzeug.security import generate_password_hash, check_password_hash
from slugify import slugify

from openai import OpenAI

# -------------------------------------------------------------------
# Config & OpenAI
# -------------------------------------------------------------------

DEFAULT_MODEL = os.getenv("LINGOPLAY_MODEL", "gpt-4.1-mini")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

app = Flask(__name__, instance_relative_config=True)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "dev-secret")

# DB path in instance folder
os.makedirs(app.instance_path, exist_ok=True)
DB_PATH = os.path.join(app.instance_path, "lingoplay.db")

# -------------------------------------------------------------------
# Fallback / game constants
# -------------------------------------------------------------------
PREPARED_SCRAMBLE_WORDS = [
    "apple", "banana", "school", "teacher", "pencil",
    "friend", "family", "happy", "forest", "cookie",
    "dragon", "magic", "ocean", "bubble", "rocket",
    "puzzle", "reading", "smile", "garden", "rainbow"
]

# -------------------------------------------------------------------
# DB helpers
# -------------------------------------------------------------------
def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d


def get_db():
    if "db" not in g:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = dict_factory
        g.db = conn
    return g.db


@app.teardown_appcontext
def close_db(exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()
def init_db():
    """Create the tables used by the app and run light migrations."""
    db = get_db()

    db.executescript(
        """
        PRAGMA foreign_keys = ON;

        -- Dictionary lookup log (EN <-> KO)
        CREATE TABLE IF NOT EXISTS dict_lookups (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id      INTEGER REFERENCES users(id) ON DELETE SET NULL,
            source_lang  TEXT NOT NULL,          -- 'en' or 'ko'
            target_lang  TEXT NOT NULL,          -- 'ko' or 'en'
            query        TEXT NOT NULL,          -- lookup term
            result_json  TEXT,                   -- cached result (definitions, etc.)
            created_at   TEXT DEFAULT (datetime('now'))
        );

        -- Dictionary bookmarks
        CREATE TABLE IF NOT EXISTS dict_bookmarks (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id      INTEGER REFERENCES users(id) ON DELETE CASCADE,
            source_lang  TEXT NOT NULL,
            target_lang  TEXT NOT NULL,
            word         TEXT NOT NULL,
            translation  TEXT,
            note         TEXT,
            created_at   TEXT DEFAULT (datetime('now')),
            UNIQUE(user_id, source_lang, target_lang, word)
        );

        -- All other CREATE TABLE statements are managed elsewhere.
        -- This function is safe to call multiple times and acts as a migration.
        """
    )

    # Add a flag column for library sharing (if it doesn't exist yet)
    try:
        db.execute(
            "ALTER TABLE stories ADD COLUMN is_shared_library INTEGER DEFAULT 0"
        )
    except sqlite3.OperationalError:
        # Column already exists, ignore
        pass

    # Store MCQ questions once per story
    try:
        db.execute(
            "ALTER TABLE stories ADD COLUMN mcq_questions_json TEXT"
        )
    except sqlite3.OperationalError:
        # Column already exists, ignore
        pass

    # Store per-assignment MCQ questions (copy from story at assign time)
    try:
        db.execute(
            "ALTER TABLE assignments ADD COLUMN questions_json TEXT"
        )
    except sqlite3.OperationalError:
        # Column already exists, ignore
        pass

    # Track number of trials (submissions) per assignment
    try:
        db.execute(
            "ALTER TABLE assignments ADD COLUMN attempt_count INTEGER DEFAULT 0"
        )
    except sqlite3.OperationalError:
        # Column already exists, ignore
        pass

    # Optional human-readable title for each assignment
    try:
        db.execute(
            "ALTER TABLE assignments ADD COLUMN assignment_title TEXT"
        )
    except sqlite3.OperationalError:
        # Column already exists, ignore
        pass

    db.commit()


with app.app_context():
    init_db()



with app.app_context():
    init_db()



# -------------------------------------------------------------------
# Template filters
# -------------------------------------------------------------------
@app.template_filter("dt")
def format_dt(value, fmt="%Y-%m-%d %H:%M"):
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.strftime(fmt)
    try:
        s = str(value).strip().replace("Z", "")
        if "." in s:
            s = s.split(".", 1)[0]
        dt = datetime.fromisoformat(s)
        return dt.strftime(fmt)
    except Exception:
        return value

# -------------------------------------------------------------------
# Auth & current user
# -------------------------------------------------------------------
def get_user_by_identifier(identifier: str):
    db = get_db()
    ident = (identifier or "").strip().lower()
    row = db.execute("SELECT * FROM users WHERE lower(email)=?", (ident,)).fetchone()
    if row:
        return row
    row = db.execute("SELECT * FROM users WHERE lower(username)=?", (ident,)).fetchone()
    return row


@app.before_request
def load_current_user():
    g.current_user = None
    uid = session.get("user_id")
    if uid:
        db = get_db()
        g.current_user = db.execute("SELECT * FROM users WHERE id = ?", (uid,)).fetchone()


@app.context_processor
def inject_user():
    return {"current_user": g.get("current_user")}


def is_safe_url(target: str) -> bool:
    try:
        ref = urlparse(request.host_url)
        test = urlparse(urljoin(request.host_url, target))
        return test.scheme in ("http", "https") and ref.netloc == test.netloc
    except Exception:
        return False


def login_required(view_func):
    @wraps(view_func)
    def _wrapped(*args, **kwargs):
        if not (g.get("current_user") and g.current_user.get("id")):
            next_url = request.full_path if request.query_string else request.path
            return redirect(url_for("login", next=next_url))
        return view_func(*args, **kwargs)

    return _wrapped


def current_user_is_admin() -> bool:
    # Admin is the special username "testtest"
    if session.get("username") == "testtest":
        return True
    return False

# -------------------------------------------------------------------
# Auth routes
# -------------------------------------------------------------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        db = get_db()
        username = (request.form.get("username") or "").strip()
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""
        confirm = request.form.get("confirm") or ""

        # Optional fields
        native_en_raw = (request.form.get("is_english_native") or "").strip().lower()
        is_english_native = None
        if native_en_raw == "yes":
            is_english_native = 1
        elif native_en_raw == "no":
            is_english_native = 0

        age_raw = (request.form.get("age") or "").strip()
        age = None
        if age_raw:
            if not age_raw.isdigit():
                flash("Please enter a valid age (number).", "warning")
                return redirect(url_for("register"))
            age = int(age_raw)
            if age < 5 or age > 120:
                flash("Please enter an age between 5 and 120.", "warning")
                return redirect(url_for("register"))

        gender = (request.form.get("gender") or "").strip().lower() or None
        if gender and gender not in ("female", "male", "nonbinary", "prefer_not"):
            flash("Please choose a valid gender option.", "warning")
            return redirect(url_for("register"))

        if not username or not email or not password:
            flash("Please fill in all required fields.", "warning")
            return redirect(url_for("register"))
        if password != confirm:
            flash("Passwords do not match.", "warning")
            return redirect(url_for("register"))
        if len(password) < 8:
            flash("Password must be at least 8 characters.", "warning")
            return redirect(url_for("register"))
        if not re.match(r"^[A-Za-z0-9_.-]{3,32}$", username):
            flash("Username must be 3–32 chars (letters, numbers, _, ., -).", "warning")
            return redirect(url_for("register"))
        if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
            flash("Please enter a valid email address.", "warning")
            return redirect(url_for("register"))

        try:
            pwd_hash = generate_password_hash(password)
            db.execute(
                """
                INSERT INTO users
                (username, email, password_hash, is_english_native, age, gender)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (username, email, pwd_hash, is_english_native, age, gender),
            )
            db.commit()
            flash("Registration successful. You can now sign in.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError as e:
            if "users.username" in str(e):
                flash("That username is already taken.", "danger")
            elif "users.email" in str(e):
                flash("An account with that email already exists.", "danger")
            else:
                flash("Could not create account. Please try again.", "danger")
            return redirect(url_for("register"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    next_url = request.args.get("next") or request.form.get("next")

    if request.method == "POST":
        username = (request.form.get("username") or "").strip().lower()
        password = request.form.get("password") or ""

        if not username or not password:
            error = "Please enter both username and password."
        else:
            db = get_db()
            user = db.execute(
                "SELECT * FROM users WHERE lower(username)=?",
                (username,),
            ).fetchone()

            if not user or not check_password_hash(user["password_hash"], password):
                error = "Wrong username or password."
            else:
                session["username"] = user["username"]
                session["user_id"] = user["id"]
                flash(f"Welcome back, {user['username']}!", "success")
                if next_url and is_safe_url(next_url):
                    if next_url.endswith("?"):
                        next_url = next_url[:-1]
                    return redirect(next_url)
                return redirect(url_for("index"))

    return render_template("login.html", next=next_url or "", error=error)


@app.route("/logout")
def logout():
    session.pop("user_id", None)
    session.pop("username", None)
    # flash("You have been signed out.", "info")
    return redirect(url_for("index"))

# -------------------------------------------------------------------
# Learner profile & logging
# -------------------------------------------------------------------
def get_learner_profile():
    u = g.get("current_user") or {}
    age = u.get("age")
    try:
        age = int(age) if age is not None else None
    except Exception:
        age = None

    native = u.get("is_english_native")
    if native is None:
        is_native = None
    else:
        try:
            is_native = bool(int(native))
        except Exception:
            is_native = None

    gender = (u.get("gender") or "").strip().lower() or None
    if gender not in ("female", "male", "nonbinary", "prefer_not"):
        gender = None

    return {
        "age": age,
        "is_english_native": is_native,
        "gender": gender,
        "username": u.get("username") or "guest",
    }


def log_input(action: str, payload: dict):
    db = get_db()
    db.execute(
        "INSERT INTO input_logs (action, payload_json) VALUES (?, ?)",
        (action, json.dumps(payload)),
    )
    db.commit()

# -------------------------------------------------------------------
# MCQ generator
# -------------------------------------------------------------------
def generate_mcq_questions(story_text: str, num_questions: int = 5) -> list[dict]:
    """
    Use the LLM to generate simple reading comprehension MCQs for the story.
    Returns a list like:
    [
      {
        "question": "...",
        "options": ["A", "B", "C", "D"],
        "answer_index": 1
      },
      ...
    ]
    """
    if client is None:
        return []

    system = (
        "You are an English reading comprehension question writer for young learners. "
        "You make short, clear multiple-choice questions (4 options) about the given story. "
        "Questions should be answerable directly from the story, not from prior knowledge."
    )
    user = (
        "Story:\n"
        f"{story_text[:2500]}\n\n"
        f"Create {num_questions} reading comprehension questions for this story.\n"
        "Each question must have exactly 4 options.\n"
        "Return ONLY valid JSON representing a list, where each item has:\n"
        "{ \"question\": str, \"options\": [str, str, str, str], \"answer_index\": int }\n"
        "Do not include explanations, comments, or any extra text outside of the JSON."
    )

    resp = client.responses.create(
        model=DEFAULT_MODEL,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.5,
        max_output_tokens=800,
    )
    raw = getattr(resp, "output_text", "") or "[]"
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            cleaned = []
            for item in data:
                q = {
                    "question": str(item.get("question", "")).strip(),
                    "options": list(item.get("options") or [])[:4],
                    "answer_index": int(item.get("answer_index", 0)),
                }
                if q["question"] and len(q["options"]) == 4:
                    if not (0 <= q["answer_index"] < 4):
                        q["answer_index"] = 0
                    cleaned.append(q)
            return cleaned
    except Exception as e:
        log_input("mcq_parse_error", {"error": str(e), "raw": raw})

    return []

# -------------------------------------------------------------------
# ADMIN: assignment creation page (per story)
# -------------------------------------------------------------------
@app.route("/admin/stories/<slug>/assign", methods=["GET", "POST"])
@login_required
def admin_assign_story(slug: str):
    """
    Admin page:
      - choose assignment type (finish or mcq)
      - select which students will get this story

    Usually opened via the Assign modal on admin_stories.
    """
    if not current_user_is_admin():
        return redirect(url_for("index"))

    db = get_db()

    # Story info
    story = db.execute(
        "SELECT * FROM stories WHERE slug = ?",
        (slug,),
    ).fetchone()
    if not story:
        return redirect(url_for("admin_stories"))

    # Latest draft for this story (same logic as admin_story_detail/admin_stories)
    draft = db.execute(
        """
        SELECT *
        FROM finish_drafts
        WHERE story_id = ?
        ORDER BY datetime(created_at) DESC
        LIMIT 1
        """,
        (story["id"],),
    ).fetchone()

    # All students (exclude admin "testtest"), same as admin_stories()
    users = db.execute(
        """
        SELECT id, username, email
        FROM users
        WHERE username != 'testtest'
        ORDER BY username COLLATE NOCASE
        """
    ).fetchall()

    # -----------------------
    # POST: create assignments
    # -----------------------
    if request.method == "POST":
        assignment_type = (request.form.get("assignment_type") or "").strip()
        user_ids = request.form.getlist("user_ids")

        # NEW: assignment title from form (optional)
        assignment_title = (request.form.get("assignment_title") or "").strip()
        if not assignment_title:
            # fallback: story title
            assignment_title = story["title"]

        if assignment_type not in ("finish", "mcq"):
            flash("Please choose a valid assignment type.", "warning")
            return redirect(url_for("admin_assign_story", slug=slug))

        if not user_ids:
            flash("Please select at least one student.", "warning")
            return redirect(url_for("admin_assign_story", slug=slug))

        if not draft:
            flash("This story does not have a linked draft yet.", "warning")
            return redirect(url_for("admin_assign_story", slug=slug))

        # Determine completion status of the draft
        completion_text = draft["completion_text"] if "completion_text" in draft.keys() else None
        has_completion = bool((completion_text or "").strip())
        can_finish = bool(draft and not has_completion)
        can_mcq = bool(draft and has_completion)

        if assignment_type == "finish" and not can_finish:
            flash(
                "Finish-writing is only for unfinished stories (no completion_text).",
                "warning",
            )
            return redirect(url_for("admin_assign_story", slug=slug))

        if assignment_type == "mcq" and not can_mcq:
            flash(
                "MCQ reading is only for fully completed stories.",
                "warning",
            )
            return redirect(url_for("admin_assign_story", slug=slug))

        # -------------------------------------------------
        # For MCQ: require pre-generated questions on the story
        # -------------------------------------------------
        questions_json = None
        if assignment_type == "mcq":
            base_q_json = None
            try:
                base_q_json = story["mcq_questions_json"]
            except Exception:
                base_q_json = None

            if base_q_json:
                questions_json = base_q_json
            else:
                flash(
                    "MCQ questions are not generated yet for this story. "
                    "Use the 'Generate MCQ Questions' button first.",
                    "warning",
                )
                return redirect(url_for("admin_story_detail", slug=slug))

        # -------------------------------------------------
        # Prevent duplicate assignments for same story+type+student
        # -------------------------------------------------
        placeholders = ",".join("?" for _ in user_ids)
        already = set()
        if placeholders:
            rows = db.execute(
                f"""
                SELECT assignee_id
                FROM assignments
                WHERE story_id = ? AND assignment_type = ?
                  AND assignee_id IN ({placeholders})
                """,
                (story["id"], assignment_type, *user_ids),
            ).fetchall()
            already = {str(r["assignee_id"]) for r in rows}

        selected_ids = [uid for uid in user_ids if uid not in already]

        if not selected_ids:
            flash(
                "All selected students already have this type of assignment for this story.",
                "info",
            )
            return redirect(url_for("admin_story_detail", slug=slug))

        # Create one assignment per *new* student
        created = 0
        for uid in selected_ids:
            try:
                assignee_id = int(uid)
            except ValueError:
                continue

            db.execute(
                """
                INSERT INTO assignments
                (story_id, draft_id, assignee_id, assignment_type,
                 questions_json, assigned_by, assignment_title)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    story["id"],
                    draft["id"],
                    assignee_id,
                    assignment_type,
                    questions_json,  # None for finish-writing, JSON for MCQ
                    g.current_user["id"] if g.current_user else None,
                    assignment_title,
                ),
            )
            created += 1

        db.commit()

        if created:
            msg = f"Assigned to {created} new student(s)."
            if already:
                msg += f" ({len(already)} already had this assignment and were skipped.)"
            flash(msg, "success")
        else:
            flash("No assignments were created.", "warning")

        return redirect(url_for("admin_story_detail", slug=slug))

    # -----------------------
    # GET: render standalone assign page (not used by modal)
    # -----------------------
    return render_template(
        "admin_assign_story.html",
        story=story,
        users=users,
        draft=draft,
    )


@app.get("/admin/analytics")
@login_required
def admin_analytics():
    """
    Admin analytics dashboard:
    - Tab 1: L1 vs L2 (native vs non-native) comparison
    - Tab 2: Single-student view
    """
    if not current_user_is_admin():
        flash("Admin access only.", "warning")
        return redirect(url_for("index"))

    db = get_db()

    # -----------------------------
    # Load all non-admin students
    # -----------------------------
    students = db.execute(
        """
        SELECT id, username, email, is_english_native
        FROM users
        WHERE username != 'testtest'
        ORDER BY username COLLATE NOCASE
        """
    ).fetchall()

    # Map id -> student
    student_map = {s["id"]: s for s in students}

    # -----------------------------
    # Group-level: L1 vs L2
    # -----------------------------
    # Base structures (we'll fill with SQL results)
    group_stats = {
        1: {  # L1
            "code": "L1",
            "label": "L1 · Native English",
            "student_count": 0,
            "assignments_total": 0,
            "assignments_completed": 0,
            "assignments_avg_score": 0.0,
            "assignments_avg_attempts": 0.0,
            "scramble_sessions": 0,
            "scramble_total_correct": 0,
            "scramble_total_questions": 0,
            "scramble_accuracy": 0.0,
            "dict_lookups": 0,
            "top_queries": [],
        },
        0: {  # L2
            "code": "L2",
            "label": "L2 · Non-native English",
            "student_count": 0,
            "assignments_total": 0,
            "assignments_completed": 0,
            "assignments_avg_score": 0.0,
            "assignments_avg_attempts": 0.0,
            "scramble_sessions": 0,
            "scramble_total_correct": 0,
            "scramble_total_questions": 0,
            "scramble_accuracy": 0.0,
            "dict_lookups": 0,
            "top_queries": [],
        },
    }

    # Count students in each group
    for s in students:
        native = s.get("is_english_native")
        if native in group_stats:
            group_stats[native]["student_count"] += 1

    # --- Assignments group stats ---
    rows = db.execute(
        """
        SELECT
          u.is_english_native AS native,
          COUNT(a.id)         AS total_assignments,
          SUM(CASE WHEN a.status = 'submitted' THEN 1 ELSE 0 END) AS completed_assignments,
          AVG(a.score)        AS avg_score,
          AVG(a.attempt_count) AS avg_attempts
        FROM users u
        JOIN assignments a ON a.assignee_id = u.id
        WHERE u.username != 'testtest'
        GROUP BY u.is_english_native
        """
    ).fetchall()

    for r in rows:
        native = r["native"]
        if native not in group_stats:
            continue
        g = group_stats[native]
        g["assignments_total"] = r["total_assignments"] or 0
        g["assignments_completed"] = r["completed_assignments"] or 0
        g["assignments_avg_score"] = float(r["avg_score"] or 0.0)
        g["assignments_avg_attempts"] = float(r["avg_attempts"] or 0.0)

    # --- Scramble group stats ---
    rows = db.execute(
        """
        SELECT
          u.is_english_native AS native,
          COUNT(s.id)         AS sessions,
          COALESCE(SUM(s.correct_count), 0) AS total_correct,
          COALESCE(SUM(s.total), 0)        AS total_questions
        FROM users u
        JOIN scramble_sessions s ON s.user_id = u.id
        WHERE u.username != 'testtest'
        GROUP BY u.is_english_native
        """
    ).fetchall()

    for r in rows:
        native = r["native"]
        if native not in group_stats:
            continue
        g = group_stats[native]
        g["scramble_sessions"] = r["sessions"] or 0
        g["scramble_total_correct"] = r["total_correct"] or 0
        g["scramble_total_questions"] = r["total_questions"] or 0
        if g["scramble_total_questions"] > 0:
            g["scramble_accuracy"] = 100.0 * g["scramble_total_correct"] / g["scramble_total_questions"]
        else:
            g["scramble_accuracy"] = 0.0

    # --- Dictionary usage group stats ---
    rows = db.execute(
        """
        SELECT
          u.is_english_native AS native,
          COUNT(l.id)         AS lookups
        FROM users u
        JOIN dict_lookups l ON l.user_id = u.id
        WHERE u.username != 'testtest'
        GROUP BY u.is_english_native
        """
    ).fetchall()

    for r in rows:
        native = r["native"]
        if native not in group_stats:
            continue
        g = group_stats[native]
        g["dict_lookups"] = r["lookups"] or 0

    # --- Top queries per group ---
    def top_queries_for(native_flag: int, limit: int = 5):
        qrows = db.execute(
            """
            SELECT l.query, COUNT(*) AS cnt
            FROM users u
            JOIN dict_lookups l ON l.user_id = u.id
            WHERE u.username != 'testtest'
              AND u.is_english_native = ?
            GROUP BY l.query
            ORDER BY cnt DESC, l.query ASC
            LIMIT ?
            """,
            (native_flag, limit),
        ).fetchall()
        return qrows

    for native_flag in group_stats.keys():
        group_stats[native_flag]["top_queries"] = top_queries_for(native_flag)

    # Convert to list for easier iteration in Jinja
    group_list = [group_stats[1], group_stats[0]]

    # -----------------------------
    # Single student view
    # -----------------------------
    selected_user = None
    student_assign_stats = None
    student_assign_list = []
    student_scramble_summary = None
    student_scramble_sessions = []
    student_lookups = []

    selected_id = request.args.get("user_id", type=int)
    if not selected_id and students:
        # default to first student if none chosen
        selected_id = students[0]["id"]

    if selected_id and selected_id in student_map:
        selected_user = student_map[selected_id]

        # Assignments for this student
        arows = db.execute(
            """
            SELECT *
            FROM assignments
            WHERE assignee_id = ?
            ORDER BY datetime(created_at) DESC
            """,
            (selected_id,),
        ).fetchall()
        student_assign_list = arows

        total_a = len(arows)
        completed_a = sum(1 for a in arows if (a.get("status") == "submitted"))
        avg_score = 0.0
        avg_attempts = 0.0

        scores = [float(a["score"]) for a in arows if a.get("score") is not None]
        attempts = [float(a["attempt_count"]) for a in arows if a.get("attempt_count") is not None]

        if scores:
            avg_score = sum(scores) / len(scores)
        if attempts:
            avg_attempts = sum(attempts) / len(attempts)

        # breakdown by type
        type_breakdown = {}
        for a in arows:
            t = a.get("assignment_type") or "unknown"
            if t not in type_breakdown:
                type_breakdown[t] = {"count": 0, "completed": 0}
            type_breakdown[t]["count"] += 1
            if a.get("status") == "submitted":
                type_breakdown[t]["completed"] += 1

        student_assign_stats = {
            "total": total_a,
            "completed": completed_a,
            "avg_score": avg_score,
            "avg_attempts": avg_attempts,
            "by_type": type_breakdown,
        }

        # Scramble sessions for this student
        srows = db.execute(
            """
            SELECT *
            FROM scramble_sessions
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT 10
            """,
            (selected_id,),
        ).fetchall()
        student_scramble_sessions = srows

        total_sessions = len(srows)
        total_correct = sum((r.get("correct_count") or 0) for r in srows)
        total_questions = sum((r.get("total") or 0) for r in srows)
        accuracy = 0.0
        if total_questions > 0:
            accuracy = 100.0 * total_correct / total_questions

        student_scramble_summary = {
            "sessions": total_sessions,
            "total_correct": total_correct,
            "total_questions": total_questions,
            "accuracy": accuracy,
        }

        # Recent dictionary lookups
        student_lookups = db.execute(
            """
            SELECT query, created_at
            FROM dict_lookups
            WHERE user_id = ?
            ORDER BY datetime(created_at) DESC, id DESC
            LIMIT 15
            """,
            (selected_id,),
        ).fetchall()

    return render_template(
        "admin_analytics.html",
        group_list=group_list,
        students=students,
        selected_user=selected_user,
        student_assign_stats=student_assign_stats,
        student_assign_list=student_assign_list,
        student_scramble_summary=student_scramble_summary,
        student_scramble_sessions=student_scramble_sessions,
        student_lookups=student_lookups,
    )

# -------------------------------------------------------------------
# Assignment list & detail for students
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# Assignment list & detail for students
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# Assignment list & detail for students
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# Assignment list & detail for students
# -------------------------------------------------------------------
@app.get("/assignments")
@login_required
def assignments_list():
    db = get_db()
    rows = db.execute(
        """
        SELECT
          a.id,
          a.assignment_type,
          a.status,
          a.score,
          a.attempt_count,
          a.created_at,
          a.draft_id,
          a.assignment_title,
          s.id    AS story_id,
          s.slug  AS story_slug,
          s.title AS story_title
        FROM assignments a
        JOIN stories s ON a.story_id = s.id
        WHERE a.assignee_id = ?
        ORDER BY datetime(a.created_at) DESC
        """,
        (g.current_user["id"],),
    ).fetchall()

    return render_template("assignments.html", assignments=rows)

@app.route("/assignments/<int:assignment_id>", methods=["GET", "POST"])
@login_required
def assignment_detail(assignment_id: int):
    db = get_db()
    user_id = g.current_user["id"]

    # Make sure this assignment belongs to this student
    assignment = db.execute(
        """
        SELECT * FROM assignments
        WHERE id = ? AND assignee_id = ?
        """,
        (assignment_id, user_id),
    ).fetchone()

    if not assignment:
        flash("Assignment not found or not assigned to you.", "warning")
        return redirect(url_for("assignments_list"))

    # Story linked to this assignment
    story = db.execute(
        "SELECT * FROM stories WHERE id = ?",
        (assignment["story_id"],),
    ).fetchone()

    if not story:
        flash("Story for this assignment could not be found.", "warning")
        return redirect(url_for("assignments_list"))

    # Draft, if any
    draft = None
    if assignment.get("draft_id"):
        draft = db.execute(
            "SELECT * FROM finish_drafts WHERE id = ?",
            (assignment["draft_id"],),
        ).fetchone()

    # Latest submission (if any)
    submission = db.execute(
        """
        SELECT *
        FROM assignment_submissions
        WHERE assignment_id = ? AND user_id = ?
        ORDER BY datetime(updated_at) DESC
        LIMIT 1
        """,
        (assignment_id, user_id),
    ).fetchone()

    # ------------------------------------------------------------------
    # FINISH-WRITING ASSIGNMENT
    # ------------------------------------------------------------------
    if assignment["assignment_type"] == "finish":
        if request.method == "POST":
            completion_text = (request.form.get("completion_text") or "").strip()
            if not completion_text:
                flash("Please write your ending before submitting.", "warning")
                return redirect(request.url)

            now = datetime.utcnow().isoformat(timespec="seconds")

            if submission:
                db.execute(
                    """
                    UPDATE assignment_submissions
                    SET completion_text = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (completion_text, now, submission["id"]),
                )
            else:
                db.execute(
                    """
                    INSERT INTO assignment_submissions
                    (assignment_id, user_id, story_id, draft_id,
                     completion_text, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        assignment_id,
                        user_id,
                        assignment["story_id"],
                        assignment["draft_id"],
                        completion_text,
                        now,
                        now,
                    ),
                )

            # Increment attempts in SQL
            db.execute(
                """
                UPDATE assignments
                SET status = 'submitted',
                    attempt_count = COALESCE(attempt_count, 0) + 1
                WHERE id = ?
                """,
                (assignment_id,),
            )
            db.commit()

            flash("Your story ending has been submitted.", "success")
            return redirect(url_for("assignments_list"))

        # GET: render finish-writing page
        return render_template(
            "assignment_finish.html",
            assignment=assignment,
            story=story,
            draft=draft,
            submission=submission,
        )

    # ------------------------------------------------------------------
    # MCQ ASSIGNMENT
    # ------------------------------------------------------------------
    elif assignment["assignment_type"] == "mcq":
        questions = []
        if assignment.get("questions_json"):
            try:
                questions = json.loads(assignment["questions_json"])
            except Exception as e:
                log_input("mcq_questions_parse_error", {"error": str(e)})

        if request.method == "POST":
            answers = []
            correct_count = 0

            for idx, q in enumerate(questions):
                key = f"q{idx}"
                ans_raw = request.form.get(key)
                try:
                    ans_idx = int(ans_raw)
                except (TypeError, ValueError):
                    ans_idx = None

                answers.append(ans_idx)

                # Check correctness
                if ans_idx is not None and 0 <= ans_idx < len(q.get("options", [])):
                    if ans_idx == int(q.get("answer_index", -1)):
                        correct_count += 1

            score = 0.0
            if questions:
                score = (correct_count / len(questions)) * 100.0

            now = datetime.utcnow().isoformat(timespec="seconds")

            if submission:
                db.execute(
                    """
                    UPDATE assignment_submissions
                    SET answers_json = ?, score = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (json.dumps(answers), score, now, submission["id"]),
                )
            else:
                db.execute(
                    """
                    INSERT INTO assignment_submissions
                    (assignment_id, user_id, story_id, draft_id,
                     answers_json, score, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        assignment_id,
                        user_id,
                        assignment["story_id"],
                        assignment["draft_id"],
                        json.dumps(answers),
                        score,
                        now,
                        now,
                    ),
                )

            # Increment attempt count and update score in SQL directly
            db.execute(
                """
                UPDATE assignments
                SET status = 'submitted',
                    score = ?,
                    attempt_count = COALESCE(attempt_count, 0) + 1
                WHERE id = ?
                """,
                (score, assignment_id),
            )
            db.commit()

            # Reload updated assignment & latest submission so templates see new score + attempts
            assignment = db.execute(
                """
                SELECT * FROM assignments
                WHERE id = ? AND assignee_id = ?
                """,
                (assignment_id, user_id),
            ).fetchone()

            submission = db.execute(
                """
                SELECT *
                FROM assignment_submissions
                WHERE assignment_id = ? AND user_id = ?
                ORDER BY datetime(updated_at) DESC
                LIMIT 1
                """,
                (assignment_id, user_id),
            ).fetchone()

            # Stay on MCQ page and show "Well done" modal
            return render_template(
                "assignment_mcq.html",
                assignment=assignment,
                story=story,
                questions=questions,
                submission=submission,
                just_submitted=True,
            )

        # GET: first load / coming back from list
        return render_template(
            "assignment_mcq.html",
            assignment=assignment,
            story=story,
            questions=questions,
            submission=submission,
            just_submitted=False,
        )

    # ------------------------------------------------------------------
    # Fallback: unknown type
    # ------------------------------------------------------------------
    flash("Unknown assignment type.", "warning")
    return redirect(url_for("assignments_list"))




# -------------------------------------------------------------------
# Story generation helpers
# -------------------------------------------------------------------
def _reading_prefs_from_profile(profile: dict, explicit_level: str | None):
    """
    Decide phonics / early-reader base level from age.
    explicit_level is something like "phonics", "early-reader", or "" (auto).
    """
    level = (explicit_level or "phonics").strip().lower()

    age = profile.get("age")
    is_native = profile.get("is_english_native")

    if explicit_level not in ("custom",):
        if age is not None:
            if age <= 7:
                level = "phonics"
            elif 8 <= age <= 10:
                level = "early-reader"

    notes = []
    if is_native is False:
        notes += [
            "Prefer high-frequency, decodable words; avoid idioms and slang.",
            "Keep sentences short (≤10–12 words) and concrete.",
            "Rephrase rare words with simpler synonyms.",
        ]
    if age is not None and age <= 7:
        notes += [
            "Use clear repetition and predictable patterns.",
            "One action per sentence; present tense preferred.",
        ]
    elif age is not None and 8 <= age <= 10:
        notes += [
            "Keep sentences simple (8–12 words) with occasional compound sentences.",
            "Use concrete details and gentle cause-effect.",
        ]

    return level, notes


def naive_story_from_prompt(prompt: str, language: str = "en") -> str:
    """
    Fallback English-only story if the LLM is unavailable.
    """
    return (
        "We’ll write a longer, simple story using your words. "
        "The hero practices sounds with friends, tries again after small mistakes, "
        "and speaks more clearly with each step. The day changes, "
        "little goals appear, and confidence grows. In the end, "
        "the hero uses today’s sounds in real life."
    )


def llm_story_from_prompt(
    prompt: str,
    language: str,
    level: str,
    author: str,
    learner_profile: dict | None = None,
) -> str:
    # We always generate in English
    language = "en"

    if client is None:
        return naive_story_from_prompt(prompt, language)

    profile = learner_profile or {}
    level, pref_notes = _reading_prefs_from_profile(profile, level)

    gender = (profile.get("gender") or "").lower()
    if gender == "male":
        pronoun_hint = "Use neutral narration; if pronouns appear, 'he/him' is acceptable but keep inclusive tone."
    elif gender == "female":
        pronoun_hint = "Use neutral narration; if pronouns appear, 'she/her' is acceptable but keep inclusive tone."
    elif gender == "nonbinary":
        pronoun_hint = "Use neutral narration; if pronouns appear, prefer 'they/them' without making gender a theme."
    else:
        pronoun_hint = "Use neutral narration; avoid making gender a theme."

    level_note = {
        "phonics": "Very short sentences; repeat target sounds; decodable words; high pictureability.",
        "early-reader": "Simple sentences (8–12 words); concrete vocabulary; mild variety.",
        "custom": "Neutral elementary reading level unless the prompt implies otherwise.",
    }.get(level, "Use simple sentences.")

    lang_note = "Write entirely in English."

    extra_scaffold = ""
    if profile.get("is_english_native") is False:
        extra_scaffold = (
            "Use Tier-1/Tier-2 vocabulary; define any rare word in-line via easy context, not parentheses. "
        )

    system = (
        "You are a children's story generator for phonics & early readers. "
        "IMPORTANT: Do not use labels like [Beginning] or section headings. "
        "Write a single continuous story without metadata."
    )

    user = (
        f"Author/Learner: {author or 'guest'}\n"
        f"Target words/phonics: {prompt}\n"
        f"Level: {level}\n"
        f"{level_note}\n"
        f"{lang_note}\n"
        f"{pronoun_hint}\n"
        f"{extra_scaffold}"
        "The content of the story book should be strongly related to the given target words and the story title.\n"
        "Length: 180–260 words.\n"
        "Keep sentences short and decodable; use gentle repetition; warm tone; hopeful ending.\n"
        "Personalization notes:\n- " + "\n- ".join(pref_notes)
    )

    resp = client.responses.create(
        model=DEFAULT_MODEL,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.4,
        max_output_tokens=800,
    )
    text = getattr(resp, "output_text", "") or ""
    return text.strip() or naive_story_from_prompt(prompt, language)

# -------------------------------------------------------------------
# Simple vocab helpers
# -------------------------------------------------------------------
def parse_vocab_from_prompt(prompt: str) -> list[str]:
    raw = re.split(r"[,\n;]+", prompt)
    words = []
    for tok in raw:
        t = re.sub(r"[^A-Za-z0-9' _-]+", "", tok).strip()
        if not t:
            continue
        if len(t) <= 1:
            continue
        words.append(t.lower())
    seen, out = set(), []
    for w in words:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out[:12]


def kid_def_fallback(word: str) -> tuple[str, str]:
    d = "a simple word used in this story"
    e = f"I can read the word '{word}' in the story."
    return d, e


def simple_bilingual_defs(words: list[str]) -> list[dict]:
    out = []
    for w in words:
        d_en, e_en = kid_def_fallback(w)
        d_ko, e_ko = d_en, e_en
        out.append(
            {
                "word": w,
                "definition_en": d_en,
                "example_en": e_en,
                "definition_ko": d_ko,
                "example_ko": e_ko,
            }
        )
    return out

# -------------------------------------------------------------------
# Finish drafts helper
# -------------------------------------------------------------------
def make_partial_from_story(full_text: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", full_text.strip())
    if len(sentences) < 4:
        keep = sentences[: max(1, int(len(sentences) * 0.75))]
    else:
        keep = sentences[: max(3, int(len(sentences) * 0.8))]
    partial = " ".join(keep).strip()
    if partial and not partial.endswith((".", "!", "?")):
        partial += "."
    partial += "\n\n"
    return partial

# -------------------------------------------------------------------
# Routes: index, story_new + library/finish view
# -------------------------------------------------------------------
@app.get("/")
def index():
    db = get_db()
    stories = db.execute(
        "SELECT * FROM stories ORDER BY datetime(created_at) DESC LIMIT 5"
    ).fetchall()
    finishes = db.execute(
        "SELECT * FROM finish_drafts ORDER BY datetime(created_at) DESC LIMIT 5"
    ).fetchall()
    return render_template("index.html", stories=stories, finishes=finishes)


@app.route("/story/new", methods=["GET", "POST"])
@login_required
def story_new():
    db = get_db()
    if request.method == "POST":
        title = (request.form.get("title") or "").strip() or "My Story"
        prompt = (request.form.get("prompt") or "").strip()
        language = "en"

        english_level = (request.form.get("english_level") or "beginner").strip().lower()

        base_author = (
            g.current_user["username"]
            if (g.current_user and g.current_user.get("username"))
            else None
        ) or (request.form.get("author_name") or "guest")

        want_image = bool(request.form.get("gen_image"))

        story_type = (request.form.get("story_type") or "").strip()
        setting = (request.form.get("setting") or "").strip()
        character_kind = (request.form.get("character_kind") or "").strip()
        emotion_tone = (request.form.get("emotion_tone") or "").strip()
        tense = (request.form.get("tense") or "").strip()

        student_finish = bool(request.form.get("student_finish"))

        if not prompt:
            flash("Please provide phonics letters or vocabulary.", "warning")
            return redirect(url_for("story_new"))

        meta_bits = []

        if english_level:
            meta_bits.append(
                f"English level: {english_level} for young learners. "
                f"Use vocabulary and sentence patterns that match a {english_level} elementary student."
            )

        if story_type:
            meta_bits.append(
                f"Story type: {story_type}. Make the overall plot and events match this type."
            )

        if setting:
            meta_bits.append(
                f"Main setting: {setting}. Most scenes should happen in this place."
            )

        if character_kind:
            meta_bits.append(
                f"Main characters: {character_kind}. Use this kind of character as the focus of the story."
            )

        if emotion_tone:
            meta_bits.append(
                f"Emotional tone: {emotion_tone}. The story should feel like this overall."
            )

        if tense:
            meta_bits.append(
                f"Tense: {tense}. Keep the narration consistently in this tense as much as possible."
            )

        meta_bits.append(
            "Remember this is for young students learning English, so keep sentences clear, short, and supportive."
        )
        meta_bits.append(
            f"The title of the story is '{title}', and the content should strongly relate to this title and the target phonics/vocabulary."
        )

        gen_prompt = prompt if not meta_bits else (prompt + "\n\n" + " ".join(meta_bits))

        reading_level_for_llm = ""

        try:
            profile = get_learner_profile()
            content = llm_story_from_prompt(
                gen_prompt,
                language,
                reading_level_for_llm,
                base_author,
                learner_profile=profile,
            )
        except Exception as e:
            content = naive_story_from_prompt(prompt, language)
            flash("AI generator had an issue; used a fallback story.", "warning")
            log_input("generate_story_error", {"error": str(e)})

        visuals_data_url = None
        if want_image and client is not None:
            try:
                img_prompt = (
                    "Kid-friendly, text-free cover illustration for a children's story. "
                    "Soft colors, simple shapes, clear subject, warm tone. "
                    "No words or letters in the image.\n\n"
                    f"Story excerpt:\n{content[:1200]}"
                )
                img = client.images.generate(
                    model="gpt-image-1",
                    prompt=img_prompt,
                    size="1024x1024",
                    n=1,
                )
                b64 = img.data[0].b64_json
                visuals_data_url = f"data:image/png;base64,{b64}"
            except Exception as e:
                log_input("generate_image_error", {"error": str(e)})
                flash("Story created, but image generation had an issue.", "warning")

        story_author = base_author
        if not student_finish:
            story_author = "EduWeaver AI"

        slug_base = slugify(title) or "story"
        ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        slug = f"{slug_base}-{ts}"

        db_level = english_level or "beginner"

        db.execute(
            """
            INSERT INTO stories (title, slug, prompt, language, level, content, visuals, author_name)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                title,
                slug,
                prompt,
                language,
                db_level,
                content,
                visuals_data_url,
                story_author,
            ),
        )
        story_id = db.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]
        db.commit()

        vocab_words = parse_vocab_from_prompt(prompt)
        vocab_defs = simple_bilingual_defs(vocab_words)
        for item in vocab_defs:
            db.execute(
                """
                INSERT INTO vocab_items (story_id, word, definition, example, definition_ko, example_ko, picture_url)
                VALUES (?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    story_id,
                    item["word"].lower(),
                    item.get("definition_en") or "",
                    item.get("example_en") or "",
                    item.get("definition_ko") or "",
                    item.get("example_ko") or "",
                ),
            )
        db.commit()

        log_input(
            "generate_story",
            {
                "prompt": prompt,
                "language": language,
                "english_level": english_level,
                "request_user": base_author,
                "db_author": story_author,
                "model": DEFAULT_MODEL,
                "vocab_count": len(vocab_defs),
                "with_image": want_image,
                "story_type": story_type,
                "setting": setting,
                "character_kind": character_kind,
                "emotion_tone": emotion_tone,
                "tense": tense,
                "student_finish": student_finish,
            },
        )

        if student_finish:
            partial = make_partial_from_story(content)
            db.execute(
                """
                INSERT INTO finish_drafts
                (story_id, seed_prompt, partial_text, learner_name, language)
                VALUES (?, ?, ?, ?, ?)
                """,
                (story_id, prompt, partial, story_author, language),
            )
            draft_id = db.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]
            db.commit()

            log_input(
                "finish_seed_auto_from_story_new",
                {"story_id": story_id, "slug": slug, "draft_id": draft_id},
            )

            return redirect(
                url_for(
                    "story_new",
                    generated=1,
                    finish_url=url_for("finish_view", draft_id=draft_id),
                )
            )
        else:
            partial = make_partial_from_story(content)
            db.execute(
                """
                INSERT INTO finish_drafts
                (story_id, seed_prompt, partial_text, learner_name, language, completion_text)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (story_id, prompt, partial, story_author, language, content),
            )
            draft_id = db.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]
            db.commit()

            log_input(
                "finish_seed_auto_from_story_new_full_ai",
                {
                    "story_id": story_id,
                    "slug": slug,
                    "draft_id": draft_id,
                    "auto_completed": True,
                    "db_author": story_author,
                },
            )

            flash(
                "Story generated by EduWeaver AI and added to the Library.", "success"
            )
            return redirect(url_for("library"))

    return render_template("story_new.html")


# In app2.py, inside the library() function:

@app.get("/library")
@login_required
def library():
    """
    Library view for students:
    shows only stories that have been explicitly shared to the Library
    (stories.is_shared_library = 1), with their latest draft (finished or not).
    """
    db = get_db()
    rows = db.execute(
        """
        SELECT
          s.id          AS story_id,
          s.title       AS title,
          s.slug        AS slug,
          s.level       AS level,
          s.language    AS language,
          s.author_name AS author_name,
          s.created_at  AS story_created_at,
          s.visuals     AS visuals,  -- ADDED visuals
          fd.id         AS draft_id,
          fd.created_at AS draft_created_at
        FROM stories s
        JOIN finish_drafts fd
          ON fd.story_id = s.id
         AND datetime(fd.created_at) = (
            SELECT MAX(datetime(fd2.created_at))
            FROM finish_drafts fd2
            WHERE fd2.story_id = s.id
         )
        WHERE s.is_shared_library = 1
        ORDER BY datetime(s.created_at) DESC
        """
    ).fetchall()

    # Inject admin status for the template
    is_admin = current_user_is_admin()

    return render_template("library.html", stories=rows, is_admin=is_admin)

# --- app2.py addition ---

# --- app2.py addition (Place near /finish/<int:draft_id> route) ---

@app.get("/book/<int:draft_id>")
@login_required
def book_view(draft_id: int):
    """
    Reader view for a fully completed story draft,
    displayed with a book-like interface.
    """
    db = get_db()

    draft = db.execute(
        "SELECT * FROM finish_drafts WHERE id = ?", (draft_id,)
    ).fetchone()
    if not draft:
        flash("Story draft not found.", "warning")
        return redirect(url_for("library"))

    story = db.execute(
        "SELECT * FROM stories WHERE id = ?", (draft["story_id"],)
    ).fetchone()
    if not story:
        flash("Linked story not found.", "warning")
        return redirect(url_for("library"))

    full_content = (draft.get("completion_text") or story.get("content") or "").strip()
    if not full_content:
        flash("This story is unfinished or empty.", "warning")
        return redirect(url_for("library"))

    vocab = db.execute(
        "SELECT * FROM vocab_items WHERE story_id = ? ORDER BY word COLLATE NOCASE",
        (story["id"],),
    ).fetchall()

    # Each paragraph (split by blank line) = one page.
    pages = [p.strip() for p in full_content.split("\n\n") if p.strip()]
    pages = pages[1:]
    return render_template(
        "book_view.html",
        draft=draft,
        story=story,
        pages=pages,
        vocab=vocab,
    )

# --- End app2.py addition ---


@app.get("/finish/<int:draft_id>")
@login_required
def finish_view(draft_id: int):
    db = get_db()
    draft = db.execute(
        "SELECT * FROM finish_drafts WHERE id = ?", (draft_id,)
    ).fetchone()
    if not draft:
        flash("Draft not found.", "warning")
        return redirect(url_for("library"))
    return render_template("finish_view.html", draft=draft)

# -------------------------------------------------------------------
# ADMIN: story review pages
# -------------------------------------------------------------------
@app.get("/admin/stories")
@login_required
def admin_stories():
    """Admin-only list of all generated stories."""
    if not current_user_is_admin():
        flash("Admin access only.", "warning")
        return redirect(url_for("index"))

    db = get_db()
    # All stories
    stories = db.execute(
        """
        SELECT * FROM stories
        ORDER BY datetime(created_at) DESC
        """
    ).fetchall()

    # Latest draft per story
    draft_rows = db.execute(
        """
        SELECT fd.*
        FROM finish_drafts fd
        JOIN (
          SELECT story_id, MAX(datetime(created_at)) AS max_created
          FROM finish_drafts
          GROUP BY story_id
        ) t
        ON t.story_id = fd.story_id AND t.max_created = fd.created_at
        """
    ).fetchall()
    draft_map = {d["story_id"]: d for d in draft_rows}

    # How many assignments per story
    assign_rows = db.execute(
        """
        SELECT story_id, COUNT(*) AS cnt
        FROM assignments
        GROUP BY story_id
        """
    ).fetchall()
    assign_map = {r["story_id"]: r["cnt"] for r in assign_rows}

    # All students (exclude admin user "testtest")
    users = db.execute(
        """
        SELECT id, username, email
        FROM users
        WHERE username != 'testtest'
        ORDER BY username COLLATE NOCASE
        """
    ).fetchall()

    return render_template(
        "admin_stories.html",
        stories=stories,
        draft_map=draft_map,
        users=users,
        assign_map=assign_map,
    )
@app.post("/admin/stories/<slug>/generate-mcq")
@login_required
def admin_generate_mcq(slug: str):
    """Generate and save MCQ questions for a completed story (admin only)."""
    if not current_user_is_admin():
        flash("Admin access only.", "warning")
        return redirect(url_for("index"))

    db = get_db()

    story = db.execute(
        "SELECT * FROM stories WHERE slug = ?",
        (slug,),
    ).fetchone()
    if not story:
        flash("Story not found.", "warning")
        return redirect(url_for("admin_stories"))

    # Use the latest *completed* draft for MCQ generation
    draft = db.execute(
        """
        SELECT *
        FROM finish_drafts
        WHERE story_id = ? AND completion_text IS NOT NULL
        ORDER BY datetime(created_at) DESC
        LIMIT 1
        """,
        (story["id"],),
    ).fetchone()

    if not draft:
        flash(
            "No fully completed draft found for this story. MCQ questions require a finished text.",
            "warning",
        )
        return redirect(url_for("admin_stories"))

    base_text = (draft.get("completion_text") or story.get("content") or "").strip()
    if not base_text:
        flash("Story text is empty; cannot generate MCQ questions.", "warning")
        return redirect(url_for("admin_stories"))

    questions = generate_mcq_questions(base_text, num_questions=5)
    if not questions:
        flash("Could not generate MCQ questions. Please try again.", "danger")
        return redirect(url_for("admin_stories"))

    db.execute(
        "UPDATE stories SET mcq_questions_json = ? WHERE id = ?",
        (json.dumps(questions, ensure_ascii=False), story["id"]),
    )
    db.commit()

    flash("MCQ questions generated and saved for this story.", "success")
    return redirect(url_for("admin_stories"))
@app.post("/admin/stories/<slug>/share")
@login_required
def admin_share_story(slug):
    # simple admin gate you already use
    if session["username"] != "testtest":
        flash("Admin only area.", "warning")
        return redirect(url_for("index"))

    db = get_db()

    # get story including visuals
    story = db.execute(
        """
        SELECT id, title, language, level, prompt, visuals
        FROM stories
        WHERE slug = ?
        """,
        (slug,),
    ).fetchone()

    if not story:
        flash("Story not found.", "warning")
        return redirect(url_for("admin_stories"))

    visuals_data_url = story["visuals"]

    # if no cover image yet, generate one first
    if not visuals_data_url:
        try:
            import base64  # local import is fine

            title = story["title"] or "Untitled story"
            lang = (story["language"] or "").upper() or "EN"
            level = story["level"] or "beginner"
            prompt_text = (story["prompt"] or "").strip()

            # keep the prompt short for the image model
            if len(prompt_text) > 220:
                prompt_text = prompt_text[:220].rstrip() + "…"

            image_prompt = (
                "Children's picture-book cover illustration for a story.\n"
                f'Title: "{title}".\n'
                f"Language: {lang}, Level: {level}.\n"
                "Style: warm, friendly, simple shapes, soft pastel colors, "
                "one main character in the center, minimal background, "
                "no text or title on the image.\n"
            )
            if prompt_text:
                image_prompt += f"Story idea: {prompt_text}\n"

            img_resp = client.images.generate(
                model="gpt-image-1",
                prompt=image_prompt,
                size="1024x1024",
                n=1,
            )

            img_b64 = img_resp.data[0].b64_json
            visuals_data_url = "data:image/png;base64," + img_b64

            # save cover into stories.visuals
            db.execute(
                "UPDATE stories SET visuals = ? WHERE id = ?",
                (visuals_data_url, story["id"]),
            )
            db.commit()

        except Exception as e:
            # optional: log for debugging
            print("Error generating cover image for library share:", e)
            flash(
                "Could not generate a cover image for this story. "
                "Please try again in a moment.",
                "danger",
            )
            return redirect(url_for("admin_stories"))

    # at this point we are sure visuals exists, so we can safely share
    db.execute(
        "UPDATE stories SET is_shared_library = 1 WHERE id = ?",
        (story["id"],),
    )
    db.commit()

    flash("Story shared to Library.", "success")
    return redirect(url_for("admin_stories"))


@app.post("/admin/stories/<slug>/unshare")
@login_required
def admin_unshare_story(slug: str):
    if not current_user_is_admin():
        flash("Admin access only.", "warning")
        return redirect(url_for("index"))
    db = get_db()
    db.execute("UPDATE stories SET is_shared_library = 0 WHERE slug = ?", (slug,))
    db.commit()
    flash("Story removed from Library.", "info")
    return redirect(url_for("admin_stories"))



@app.get("/admin/stories/<slug>")
@login_required
def admin_story_detail(slug: str):
    if not current_user_is_admin():
        flash("Admin access only.", "warning")
        return redirect(url_for("index"))

    db = get_db()
    story = db.execute(
        "SELECT * FROM stories WHERE slug = ?", (slug,)
    ).fetchone()
    if not story:
        flash("Story not found.", "warning")
        return redirect(url_for("admin_stories"))

    vocab = db.execute(
        """
        SELECT * FROM vocab_items
        WHERE story_id = ?
        ORDER BY word COLLATE NOCASE
        """,
        (story["id"],),
    ).fetchall()

    draft = db.execute(
        """
        SELECT * FROM finish_drafts
        WHERE story_id = ?
        ORDER BY datetime(created_at) DESC
        LIMIT 1
        """,
        (story["id"],),
    ).fetchone()

    return render_template(
        "admin_story_detail.html",
        story=story,
        vocab=vocab,
        draft=draft,
    )

# -------------------------------------------------------------------
# WORD SCRAMBLE GAME
# -------------------------------------------------------------------
def get_words_for_student_scramble(user_id: int,
                                   min_words: int = 6,
                                   max_words: int = 12) -> list[dict]:
    """
    Get a list of words for the word scramble game for this student.

    Priority:
      1) Distinct vocab words from stories that were assigned to this user
      2) If not enough, fill with PREPARED_SCRAMBLE_WORDS

    Returns a list of dicts: { "word": "apple" }
    """
    db = get_db()

    rows = db.execute(
        """
        SELECT DISTINCT LOWER(v.word) AS word
        FROM vocab_items v
        JOIN stories s   ON v.story_id = s.id
        JOIN assignments a ON a.story_id = s.id
        WHERE a.assignee_id = ?
        ORDER BY v.word COLLATE NOCASE
        """,
        (user_id,),
    ).fetchall()

    words = [r["word"] for r in rows if r.get("word")]

    seen = set(words)
    if len(words) < min_words:
        for w in PREPARED_SCRAMBLE_WORDS:
            lw = w.lower()
            if lw not in seen:
                words.append(lw)
                seen.add(lw)
                if len(words) >= min_words:
                    break

    random.shuffle(words)
    words = words[:max_words]

    return [{"word": w} for w in words]


def scramble_word(word: str) -> str:
    """
    Scramble letters in a word. Try to avoid returning the word unchanged.
    """
    w = word.strip()
    if len(w) < 3:
        return w

    chars = list(w)
    scrambled = w
    attempts = 0
    while scrambled.lower() == w.lower() and attempts < 10:
        random.shuffle(chars)
        scrambled = "".join(chars)
        attempts += 1
    return scrambled


@app.get("/word-scramble")
@login_required
def word_scramble():
    user_id = g.current_user["id"]
    words = get_words_for_student_scramble(user_id)

    puzzles = []
    for idx, item in enumerate(words):
        w = item["word"]
        scrambled = scramble_word(w)
        puzzles.append(
            {
                "id": idx,
                "word": w,
                "scrambled": scrambled,
            }
        )

    return render_template("word_scramble.html", puzzles=puzzles)
@app.post("/api/scramble_log")
def scramble_log():
    db = get_db()
    data = request.get_json() or {}

    user_id = session.get("user_id")
    word = data.get("word")
    is_correct = 1 if data.get("is_correct") else 0

    db.execute(
        "INSERT INTO scramble_logs (user_id, word, is_correct) VALUES (?, ?, ?)",
        (user_id, word, is_correct)
    )
    db.commit()

    return {"ok": True}
@app.post("/api/scramble_session")
def scramble_session():
    db = get_db()
    data = request.get_json() or {}
    user_id = session.get("user_id")

    db.execute(
        "INSERT INTO scramble_sessions (user_id, correct_count, wrong_count, total)"
        " VALUES (?, ?, ?, ?)",
        (user_id, data["correct"], data["wrong"], data["total"])
    )
    db.commit()

    return {"ok": True}

import requests

GOOGLE_TRANSLATE_API_KEY = "AIzaSyCG14vrQaBjCyidFq_xZKClZCe1U7CdkWA"

def translate_en_ko_or_ko_en(text: str, direction: str) -> dict:
    """
    Uses Google Translation API (REST) for EN <-> KO.
    API key is hardcoded intentionally for now.
    Returns: {headword, translation, example_en, example_ko}
    """
    text = (text or "").strip()
    if not text:
        return {
            "headword": "",
            "translation": "",
            "example_en": "",
            "example_ko": "",
        }

    # Normalize direction
    if direction not in ("en_ko", "ko_en"):
        direction = "en_ko"

    target = "ko" if direction == "en_ko" else "en"

    url = f"https://translation.googleapis.com/language/translate/v2?key={GOOGLE_TRANSLATE_API_KEY}"
    data = {
        "q": text,
        "target": target,
        "format": "text"
    }

    try:
        res = requests.post(url, json=data)
        res.raise_for_status()
        translated = res.json()["data"]["translations"][0]["translatedText"]
    except Exception as e:
        print("Google Translate API error:", e)
        translated = text  # fallback

    return {
        "headword": text,
        "translation": translated,
        "example_en": "",
        "example_ko": "",
    }

@app.post("/api/dict/search")
@login_required
def api_dict_search():
    """
    POST JSON:
      - { "query": "...", "direction": "en_ko"|"ko_en" }
      - OR { "lookup_id": 123 } to replay a previous lookup.

    Response JSON:
      {
        "ok": true,
        "lookup_id": int,
        "query": "...",
        "direction": "en_ko",
        "result": {
          "headword": "...",
          "translation": "...",
          "example_en": "...",
          "example_ko": "..."
        },
        "bookmarked": true|false
      }
    """
    db = get_db()
    uid = session["user_id"]
    data = request.get_json(silent=True) or {}

    lookup_id = data.get("lookup_id")
    query = (data.get("query") or "").strip()
    direction = (data.get("direction") or "en_ko").strip()

    # --- 1) Replay existing lookup by id ---
    if lookup_id:
        row = db.execute(
            """
            SELECT id, query, direction, source_lang, target_lang,
                   headword, translation, example_en, example_ko
            FROM dict_lookups
            WHERE id = ? AND user_id = ?
            """,
            (lookup_id, uid),
        ).fetchone()

        if not row:
            return jsonify({"ok": False, "error": "Lookup not found."}), 404

        # Check bookmark status
        b = db.execute(
            "SELECT 1 FROM dict_bookmarks WHERE user_id = ? AND lookup_id = ?",
            (uid, row["id"]),
        ).fetchone()
        bookmarked = bool(b)

        return jsonify({
            "ok": True,
            "lookup_id": row["id"],
            "query": row["query"],
            "direction": row["direction"],
            "result": {
                "headword": row["headword"],
                "translation": row["translation"],
                "example_en": row["example_en"] or "",
                "example_ko": row["example_ko"] or "",
            },
            "bookmarked": bookmarked,
        })

    # --- 2) New lookup by query ---
    if not query:
        return jsonify({"ok": False, "error": "Empty query."}), 400

    if direction not in ("en_ko", "ko_en"):
        direction = "en_ko"

    source_lang = "en" if direction == "en_ko" else "ko"
    target_lang = "ko" if direction == "en_ko" else "en"

    # Call Google Translate helper
    r = translate_en_ko_or_ko_en(query, direction)

    # Insert into dict_lookups (including source_lang/target_lang)
    cur = db.execute(
        """
        INSERT INTO dict_lookups
        (user_id, query, direction, source_lang, target_lang,
         headword, translation, example_en, example_ko)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            uid,
            query,
            direction,
            source_lang,
            target_lang,
            r["headword"],
            r["translation"],
            r["example_en"],
            r["example_ko"],
        ),
    )
    db.commit()
    lookup_id = cur.lastrowid

    # New lookups are not bookmarked by default
    return jsonify({
        "ok": True,
        "lookup_id": lookup_id,
        "query": query,
        "direction": direction,
        "result": r,
        "bookmarked": False,
    })

@app.get("/api/dict/history")
@login_required
def api_dict_history():
    """
    Return recent lookups for the logged-in user.
    Response:
      { "items": [ {id, query, direction, direction_label}, ... ] }
    """
    db = get_db()
    uid = session["user_id"]

    rows = db.execute(
        """
        SELECT id, query, direction, created_at
        FROM dict_lookups
        WHERE user_id = ?
        ORDER BY created_at DESC, id DESC
        LIMIT 50
        """,
        (uid,),
    ).fetchall()

    items = []
    for r in rows:
        if r["direction"] == "ko_en":
            label = "KO → EN"
        else:
            label = "EN → KO"
        items.append({
            "id": r["id"],
            "query": r["query"],
            "direction": r["direction"],
            "direction_label": label,
        })

    return jsonify({"items": items})


@app.post("/api/dict/clear_history")
@login_required
def api_dict_clear_history():
    """
    Clear all dict lookups (and related bookmarks) for current user.
    """
    db = get_db()
    uid = session["user_id"]

    # Delete bookmarks first (to satisfy FK if you add it later)
    db.execute("DELETE FROM dict_bookmarks WHERE user_id = ?", (uid,))
    db.execute("DELETE FROM dict_lookups WHERE user_id = ?", (uid,))
    db.commit()

    return jsonify({"ok": True})

@app.get("/api/dict/bookmarks")
@login_required
def api_dict_bookmarks():
    """
    Return bookmarked lookups for the current user.
    Response:
      { "items": [ {id, query, translation}, ... ] }
    Where id is the lookup_id (used by JS to toggle).
    """
    db = get_db()
    uid = session["user_id"]

    rows = db.execute(
        """
        SELECT l.id AS id, l.query, l.translation
        FROM dict_lookups AS l
        JOIN dict_bookmarks AS b ON l.id = b.lookup_id
        WHERE b.user_id = ?
        ORDER BY b.created_at DESC, b.id DESC
        """,
        (uid,),
    ).fetchall()

    items = [
        {
            "id": r["id"],
            "query": r["query"],
            "translation": r["translation"],
        }
        for r in rows
    ]

    return jsonify({"items": items})

@app.post("/api/dict/toggle_bookmark")
@login_required
def api_dict_toggle_bookmark():
    """
    Toggle bookmark for a given lookup_id.
    POST JSON: { "lookup_id": int }
    Response:
      { "ok": true, "bookmarked": true|false }
    """
    db = get_db()
    uid = session["user_id"]
    data = request.get_json(silent=True) or {}
    lookup_id = data.get("lookup_id")

    if not lookup_id:
        return jsonify({"ok": False, "error": "Missing lookup_id"}), 400

    # Check that lookup belongs to this user
    row = db.execute(
        "SELECT id FROM dict_lookups WHERE id = ? AND user_id = ?",
        (lookup_id, uid),
    ).fetchone()
    if not row:
        return jsonify({"ok": False, "error": "Lookup not found."}), 404

    # Toggle
    existing = db.execute(
        "SELECT id FROM dict_bookmarks WHERE user_id = ? AND lookup_id = ?",
        (uid, lookup_id),
    ).fetchone()

    if existing:
        db.execute(
            "DELETE FROM dict_bookmarks WHERE user_id = ? AND lookup_id = ?",
            (uid, lookup_id),
        )
        db.commit()
        return jsonify({"ok": True, "bookmarked": False})

    db.execute(
        """
        INSERT OR IGNORE INTO dict_bookmarks (user_id, lookup_id)
        VALUES (?, ?)
        """,
        (uid, lookup_id),
    )
    db.commit()
    return jsonify({"ok": True, "bookmarked": True})


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

def upgrade_bookmarks_table():
    db = get_db()
    cols = [c["name"] for c in db.execute("PRAGMA table_info(dict_bookmarks)").fetchall()]

    if "lookup_id" not in cols:
        db.execute("ALTER TABLE dict_bookmarks ADD COLUMN lookup_id INTEGER")

    if "created_at" not in cols:
        db.execute("ALTER TABLE dict_bookmarks ADD COLUMN created_at TEXT")

    db.commit()
    print("Bookmarks table upgraded!")

if __name__ == "__main__":
    with app.app_context():
        init_db()
        upgrade_bookmarks_table()
    app.run(debug=True)
