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
    # ---- EASY (4 letters) ----
    "moon", "star", "book", "tree", "bird",
    "frog", "lion", "bear", "snow", "wind",
    "rain", "milk", "play", "jump", "swim",
    "blue", "fire", "leaf", "home", "song",

    # ---- MEDIUM (5 letters) ----
    "apple", "happy", "smile", "magic", "piano",
    "water", "tiger", "river", "ocean", "dream",
    "cloud", "light", "story", "bread", "green",
    "dance", "sweet", "world", "heart", "sound",

    # ---- HARD (6+ letters) ----
    "banana", "family", "forest", "cookie", "dragon",
    "school", "pencil", "friend", "rocket", "garden",
    "rainbow", "castle", "planet", "purple", "silver",
    "butter", "animal", "market", "window", "orange",
    "reading", "puzzle", "bubble", "winter", "summer",
    "island", "jungle", "wonder", "travel", "flower",
    "bridge", "adventure", "sunshine", "mountain", "treasure",
    "chicken", "station", "picture", "laughter", "morning",
    "evening", "holiday", "science", "teacher", "library",

    # ---- EXTRA HARD (longer & fun) ----
    "elephant", "kangaroo", "chocolate", "beautiful", "starlight",
    "painting", "storybook", "discovery", "friendship", "imagination",
    "happiness", "backpack", "spaceship", "playground", "fireworks",
    "waterfall", "adventure", "butterfly", "mountains", "snowflake"
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

from markupsafe import Markup, escape

@app.template_filter("nl2br")
def nl2br(value: str) -> Markup:
    """Convert newlines to <br> tags, with HTML escaping."""
    if not value:
        return Markup("")
    # Escape HTML, then replace newline chars with <br>
    return Markup(escape(value).replace("\n", Markup("<br>\n")))
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
        -- ... (existing dict_lookups table creation)

        -- Dictionary bookmarks
        -- ... (existing dict_bookmarks table creation)

        -- All other CREATE TABLE statements are managed elsewhere.
        -- This function is safe to call multiple times and acts as a migration.
        """
    )
    # ... (existing ALTER TABLE statements)

    # Store MCQ questions once per story
    try:
        db.execute(
            "ALTER TABLE stories ADD COLUMN mcq_questions_json TEXT"
        )
    except sqlite3.OperationalError:
        # Column already exists, ignore
        pass

    # ... (rest of existing migrations)

    # ------------------------------------------------------------------
    # NEW MIGRATIONS FOR ADMIN REVIEW FEATURE: assignment_submissions
    # The traceback indicates these columns are missing.
    # ------------------------------------------------------------------

    # Add column for Admin Comment/Feedback to submissions
    try:
        db.execute(
            "ALTER TABLE assignment_submissions ADD COLUMN comment TEXT"
        )
    except sqlite3.OperationalError:
        # Column already exists, ignore
        pass

    # Add column to track when the submission was reviewed
    try:
        db.execute(
            "ALTER TABLE assignment_submissions ADD COLUMN reviewed_at TEXT"
        )
    except sqlite3.OperationalError:
        # Column already exists, ignore
        pass

    # ------------------------------------------------------------------

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
    # Admin is the special username "adminlexi"
    if session.get("username") == "testtest":
        return True
    return False

# -------------------------------------------------------------------
# Auth routes
# -------------------------------------------------------------------

@app.post("/register/check-step1")
def register_check_step1():
    """
    AJAX endpoint to validate Step 1:
    - required fields
    - basic format
    - duplicate username/email
    Returns JSON {ok: bool, errors: [..]}.
    """
    db = get_db()
    data = request.get_json(force=True) or {}

    username = (data.get("username") or "").strip()
    email    = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    confirm  = data.get("confirm") or ""
    l1       = (data.get("l1_language") or "").strip().lower()
    l2       = (data.get("l2_language") or "").strip().lower()
    age_raw  = (data.get("age") or "").strip()
    gender   = (data.get("gender") or "").strip().lower()

    errors = []

    # Required fields
    if not username or not email or not password or not confirm:
        errors.append("Please fill in username, email, and password.")
    if not l1 or not l2 or not age_raw or not gender:
        errors.append("Please fill in L1, L2, age, and gender.")

    # Password checks
    if password != confirm:
        errors.append("Passwords do not match.")
    if len(password) < 8:
        errors.append("Password must be at least 8 characters.")

    # Username format
    if not re.match(r"^[A-Za-z0-9_.-]{3,32}$", username):
        errors.append("Username must be 3–32 chars (letters, numbers, _, ., -).")

    # Email format
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        errors.append("Please enter a valid email address.")

    # L1 / L2
    allowed_l1 = {"korean", "english", "chinese", "japanese", "spanish", "other"}
    allowed_l2 = {"none", "korean", "english", "chinese", "japanese", "spanish", "other"}

    if l1 and l1 not in allowed_l1:
        errors.append("Please choose a valid first language (L1).")
    if l2 and l2 not in allowed_l2:
        errors.append("Please choose a valid second language (L2).")

    # Age
    if not age_raw.isdigit():
        errors.append("Please enter a valid age (number).")
    else:
        age = int(age_raw)
        if age < 5 or age > 120:
            errors.append("Please enter an age between 5 and 120.")

    # Gender
    allowed_genders = {"female", "male", "nonbinary", "prefer_not"}
    if gender and gender not in allowed_genders:
        errors.append("Please choose a valid gender option.")

    # If format errors already, no need to hit DB
    if errors:
        return jsonify({"ok": False, "errors": errors}), 400

    # Duplicate checks
    row = db.execute(
        "SELECT id FROM users WHERE lower(username)=?",
        (username.lower(),),
    ).fetchone()
    if row:
        errors.append("That username is already taken.")

    row = db.execute(
        "SELECT id FROM users WHERE lower(email)=?",
        (email.lower(),),
    ).fetchone()
    if row:
        errors.append("An account with that email already exists.")

    if errors:
        return jsonify({"ok": False, "errors": errors}), 400

    return jsonify({"ok": True})

import re
import sqlite3
from flask import (
    request, render_template, flash, url_for
)
from werkzeug.security import generate_password_hash

@app.route("/register", methods=["GET", "POST"])
def register():
    db = get_db()

    if request.method == "POST":
        # ----- Read fields -----
        username = (request.form.get("username") or "").strip()
        email    = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""
        confirm  = request.form.get("confirm") or ""

        l1_language_raw = (request.form.get("l1_language") or "").strip().lower()
        l2_language_raw = (request.form.get("l2_language") or "").strip().lower()
        age_raw         = (request.form.get("age") or "").strip()
        gender_raw      = (request.form.get("gender") or "").strip().lower()

        level_score_raw = request.form.get("level_score")
        level_name      = (request.form.get("level_name") or "").strip()

        def render_fail(msg: str):
            flash(msg, "warning")
            return render_template(
                "register.html",
                registered=False,
                registered_level_name=None,
                registered_level_score=None,
            )

        # ----- Basic required checks -----
        if not username or not email or not password or not confirm:
            return render_fail("Please fill in username, email, and password.")

        if not l1_language_raw or not l2_language_raw or not age_raw or not gender_raw:
            return render_fail("Please fill in all language and profile fields (L1, L2, age, gender).")

        if not level_score_raw or not level_name:
            return render_fail("Please complete the level test before creating your account.")

        # ----- Password checks -----
        if password != confirm:
            return render_fail("Passwords do not match.")

        if len(password) < 8:
            return render_fail("Password must be at least 8 characters.")

        # ----- Username / email format -----
        if not re.match(r"^[A-Za-z0-9_.-]{3,32}$", username):
            return render_fail("Username must be 3–32 characters (letters, numbers, _, ., -).")

        if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
            return render_fail("Please enter a valid email address.")

        # ----- L1 / L2 sanity -----
        allowed_l1 = {"korean", "english", "chinese", "japanese", "spanish", "other"}
        allowed_l2 = {"none", "korean", "english", "chinese", "japanese", "spanish", "other"}

        if l1_language_raw not in allowed_l1:
            return render_fail("Please choose a valid first language (L1).")

        if l2_language_raw not in allowed_l2:
            return render_fail("Please choose a valid second language (L2).")

        # ----- Age -----
        if not age_raw.isdigit():
            return render_fail("Please enter a valid age (number).")

        age = int(age_raw)
        if age < 5 or age > 120:
            return render_fail("Please enter an age between 5 and 120.")

        # ----- Gender -----
        allowed_genders = {"female", "male", "nonbinary", "prefer_not"}
        if gender_raw not in allowed_genders:
            return render_fail("Please choose a valid gender option.")

        # ----- Level score parsing -----
        try:
            level_score = int(level_score_raw)
        except (TypeError, ValueError):
            return render_fail("Please complete the level test.")

        if level_score < 0 or level_score > 10:
            return render_fail("Level test score is invalid. Please try the test again.")

        # Normalize level name
        valid_levels = {"Beginner", "Intermediate", "Advanced"}
        if level_name not in valid_levels:
            return render_fail("Level test result is invalid. Please try the test again.")

        # ----- Check duplicates -----
        existing = db.execute(
            "SELECT id FROM users WHERE lower(username)=?",
            (username.lower(),),
        ).fetchone()
        if existing:
            flash("That username is already taken.", "danger")
            return render_template(
                "register.html",
                registered=False,
                registered_level_name=None,
                registered_level_score=None,
            )

        existing = db.execute(
            "SELECT id FROM users WHERE lower(email)=?",
            (email.lower(),),
        ).fetchone()
        if existing:
            flash("An account with that email already exists.", "danger")
            return render_template(
                "register.html",
                registered=False,
                registered_level_name=None,
                registered_level_score=None,
            )

        # ----- Insert user + level test result (score NOT NULL) -----
        try:
            pwd_hash = generate_password_hash(password)

            cur = db.execute(
                """
                INSERT INTO users
                  (username, email, password_hash, l1_language, l2_language, age, gender)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (username, email, pwd_hash, l1_language_raw, l2_language_raw, age, gender_raw),
            )
            user_id = cur.lastrowid

            db.execute(
                """
                INSERT INTO level_test_results (user_id, score, total, level)
                VALUES (?, ?, ?, ?)
                """,
                (user_id, level_score, 10, level_name),
            )

            db.commit()

            # Show success modal with level info
            return render_template(
                "register.html",
                registered=True,
                registered_level_name=level_name,
                registered_level_score=level_score,
            )

        except sqlite3.IntegrityError as e:
            db.rollback()
            if "users.username" in str(e):
                flash("That username is already taken.", "danger")
            elif "users.email" in str(e):
                flash("An account with that email already exists.", "danger")
            else:
                flash("Could not create account. Please try again.", "danger")

            return render_template(
                "register.html",
                registered=False,
                registered_level_name=None,
                registered_level_score=None,
            )

    # GET
    return render_template(
        "register.html",
        registered=False,
        registered_level_name=None,
        registered_level_score=None,
    )


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
                # flash(f"Welcome back, {user['username']}!", "success")
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
def generate_mcq_questions(base_text: str, num_questions: int = 5):
    """
    Generate simple 4-option MCQ questions from the base_text.
    Returns a list of dicts:
      [{"question": "...", "options": ["A","B","C","D"], "correct_index": 0}, ...]
    """
    if client is None:
        print("OpenAI client not configured; skipping MCQ generation.")
        return []

    instructions = """
You are a helpful assistant that creates multiple-choice reading comprehension questions
for elementary and middle school students. Given a short story, generate
OBJECTIVE and clear questions that test understanding of the text.

Rules:
- Output ONLY JSON with this structure:
  {
    "questions": [
      {
        "question": "string",
        "options": ["A", "B", "C", "D"],
        "correct_index": 0
      },
      ...
    ]
  }
- Exactly 4 options per question.
- correct_index is 0, 1, 2, or 3 and matches the correct option.
- No explanations, no commentary, no markdown – ONLY JSON.
"""

    # Build user content including the story
    user_prompt = f"""
Story:
\"\"\"{base_text}\"\"\"

Please generate {num_questions} multiple-choice questions that are appropriate for
the student's level. Follow the JSON schema exactly.
"""

    try:
        resp = client.responses.create(
            model=DEFAULT_MODEL,
            max_output_tokens=800,
            temperature=0.4,
            instructions=instructions,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": user_prompt.strip(),
                        }
                    ],
                }
            ],
        )
    except Exception as e:
        print("Error calling OpenAI for MCQ generation:", e)
        return []

    # Safely get the text content
    raw = getattr(resp, "output_text", "") or ""
    raw = raw.strip()
    if not raw:
        print("MCQ generation: empty output_text")
        return []

    # 1) First, try direct JSON parse
    data = None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # 2) Fallback: extract first {...} block
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            print("MCQ generation: no JSON object found in output:", raw[:500])
            return []
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError as e:
            print("MCQ generation: JSON parse failed after extraction:", e, match.group(0)[:500])
            return []

    questions = data.get("questions") or []
    cleaned = []

    for q in questions:
        if not isinstance(q, dict):
            continue

        question_text = (q.get("question") or "").strip()
        options = q.get("options") or []
        correct_index = q.get("correct_index")

        # Normalize options
        options = [str(o).strip() for o in options if str(o).strip()]
        if len(options) != 4:
            # Skip malformed entries so we don't break the UI
            continue

        # Normalize correct_index
        if isinstance(correct_index, str):
            if correct_index.isdigit():
                correct_index = int(correct_index)
            else:
                # if they gave something like "A", map to index
                letter_map = {"A": 0, "B": 1, "C": 2, "D": 3}
                correct_index = letter_map.get(correct_index.upper(), 0)

        if not isinstance(correct_index, int) or not (0 <= correct_index < len(options)):
            correct_index = 0  # safe default

        if question_text:
            cleaned.append(
                {
                    "question": question_text,
                    "options": options,
                    "correct_index": correct_index,
                }
            )

    return cleaned

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

    # All students (exclude admin "adminlexi"), same as admin_stories()
    users = db.execute(
        """
        SELECT id, username, email
        FROM users
        WHERE username != 'adminlexi'
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


from analytics_mock import MOCK_GROUP_LIST  # add this import at the top
# app.py additions
# ... (around line 1251)

from analytics_mock import MOCK_GROUP_LIST  # Already here, needed for mock data


# Helper function to get a single user's mock data structure
def get_user_mock_data(user_id: int):
    """
    Returns a mock data structure for a single user, or an empty one.
    In a real app, this would query the DB for this user's stats.
    """
    # Simply pick one of the mock groups and label it as the specific user.
    # The real implementation is complex, but this satisfies the requirement to display charts.
    user_data = MOCK_GROUP_LIST[0].copy()
    user_data['code'] = 'User'
    user_data['user_id'] = user_id
    user_data['username'] = f'Student_{user_id}'

    # To show different data, maybe slightly alter the scores
    if user_id % 2 == 0:
        user_data['scramble_accuracy'] = user_data['scramble_accuracy'] * 0.9
        user_data['mcq_avg_score'] = user_data['mcq_avg_score'] * 0.95

    return user_data


@app.get("/admin/analytics")
@login_required
def admin_analytics():
    """
    Admin analytics dashboard:
    L1 vs L2 charts (default) or single User charts.
    """
    if not current_user_is_admin():
        flash("Admin access only.", "warning")
        return redirect(url_for("index"))

    db = get_db()

    # Fetch all non-admin users for the selector
    all_users = db.execute(
        """
        SELECT id, username
        FROM users
        WHERE username != 'adminlexi'
        ORDER BY username COLLATE NOCASE
        """
    ).fetchall()

    selected_user_id = request.args.get("user_id", type=int)

    if selected_user_id:
        # User-specific view
        selected_user = db.execute(
            "SELECT id, username, l1_language, l2_language FROM users WHERE id = ?",
            (selected_user_id,),
        ).fetchone()

        if not selected_user:
            flash("User not found.", "warning")
            return redirect(url_for("admin_analytics"))

        # In a real app, you'd fetch the user's real stats here.
        # For now, we wrap the mock data for the selected user.
        user_stats = get_user_mock_data(selected_user_id)

        # Mock the current level from level_test_results (or a default)
        current_level_row = db.execute(
            """
            SELECT level FROM level_test_results
            WHERE user_id = ?
            ORDER BY id DESC LIMIT 1
            """,
            (selected_user_id,)
        ).fetchone()

        current_level = (current_level_row["level"] or "Beginner") if current_level_row else "Beginner"

        user_data = {
            "user": selected_user,
            "stats": user_stats,
            "current_level": current_level,
        }

        # Package the single user's mock data into the structure expected by the charts
        # Note: chart JS will use this 'single_group_list' when user_id is set
        single_group_list = [user_stats]

        return render_template(
            "admin_analytics.html",
            all_users=all_users,
            selected_user_id=selected_user_id,
            user_data=user_data,
            group_list=single_group_list,
            view_mode="user"
        )

    # Class Overview view (default)
    return render_template(
        "admin_analytics.html",
        all_users=all_users,
        selected_user_id=None,
        group_list=MOCK_GROUP_LIST,
        view_mode="class"
    )


@app.post("/admin/analytics/update-level")
@login_required
def admin_update_user_level():
    """
    POST endpoint to update a student's level based on admin input.
    """
    if not current_user_is_admin():
        flash("Admin access only.", "warning")
        return redirect(url_for("index"))

    db = get_db()
    user_id = request.form.get("user_id", type=int)
    new_level = (request.form.get("new_level") or "").strip()

    if not user_id or not new_level or new_level not in LEVEL_ORDER.keys():
        flash("Invalid user or level selection.", "warning")
        return redirect(url_for("admin_analytics"))

    # We update the *latest* level test result entry to reflect the admin adjustment.
    # A more robust system would insert a new record, but updating the latest is simpler for this structure.
    try:
        # Find the latest level result ID for the user
        latest_id_row = db.execute(
            """
            SELECT id FROM level_test_results
            WHERE user_id = ?
            ORDER BY id DESC LIMIT 1
            """,
            (user_id,)
        ).fetchone()

        if latest_id_row:
            db.execute(
                """
                UPDATE level_test_results
                SET level = ?, score = ?, total = ?
                WHERE id = ?
                """,
                (new_level.title(), 10, 10, latest_id_row["id"]),
            )
        else:
            # If no existing result, insert one
            db.execute(
                """
                INSERT INTO level_test_results (user_id, score, total, level)
                VALUES (?, ?, ?, ?)
                """,
                (user_id, 10, 10, new_level.title()),
            )

        db.commit()
        flash(f"Successfully updated user {user_id}'s level to {new_level.title()}.", "success")

    except sqlite3.Error as e:
        db.rollback()
        flash(f"Database error during level update: {e}", "danger")

    return redirect(url_for("admin_analytics", user_id=user_id))

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

            # --- ENHANCED FIXES FOR STORY FLOW AND DETAIL ---
            "**FORMAT**: Group the story into 4 to 6 cohesive paragraphs, separated by a double newline.\n"
            "**FLOW**: Ensure sentences within each paragraph flow logically, linking ideas and actions naturally.\n"
            "**STYLE**: Use simple conjunctions (like 'and', 'but', 'so') to slightly vary sentence structure and connect related short sentences, avoiding the choppy repetition of starting phrases (e.g., repeating the subject pronoun).\n"
            "**TONE**: Maintain a clear, simple, warm narrative voice with gentle repetition and a hopeful ending.\n"
            # --- END ENHANCED FIXES ---

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

        # base_author = (
        #     g.current_user["username"]
        #     if (g.current_user and g.current_user.get("username"))
        #     else None
        # ) or (request.form.get("author_name") or "guest")

        base_author = (request.form.get("author_name") or "").strip()
        want_image = bool(request.form.get("gen_image"))

        story_type = (request.form.get("story_type") or "").strip()
        # setting = (request.form.get("setting") or "").strip()
        # character_kind = (request.form.get("character_kind") or "").strip()
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

        # if setting:
        #     meta_bits.append(
        #         f"Main setting: {setting}. Most scenes should happen in this place."
        #     )

        # if character_kind:
        #     meta_bits.append(
        #         f"Main characters: {character_kind}. Use this kind of character as the focus of the story."
        #     )

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
        if story_author is None or story_author =="":
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
                # "setting": setting,
                # "character_kind": character_kind,
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



# -------------------------------------------------------------------
# Reading level helpers (Beginner / Intermediate / Advanced)
# -------------------------------------------------------------------
LEVEL_ORDER = {
    "beginner": 1,
    "intermediate": 2,
    "advanced": 3,
}


def level_rank(name: str | None) -> int | None:
    """Map a level name to an integer rank (lower is easier)."""
    if not name:
        return None
    return LEVEL_ORDER.get(str(name).strip().lower())


@app.get("/library")
@login_required
def library():
    """
    Library view for students:
    - Shows only stories that have been explicitly shared to the Library
      (stories.is_shared_library = 1), with their latest draft.
    - Students see only stories at their level or easier.
    - Admin user 'adminlexi' sees all.
    """
    db = get_db()

    # 1) Load all shared stories + latest finished draft
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
          s.visuals     AS visuals,
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

    # 2) Admin check via session username
    username = session.get("username", "")
    is_admin = (username == "adminlexi")

    # 3) Get the user's reading level from level_test_results
    user_id = session.get("user_id")
    user_rank = None

    if user_id is not None:
        latest_level_row = db.execute(
            """
            SELECT level
            FROM level_test_results
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (user_id,),
        ).fetchone()

        if latest_level_row:
            # level_test_results.level is e.g. 'Beginner', 'Intermediate', 'Advanced'
            user_rank = level_rank(latest_level_row["level"])

    # 4) If not admin and we have a valid user level, filter stories
    if (not is_admin) and user_rank is not None:
        filtered_rows = []
        for r in rows:
            story_level_name = r["level"]  # e.g. 'beginner', 'intermediate', 'advanced'
            story_rank = level_rank(story_level_name)

            # If a story has no level set, we can choose to show it.
            # Otherwise, only show if its rank <= user's rank.
            if story_rank is None or story_rank <= user_rank:
                filtered_rows.append(r)

        rows = filtered_rows

    # 5) Render template
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
# --- app.py additions (Conceptual placement near admin_stories route) ---

@app.get("/admin/stories")
@login_required
def admin_stories():
    # ... (existing admin check)
    if not current_user_is_admin():
        flash("Admin access only.", "warning")
        return redirect(url_for("index"))

    db = get_db()

    # 1. Fetch ALL Stories (Existing)
    stories = db.execute(
        """
        SELECT * FROM stories
        ORDER BY datetime(created_at) DESC
        """
    ).fetchall()

    # 2. Latest Draft per Story (Existing)
    # ... (existing draft_map logic) ...
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

    # 3. Assignment Count per Story (Existing)
    # ... (existing assign_map logic) ...
    assign_rows = db.execute(
        """
        SELECT story_id, COUNT(*) AS cnt
        FROM assignments
        GROUP BY story_id
        """
    ).fetchall()
    assign_map = {r["story_id"]: r["cnt"] for r in assign_rows}

    # 4. All Students (Existing)
    users = db.execute(
        """
        SELECT id, username, email
        FROM users
        WHERE username != 'adminlexi'
        ORDER BY username COLLATE NOCASE
        """
    ).fetchall()
    user_map = {u["id"]: u["username"] for u in users}

    # 5. NEW: Fetch ALL Submissions Needing Review (Finish-Writing only)
    # Note: MCQ assignments are auto-scored, so we focus on 'finish' submissions.
    submissions_to_review = db.execute(
        """
        SELECT
          sub.id AS submission_id,
          sub.completion_text,
          sub.score AS current_score,
          sub.comment AS admin_comment,
          sub.created_at AS submitted_at,

          a.id AS assignment_id,
          a.assignment_type,
          a.assignment_title,
          a.status AS assignment_status,
          a.assignee_id,

          s.title AS story_title

        FROM assignment_submissions sub
        JOIN assignments a ON sub.assignment_id = a.id
        JOIN stories s ON sub.story_id = s.id

        -- We only care about submissions for finish-writing that are not fully graded ('reviewed')
        -- Status 'submitted' means it needs admin review.
        WHERE a.assignment_type = 'finish' 
          AND a.status IN ('submitted', 'graded')

        -- ORDER BY status (submitted first) and date
        ORDER BY a.status ASC, datetime(sub.created_at) DESC
        """
    ).fetchall()

    # Enrich submissions with student username
    for sub in submissions_to_review:
        sub['assignee_username'] = user_map.get(sub['assignee_id'], 'Unknown User')

        # Determine current status for display
        if sub['assignment_status'] == 'graded':
            sub['review_status_label'] = 'Graded'
            sub['review_status_color'] = '#10b981'  # Success/Green
        else:
            sub['review_status_label'] = 'New Submission'
            sub['review_status_color'] = '#f59e0b'  # Warning/Orange

    return render_template(
        "admin_stories.html",
        stories=stories,
        draft_map=draft_map,
        users=users,
        assign_map=assign_map,
        submissions_to_review=submissions_to_review,  # NEW
    )


@app.post("/admin/review-submission/<int:submission_id>")
@login_required
def admin_review_submission(submission_id: int):
    """
    Admin POST endpoint to score and comment on a finish-writing submission.
    """
    if not current_user_is_admin():
        flash("Admin access only.", "warning")
        return redirect(url_for("index"))

    db = get_db()

    # Score should be between 0 and 100
    score_raw = request.form.get("score", type=float)
    admin_comment = (request.form.get("comment") or "").strip()

    score = max(0.0, min(100.0, score_raw)) if score_raw is not None else None

    if score is None:
        flash("Invalid score provided.", "warning")
        return redirect(url_for("admin_stories"))  # This will reload the page/tab

    # 1. Update the score and comment on the submission record
    db.execute(
        """
        UPDATE assignment_submissions
        SET score = ?, comment = ?, reviewed_at = datetime('now')
        WHERE id = ?
        """,
        (score, admin_comment, submission_id),
    )

    # 2. Update the parent assignment's status and final score
    # Find the assignment_id first
    submission = db.execute(
        "SELECT assignment_id FROM assignment_submissions WHERE id = ?",
        (submission_id,),
    ).fetchone()

    if submission:
        db.execute(
            """
            UPDATE assignments
            SET status = 'graded', score = ?
            WHERE id = ?
            """,
            (score, submission['assignment_id']),
        )

    db.commit()

    flash(f"Submission #{submission_id} graded (Score: {score:.1f}%).", "success")
    # Redirect to the review tab and try to keep the accordion open using a fragment
    return redirect(url_for("admin_stories", tab='review', fragment=f'submission-{submission_id}'))


# --- End app.py additions ---

@app.post("/admin/stories/<slug>/generate-mcq")
@login_required
def admin_generate_mcq(slug: str):
    """Generate and save MCQ questions for a completed story (admin only)."""
    print("ABC")
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
    # Correct admin gate (works for testtest etc.)
    if not current_user_is_admin():
        flash("Admin access only.", "warning")
        return redirect(url_for("index"))

    db = get_db()

    # 1. Load story (including visuals column)
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

    # 2. If no cover yet, generate one and save into stories.visuals
    if not visuals_data_url:
        try:
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
            print("Error generating cover image for library share:", e)
            flash(
                "Could not generate a cover image for this story. "
                "Please try again in a moment.",
                "danger",
            )
            return redirect(url_for("admin_stories"))

    # 3. Mark story as shared to Library
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
                                   min_words: int = 18,
                                   max_words: int = 36) -> list[dict]:
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

    # get a decent pool to choose from
    # enough to build 18 puzzles
    words = get_words_for_student_scramble(user_id, min_words=18, max_words=60)

    # categorize by length from student's vocab + fallback pool
    easy_words = []   # 4 letters
    medium_words = [] # 5 letters
    hard_words = []   # 6+ letters

    for item in words:
        w = item["word"].strip().lower()
        L = len(w)
        if L == 4:
            easy_words.append(w)
        elif L == 5:
            medium_words.append(w)
        elif L >= 6:
            hard_words.append(w)

    # --------- FALLBACK: ensure each bucket has at least 6 words ---------
    # Build fallback pools by length from PREPARED_SCRAMBLE_WORDS
    fallback_easy = [w.lower() for w in PREPARED_SCRAMBLE_WORDS if len(w) == 4]
    fallback_medium = [w.lower() for w in PREPARED_SCRAMBLE_WORDS if len(w) == 5]
    fallback_hard = [w.lower() for w in PREPARED_SCRAMBLE_WORDS if len(w) >= 6]

    def fill_bucket(bucket, fallback, needed):
        """Top up 'bucket' from 'fallback' until it has 'needed' items."""
        seen = set(bucket)
        for w in fallback:
            if len(bucket) >= needed:
                break
            if w not in seen:
                bucket.append(w)
                seen.add(w)

    # first, try to satisfy each bucket from its own length pool
    fill_bucket(easy_words, fallback_easy, 6)
    fill_bucket(medium_words, fallback_medium, 6)
    fill_bucket(hard_words, fallback_hard, 6)

    # shuffle now that we have enough in each bucket
    random.shuffle(easy_words)
    random.shuffle(medium_words)
    random.shuffle(hard_words)

    # select exactly 6 from each bucket
    easy_selected = easy_words[:6]
    medium_selected = medium_words[:6]
    hard_selected = hard_words[:6]

    # safety: if some bucket is STILL short (e.g. fallback lists edited)
    # we can borrow from others so we never crash; but normally this won't trigger.
    def top_up(target_list, needed, sources):
        while len(target_list) < needed:
            pulled = False
            for src in sources:
                if src:
                    target_list.append(src.pop())
                    pulled = True
                    break
            if not pulled:
                break

    top_up(easy_selected, 6, [medium_words, hard_words])
    top_up(medium_selected, 6, [easy_words, hard_words])
    top_up(hard_selected, 6, [medium_words, easy_words])

    # build puzzles in fixed order: 6 easy → 6 medium → 6 hard
    ordered_words = (
        [("easy", w) for w in easy_selected] +
        [("medium", w) for w in medium_selected] +
        [("hard", w) for w in hard_selected]
    )

    puzzles = []
    for idx, (difficulty, w) in enumerate(ordered_words):
        scrambled = scramble_word(w)
        puzzles.append(
            {
                "id": idx,
                "word": w,
                "scrambled": scrambled,
                "difficulty": difficulty,  # optional, not required by your JS
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
