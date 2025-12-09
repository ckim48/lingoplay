import os
import re
import json
import sqlite3
import random
from datetime import datetime
from functools import wraps
from urllib.parse import urlparse, urljoin
from typing import Optional

from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, g, session, jsonify, abort
)
from werkzeug.security import generate_password_hash, check_password_hash
from slugify import slugify

# NOTE: Assuming the analytics_mock.py file is present for MOCK_GROUP_LIST
try:
    from analytics_mock import MOCK_GROUP_LIST
except ImportError:
    MOCK_GROUP_LIST = []
    print("WARNING: analytics_mock.py not found. Mock data will be empty.")

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


# -------------------------------------------------------------------
# Jinja2 Custom Filters (FIXED: Registering 'chr' and 'from_json')
# -------------------------------------------------------------------
@app.template_filter('chr')
def char_filter(value):
    """Makes the Python built-in chr() function available in Jinja."""
    try:
        return chr(value)
    except (TypeError, ValueError):
        # Handle cases where the input is not a valid integer for chr()
        return ''


@app.template_filter('from_json')
def from_json_filter(s):
    """Parses a JSON string into a Python object."""
    if s:
        try:
            # Handle string input
            if isinstance(s, str):
                return json.loads(s)
            # If it's already a dict (unlikely from DB but safe)
            if isinstance(s, dict):
                return s
            return None
        except (TypeError, json.JSONDecodeError):
            return None
    return None


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
        """
    )

    # MIGRATIONS FOR CONFIRMED TABLES: stories, assignment_submissions, assignments

    # 1. Stories migrations
    try:
        db.execute(
            "ALTER TABLE stories ADD COLUMN mcq_questions_json TEXT"
        )
    except sqlite3.OperationalError:
        pass

    # 2. assignment_submissions migrations
    try:
        db.execute(
            "ALTER TABLE assignment_submissions ADD COLUMN comment TEXT"
        )
    except sqlite3.OperationalError:
        pass

    try:
        db.execute(
            "ALTER TABLE assignment_submissions ADD COLUMN reviewed_at TEXT"
        )
    except sqlite3.OperationalError:
        pass

    # 3. assignments migrations (Needed for assignment title, JSON, attempts, score)
    try:
        db.execute(
            "ALTER TABLE assignments ADD COLUMN assignment_title TEXT"
        )
    except sqlite3.OperationalError:
        pass

    try:
        db.execute(
            "ALTER TABLE assignments ADD COLUMN questions_json TEXT"
        )
    except sqlite3.OperationalError:
        pass

    try:
        db.execute(
            "ALTER TABLE assignments ADD COLUMN attempt_count INTEGER DEFAULT 0"
        )
    except sqlite3.OperationalError:
        pass

    try:
        db.execute(
            "ALTER TABLE assignments ADD COLUMN score REAL"
        )
    except sqlite3.OperationalError:
        pass

    db.commit()


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
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    confirm = data.get("confirm") or ""
    l1 = (data.get("l1_language") or "").strip().lower()
    l2 = (data.get("l2_language") or "").strip().lower()
    age_raw = (data.get("age") or "").strip()
    gender = (data.get("gender") or "").strip().lower()

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


@app.route("/register", methods=["GET", "POST"])
def register():
    db = get_db()

    if request.method == "POST":
        # ----- Read fields -----
        username = (request.form.get("username") or "").strip()
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""
        confirm = request.form.get("confirm") or ""

        l1_language_raw = (request.form.get("l1_language") or "").strip().lower()
        l2_language_raw = (request.form.get("l2_language") or "").strip().lower()
        age_raw = (request.form.get("age") or "").strip()
        gender_raw = (request.form.get("gender") or "").strip().lower()

        level_score_raw = request.form.get("level_score")
        level_name = (request.form.get("level_name") or "").strip()

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
                registered_level_name=None,
                registered_level_score=None,
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
def admin_assign_story(slug: str):
    """
    Admin page:
      - choose assignment type (finish or mcq)
      - select which students will get this story
      FIXED: Inserts into the 'assignments' table.
    """
    if not current_user_is_admin():
        flash("Admin access only.", "warning")
        return redirect(url_for("index"))

    db = get_db()

    # Story info
    story = db.execute(
        "SELECT * FROM stories WHERE slug = ?",
        (slug,),
    ).fetchone()

    if story is None:
        flash("Story not found.", "warning")
        return redirect(url_for("admin_stories"))

    # Latest draft for this story (if any)
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

    # All non-admin users (students) for selection
    users = db.execute(
        """
        SELECT id, username, email
        FROM users
        WHERE is_admin = 0
        ORDER BY username ASC
        """
    ).fetchall()

    if request.method == "POST":
        assignment_title = (request.form.get("assignment_title") or "").strip()
        assignment_type = (request.form.get("worksheet_type") or "").strip()
        raw_user_ids = request.form.getlist("user_ids") or []
        user_ids = [uid for uid in raw_user_ids if uid]

        if not assignment_title:
            # Fallback auto title
            if assignment_type == "reading":
                assignment_title = f"{story['title']} · MCQ Reading"
            else:
                assignment_title = f"{story['title']} · Finish Writing"

        if assignment_type not in {"writing", "reading"}:
            flash("Please choose an assignment type.", "warning")
            return redirect(url_for("admin_story_detail", slug=slug))

        if not user_ids:
            flash("Please select at least one student.", "warning")
            return redirect(url_for("admin_story_detail", slug=slug))

        if not draft:
            flash("This story does not have a usable draft yet.", "warning")
            return redirect(url_for("admin_story_detail", slug=slug))

        # -------------------------------------------------
        # For READING: require pre-generated questions on the story
        # -------------------------------------------------
        questions_json = None
        if assignment_type == "reading":
            base_q_json = None
            try:
                base_q_json = story["mcq_questions_json"]
            except Exception:
                base_q_json = None

            if base_q_json and str(base_q_json).strip():
                questions_json = base_q_json
            else:
                flash(
                    "MCQ questions are not generated yet for this story. "
                    "Use the 'Generate MCQ' button first.",
                    "warning",
                )
                return redirect(url_for("admin_story_detail", slug=slug))

        # -------------------------------------------------
        # Prevent duplicate assignments for same story+type+student
        # FIX: Query assignments table using assignee_id
        # -------------------------------------------------
        placeholders = ",".join("?" for _ in user_ids)
        already = set()
        if placeholders:
            rows = db.execute(
                f"""
                SELECT assignee_id
                from assignments
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
            assignee_id = int(uid)
            # FIX: Insert into 'assignments' table, using 'assignee_id'
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
                    g.current_user["id"] if getattr(g, "current_user", None) else None,
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

    # GET → show manual assignment page (rarely used; modal is main)
    return render_template(
        "admin_assign_story.html",
        story=story,
        users=users,
        draft=draft,
    )


# Helper function to get a single user's mock data structure
def get_user_mock_data(user_id: int):
    """
    Returns a mock data structure for a single user, or an empty one.
    In a real app, this would query the DB for this user's stats.
    """
    # Simply pick one of the mock groups and label it as the specific user.
    if not MOCK_GROUP_LIST:
        return {}

    user_data = MOCK_GROUP_LIST[0].copy()
    user_data['code'] = 'User'
    user_data['user_id'] = user_id
    user_data['username'] = f'Student_{user_id}'

    # To show different data, maybe slightly alter the scores
    if user_id % 2 == 0:
        user_data['scramble_accuracy'] = user_data.get('scramble_accuracy', 1.0) * 0.9
        user_data['mcq_avg_score'] = user_data.get('mcq_avg_score', 1.0) * 0.95

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
            "SELECT * FROM users WHERE id = ?",
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

    # NOTE: Assuming LEVEL_ORDER exists or using a simple check.
    LEVEL_ORDER = {"beginner": 1, "intermediate": 2, "advanced": 3}
    if not user_id or not new_level or new_level.lower() not in LEVEL_ORDER.keys():
        flash("Invalid user or level selection.", "warning")
        return redirect(url_for("admin_analytics"))

    # We update the *latest* level test result entry to reflect the admin adjustment.
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
@app.get("/assignments")
@login_required
def assignments_list():
    db = get_db()
    # FIX: Query the 'assignments' table using 'assignee_id'
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
        from assignments a
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

    # FIX: Query the 'assignments' table using 'id' and 'assignee_id'
    assignment = db.execute(
        """
        SELECT * from assignments
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

    # FIX: Query the 'assignment_submissions' table
    submission = db.execute(
        """
        SELECT *
        from assignment_submissions
        WHERE assignment_id = ? AND user_id = ?
        ORDER BY datetime(updated_at) DESC
        LIMIT 1
        """,
        (assignment_id, user_id),
    ).fetchone()

    # ------------------------------------------------------------------
    # WRITING ASSIGNMENT (Type: writing)
    # ------------------------------------------------------------------
    if assignment["assignment_type"] == "writing":

        # --- 1. Load Writing Worksheet Data ---
        writing_sections = []
        writing_checklist = []
        if assignment.get("questions_json"):
            try:
                # The payload for writing worksheets is structured with 'sections' and 'checklist'
                worksheet_data = json.loads(assignment["questions_json"])
                writing_sections = worksheet_data.get("sections") or []
                writing_checklist = worksheet_data.get("checklist") or []
            except Exception as e:
                log_input("writing_questions_parse_error", {"error": str(e), "assignment_id": assignment_id})

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

            # Update 'assignments' table status and attempt count
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
            writing_sections=writing_sections,  # Pass writing structure sections
            writing_checklist=writing_checklist,  # Pass writing checklist
        )

    # ------------------------------------------------------------------
    # READING ASSIGNMENT (Type: reading - handles MCQ, Fill-in-the-Blank, Short Answer)
    # ------------------------------------------------------------------
    elif assignment["assignment_type"] == "reading":
        mcq_questions = []
        fill_in_blank_questions = []
        short_answer_questions = []

        questions_payload = {}

        # Attempt to load questions from the assignment JSON data
        if assignment.get("questions_json"):
            try:
                questions_payload = json.loads(assignment["questions_json"])

                # 1. Structured Worksheet JSON
                if questions_payload.get("type") == "reading":
                    mcq_questions = questions_payload.get("mcq") or []
                    fill_in_blank_questions = questions_payload.get("fill_in_blank") or []
                    short_answer_questions = questions_payload.get("short_answer") or []

                # 2. Fallback: If payload is a flat list of MCQs (legacy structure)
                elif isinstance(questions_payload, list) and all('correct_index' in q for q in questions_payload):
                    mcq_questions = questions_payload

            except Exception as e:
                # Log parsing errors if the JSON is malformed
                log_input("reading_questions_parse_error", {"error": str(e), "assignment_id": assignment_id})

        # 3. Legacy Fallback: Check story's mcq_questions_json
        if not mcq_questions and story.get("mcq_questions_json"):
            try:
                # Assume legacy JSON is a flat list of MCQs
                mcq_questions = json.loads(story["mcq_questions_json"])
            except Exception as e:
                log_input("story_mcq_questions_parse_error", {"error": str(e), "story_id": story["id"]})

        if request.method == "POST":
            # --- POST LOGIC FOR READING WORKSHEET ---
            answers = []
            correct_count = 0

            # --- Scoring Logic (Only MCQs are auto-scored) ---
            if mcq_questions:
                for idx, q in enumerate(mcq_questions):
                    key = f"q{idx}"
                    ans_raw = request.form.get(key)
                    try:
                        ans_idx = int(ans_raw)
                    except (TypeError, ValueError):
                        ans_idx = None

                    answers.append(ans_idx)

                    # Check correctness
                    correct_index = int(q.get("correct_index", -1))
                    if ans_idx is not None and 0 <= ans_idx < len(q.get("options", [])):
                        if ans_idx == correct_index:
                            correct_count += 1

                total_questions = len(mcq_questions)
                score = (correct_count / total_questions) * 100.0
            else:
                # If there are no MCQs, score is 0.0
                score = 0.0

            # Gather all non-MCQ answers for storage (Fill-in-the-Blank and Short Answer)
            all_answers = {
                "mcq_answers": answers,
                "fill_in_blank_responses": [request.form.get(f"fill{i}") for i in range(len(fill_in_blank_questions))],
                "short_answer_responses": [request.form.get(f"short{i}") for i in range(len(short_answer_questions))],
            }

            now = datetime.utcnow().isoformat(timespec="seconds")

            # Store the submission
            if submission:
                db.execute(
                    """
                    UPDATE assignment_submissions
                    SET answers_json = ?, score = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (json.dumps(all_answers), score, now, submission["id"]),
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
                        json.dumps(all_answers),
                        score,
                        now,
                        now,
                    ),
                )

            # Update 'assignments' table status, score, and attempt count
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

            # Reload updated assignment & latest submission
            assignment = db.execute(
                """
                SELECT * from assignments
                WHERE id = ? AND assignee_id = ?
                """,
                (assignment_id, user_id),
            ).fetchone()

            submission = db.execute(
                """
                SELECT *
                from assignment_submissions
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
                mcq_questions=mcq_questions,
                fill_in_blank_questions=fill_in_blank_questions,
                short_answer_questions=short_answer_questions,
                submission=submission,
                just_submitted=True,
            )

        # GET: first load / coming back from list
        return render_template(
            "assignment_mcq.html",
            assignment=assignment,
            story=story,
            mcq_questions=mcq_questions,
            fill_in_blank_questions=fill_in_blank_questions,
            short_answer_questions=short_answer_questions,
            submission=submission,
            just_submitted=False,
        )

    # ------------------------------------------------------------------
    # Fallback: unknown type
    # ------------------------------------------------------------------
    flash("Unknown assignment type.", "warning")
    return redirect(url_for("assignments_list"))


# -------------------------------------------------------------------
# Story generation helpers (no changes)
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
# Simple vocab helpers (no changes)
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
# Finish drafts helper (no changes)
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
# Routes: index, story_new + library/finish view (no changes)
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

        base_author = (request.form.get("author_name") or "").strip()

        story_type = (request.form.get("story_type") or "").strip()
        emotion_tone = (request.form.get("emotion_tone") or "").strip()
        tense = (request.form.get("tense") or "").strip()

        # We always require a prompt
        if not prompt:
            flash("Please provide phonics letters or vocabulary.", "warning")
            return redirect(url_for("story_new"))

        # ---- Build meta prompt bits ----
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

        # ---- Generate story text ----
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

        # ---- Always try to generate cover image (if client is available) ----
        visuals_data_url = None
        if client is not None:
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
                print("generate_image_error", {"error": str(e)})
                flash("Story created, but image generation had an issue.", "warning")

        story_author = base_author
        if story_author is None or story_author == "":
            story_author = "EduWeaver AI"

        slug_base = slugify(title) or "story"
        ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        slug = f"{slug_base}-{ts}"

        db_level = english_level or "beginner"

        # ---- Insert story ----
        db.execute(
            """
            INSERT INTO stories (title, slug, prompt, language, level, content, visuals, author_name, is_shared_library)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?,?)
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
                1
            ),
        )
        story_id = db.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]
        db.commit()

        # ---- Insert vocab items ----
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

        # ---- Log story generation ----
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
                "with_image": bool(visuals_data_url),
                "story_type": story_type,
                "emotion_tone": emotion_tone,
                "tense": tense,
                # No more student_finish flag: story is always fully generated
            },
        )

        # ---- Always create a finish_drafts row WITH completion_text ----
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

        flash("Story generated by EduWeaver AI and added to the Library.", "success")
        return redirect(url_for("library"))

    return render_template("story_new.html")


# -------------------------------------------------------------------
# Reading level helpers
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


@app.get("/admin/students")
@login_required
def admin_students():
    if not current_user_is_admin():
        flash("Admin access only.", "warning")
        return redirect(url_for("index"))

    db = get_db()

    user_rows = db.execute(
        """
        SELECT id, username, email, l1_language, l2_language
        FROM users
        WHERE username != 'adminlexi'
        ORDER BY username COLLATE NOCASE
        """
    ).fetchall()

    students = []

    for u in user_rows:
        user_id = u["id"]

        level_row = db.execute(
            """
            SELECT level
            FROM level_test_results
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (user_id,),
        ).fetchone()
        current_level = level_row["level"] if level_row else None

        # FIX: Query the 'assignments' table using 'assignee_id'
        stats_row = db.execute(
            """
            SELECT
              COUNT(*) AS total_assignments,
              SUM(
                CASE WHEN status IN ('submitted','graded')
                     THEN 1 ELSE 0 END
              ) AS completed_assignments,
              AVG(
                CASE WHEN score IS NOT NULL
                     THEN score END
              ) AS avg_score
            from assignments 
            WHERE assignee_id = ? 
            """,
            (user_id,),
        ).fetchone()

        total_assignments = stats_row["total_assignments"] or 0
        completed_assignments = stats_row["completed_assignments"] or 0
        avg_score = stats_row["avg_score"] if stats_row["avg_score"] is not None else 0.0

        students.append({
            "id": user_id,
            "username": u["username"],
            "email": u["email"],
            "l1_language": u.get("l1_language"),
            "l2_language": u.get("l2_language"),
            "current_level": current_level,
            "total_assignments": total_assignments,
            "completed_assignments": completed_assignments,
            "overdue_assignments": 0,
            "avg_score": avg_score,
            "last_seen_at": None,
        })

    total_students = len(students)

    active_this_week = 0

    if total_students:
        avg_completion_rate = (
                sum(
                    ((s["completed_assignments"] or 0) / (s["total_assignments"] or 1) * 100.0)
                    for s in students
                ) / total_students
        )
        avg_score_all = (
                sum((s["avg_score"] or 0.0) for s in students) / total_students
        )
    else:
        avg_completion_rate = 0.0
        avg_score_all = 0.0

    q = request.args.get("q", "").strip()
    if q:
        q_lower = q.lower()
        students = [
            s for s in students
            if q_lower in (s["username"] or "").lower()
               or q_lower in (s["email"] or "").lower()
        ]

    return render_template(
        "admin_students.html",
        students=students,
        total_students=total_students,
        active_this_week=active_this_week,
        avg_completion_rate=avg_completion_rate,
        avg_score_all=avg_score_all,
    )


# In app.py - Replace your existing library function with this:

@app.get("/library")
@login_required
def library():
    """
    Library view for students:
    - Shows shared stories filtered by user level and optional search/sort parameters.
    """
    db = get_db()

    # --- GET QUERY PARAMETERS ---
    search_query = request.args.get("q", "").strip()
    sort_by = request.args.get("sort", "newest")  # Default sort

    # --- 1) DETERMINE USER/ADMIN CONTEXT AND LEVEL ---
    username = session.get("username", "")
    is_admin = (username == "testtest")  # Assuming "testtest" is the admin user
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
            user_rank = level_rank(latest_level_row["level"])

    # --- 2) BUILD BASE QUERY AND PARAMETERS ---
    params = []
    where_clauses = ["s.is_shared_library = 1"]

    # LEVEL FILTERING (Only for non-admin users)
    if not is_admin and user_rank is not None:
        # We need to find stories where the story's level rank is <= user's rank
        # This requires manually mapping levels since SQLite can't directly compare text ranks.
        # This implementation requires fetching all level-ranked stories first, which is complex for SQL dynamic WHERE clauses.
        # For simplicity and performance within this limited scope, we fetch all, then filter in Python (see Step 3).
        # A proper fix would be a complex subquery or using a separate level_rank table/view.
        # We'll skip adding level-based WHERE clause for now and rely purely on Python filtering later.
        pass

    # SEARCH QUERY FILTERING
    if search_query:
        # Search by title, author, or original prompt (vocabulary)
        search_term = f"%{search_query.lower()}%"
        where_clauses.append(
            """
            (
                LOWER(s.title) LIKE ? OR
                LOWER(s.author_name) LIKE ? OR
                LOWER(s.prompt) LIKE ?
            )
            """
        )
        params.extend([search_term, search_term, search_term])

    # --- 3) DETERMINE SORT ORDER ---
    order_map = {
        "newest": "ORDER BY datetime(s.created_at) DESC",
        "title (a-z)": "ORDER BY s.title COLLATE NOCASE ASC",
        # We sort by level rank ascending (easiest first). Requires calculation.
        "level (easiest)": "ORDER BY s.level ASC, datetime(s.created_at) DESC",
        # Note: SQLite simple ORDER BY on level text may be inaccurate for ranks.
    }
    order_by = order_map.get(sort_by.lower(), order_map["newest"])

    # --- 4) EXECUTE MAIN QUERY ---
    query = f"""
        SELECT
          s.id AS story_id, s.title AS title, s.slug AS slug, s.level AS level,
          s.language AS language, s.author_name AS author_name, 
          s.created_at AS story_created_at, s.visuals AS visuals,
          fd.id AS draft_id
        FROM stories s
        JOIN finish_drafts fd
          ON fd.story_id = s.id
         AND datetime(fd.created_at) = (
            SELECT MAX(datetime(fd2.created_at))
            FROM finish_drafts fd2
            WHERE fd2.story_id = s.id
         )
        WHERE {' AND '.join(where_clauses)}
        {order_by}
    """

    rows = db.execute(query, params).fetchall()

    # --- 5) PYTHON-SIDE LEVEL FILTERING (For accuracy on rank) ---
    if not is_admin and user_rank is not None:
        filtered_rows = []
        for r in rows:
            story_rank = level_rank(r["level"])
            if story_rank is None or story_rank <= user_rank:
                filtered_rows.append(r)
        rows = filtered_rows

    # --- 6) RENDER ---
    return render_template(
        "library.html",
        stories=rows,
        is_admin=is_admin,
        search_query=search_query,
        sort_by=sort_by
    )


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
@app.route("/admin/stories")
@login_required
def admin_stories():
    if not current_user_is_admin():
        flash("Admin access only.", "warning")
        return redirect(url_for("index"))

    db = get_db()

    # Which tab to show (Assign / Assigned / Submitted)
    active_tab = request.args.get("tab") or "assign"
    fragment = request.args.get("fragment") or ""

    # ------------------------------------------------------------------
    # TAB 1: Story list for building assignments
    # ------------------------------------------------------------------
    stories = db.execute(
        """
        SELECT
            s.id,
            s.slug,
            s.title,
            s.language,
            s.level,
            s.created_at
        FROM stories s
        ORDER BY s.created_at DESC
        """
    ).fetchall()

    # All non-admin users (students) for assignment selection
    users = db.execute(
        "SELECT id, username, email FROM users ORDER BY username"
    ).fetchall()

    # ------------------------------------------------------------------
    # TAB 2: Assigned Assignments (Groups based on assignments table)
    # ------------------------------------------------------------------
    assignment_rows = db.execute(
        """
        SELECT
            a.id              AS assignment_id,
            a.story_id        AS story_id,
            a.assignment_type AS assignment_type,
            a.assignment_title AS assignment_title,
            a.created_at      AS assignment_created_at,

            s.title           AS story_title,
            s.language        AS language,
            s.level           AS level,

            a.assignee_id     AS assignee_id,
            a.status          AS assignee_status,
            a.score           AS assignee_score,
            a.attempt_count   AS assignee_attempts,

            u.username        AS assignee_username,
            u.email           AS assignee_email

        FROM assignments a 
        JOIN stories s
          ON a.story_id = s.id
        JOIN users u
          ON a.assignee_id = u.id 
        ORDER BY a.created_at DESC, a.id DESC
        """
    ).fetchall()

    # Grouping logic: Group by template characteristics (story, type, title)
    assignment_groups = {}

    for row in assignment_rows:
        # Group key: (story_id, assignment_type, assignment_title)
        key = (row["story_id"], row["assignment_type"], row["assignment_title"])

        if key not in assignment_groups:
            assignment_groups[key] = {
                "group_id": len(assignment_groups) + 1,
                "assignment_title": row["assignment_title"],
                "assignment_type": row["assignment_type"],
                "story_id": row["story_id"],
                "story_title": row["story_title"],
                "language": row["language"],
                "level": row["level"],
                "created_at": row["assignment_created_at"],
                "primary_assignment_id": row["assignment_id"],  # ID of the first row for the edit link
                "assignees": [],
                "count_assigned": 0,
                "count_submitted": 0,
                "count_graded": 0,
            }

        g = assignment_groups[key]

        # Add the individual assignee record
        g["assignees"].append(
            {
                "id": row["assignee_id"],
                "username": row["assignee_username"],
                "email": row["assignee_email"],
                "status": row["assignee_status"],
                "score": row["assignee_score"],
                "attempt_count": row["assignee_attempts"],
            }
        )

        status = (row["assignee_status"] or "assigned").lower()
        if status == "submitted":
            g["count_submitted"] += 1
        elif status == "graded":
            g["count_graded"] += 1
        else:
            g["count_assigned"] += 1

    # dict → list, 최신 순
    assignments = sorted(
        assignment_groups.values(),
        key=lambda x: x["created_at"] or "",
        reverse=True,
    )

    # ------------------------------------------------------------------
    # TAB 3: Submitted Work – assignment_submissions 기반
    # ------------------------------------------------------------------
    submissions_to_review = db.execute(
        """
        SELECT
            ws.id              AS submission_id,
            ws.assignment_id   AS assignment_id,
            ws.user_id         AS assignee_id,
            u.username         AS assignee_username,

            a.assignment_title AS assignment_title,
            s.title            AS story_title,

            ws.completion_text AS completion_text,
            ws.score           AS current_score,
            ws.comment         AS admin_comment,
            ws.created_at      AS submitted_at,
            ws.reviewed_at     AS reviewed_at,
            CASE
                WHEN ws.reviewed_at IS NULL THEN 'Pending review'
                ELSE 'Reviewed'
            END AS review_status_label,
            CASE
                WHEN ws.reviewed_at IS NULL THEN '#fee2e2'
                ELSE '#dcfce7'
            END AS review_status_color
        FROM assignment_submissions ws 
        JOIN assignments a 
          ON ws.assignment_id = a.id
        JOIN stories s
          ON a.story_id = s.id
        JOIN users u
          ON ws.user_id = u.id
        WHERE ws.completion_text IS NOT NULL OR ws.answers_json IS NOT NULL
        ORDER BY ws.created_at DESC
        """
    ).fetchall()

    return render_template(
        "admin_stories.html",
        stories=stories,
        users=users,
        assignments=assignments,
        submissions_to_review=submissions_to_review,
        active_tab=active_tab,
        fragment=fragment,
    )


@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    db = get_db()
    user = g.current_user

    if request.method == "POST":
        # --- 회원정보 수정 폼 처리 ---
        username = (request.form.get("username") or "").strip()
        email = (request.form.get("email") or "").strip()
        age_raw = (request.form.get("age") or "").strip()
        gender = (request.form.get("gender") or "").strip() or None
        is_english_native = request.form.get("is_english_native") or None

        age = None
        if age_raw:
            try:
                age = int(age_raw)
            except ValueError:
                flash("Age must be a number.", "danger")
                return redirect(url_for("profile"))

        # 비어 있으면 기존 값 유지
        if not username:
            username = user.get("username")
        if not email:
            email = user.get("email")

        db.execute(
            """
            UPDATE users
            SET username = ?, email = ?, age = ?, gender = ?, is_english_native = ?
            WHERE id = ?
            """,
            (username, email, age, gender, is_english_native, user["id"]),
        )
        db.commit()
        flash("Profile updated.", "success")
        return redirect(url_for("profile"))

    # --- GET: 대시보드 데이터 로딩 ---
    # 최신 유저 정보
    user_row = db.execute(
        "SELECT * FROM users WHERE id = ?",
        (user["id"],)
    ).fetchone()
    my_level_row = db.execute(
        "SELECT level FROM level_test_results WHERE user_id = ? ORDER BY id DESC LIMIT 1"
        ,
        (user["id"],),
    ).fetchone()
    my_level = my_level_row["level"] if my_level_row else "Unknown"

    # 학습 프로필 (나이/성별/영어 모국어 여부)
    learner_profile = get_learner_profile()

    # 통계: 읽은 책(스토리), 완료한 워크시트, 평균 점수
    # FIX: Use assignment_submissions and assignments tables
    stats = db.execute(
        """
        SELECT
            COUNT(DISTINCT a.story_id) AS books_read,
            COUNT(sub.id) AS worksheets_completed,
            AVG(CASE WHEN sub.score IS NOT NULL THEN sub.score END) AS avg_score
        from assignment_submissions sub
        JOIN assignments a ON sub.assignment_id = a.id
        WHERE sub.user_id = ?
        """,
        (user["id"],),
    ).fetchone()

    # FIX: Use assignment_submissions and assignments tables
    submissions = db.execute(
        """
        SELECT
            sub.id AS submission_id,
            sub.completion_text,
            sub.score AS current_score,
            sub.comment AS admin_comment,
            sub.created_at AS submitted_at,
            a.assignment_title,
            s.title AS story_title,
            s.level AS story_level
        from assignment_submissions sub
        JOIN assignments a ON sub.assignment_id = a.id
        JOIN stories s ON a.story_id = s.id
        WHERE sub.user_id = ?
        ORDER BY datetime(sub.created_at) DESC
        LIMIT 10
        """,
        (user["id"],),
    ).fetchall()

    return render_template(
        "profile.html",
        profile_user=user_row,
        learner_profile=learner_profile,
        stats=stats,
        my_level=my_level,
        submissions=submissions,
    )


@app.post("/admin/assignments/<int:assignment_id>/edit")
@login_required
def admin_edit_worksheet_assignment(assignment_id: int):
    """
    NOTE: Function name changed to match endpoint used in admin_stories.html.
    - assignment 제목 수정 (같은 assignment กลุ่ม 전체에 적용)
    - 추가 학생들에게도 동일 assignment 배정
    - 'Unassign' 버튼: 이 assignment를 모든 학생에게서 제거
    """
    if not current_user_is_admin():
        abort(403)

    db = get_db()

    # 대표 assignment row 가져오기 (per-user assignment)
    # FIX: Use assignments table
    assignment = db.execute(
        """
        SELECT
            a.*,
            s.title AS story_title
        from assignments a
        JOIN stories s ON a.story_id = s.id
        WHERE a.id = ?
        """,
        (assignment_id,),
    ).fetchone()

    if assignment is None:
        abort(404)

    action = request.form.get("action")
    assignment_type = assignment["assignment_type"]

    # ----------------------------
    # 1) Unassign: 이 assignment 그룹 전체 삭제
    # ----------------------------
    if action == "delete":
        # FIX: Delete submissions first, then assignments
        db.execute(
            """
            DELETE from assignment_submissions
            WHERE assignment_id IN (
                SELECT id from assignments
                WHERE story_id = ?
                  AND assignment_type = ?
                  AND assignment_title = ?
            )
            """,
            (assignment["story_id"], assignment_type, assignment["assignment_title"]),
        )
        db.execute(
            """
            DELETE from assignments
            WHERE story_id = ?
              AND assignment_type = ?
              AND assignment_title = ?
            """,
            (assignment["story_id"], assignment_type, assignment["assignment_title"]),
        )
        db.commit()
        flash("Assignment has been unassigned from all students.", "success")
        return redirect(url_for("admin_stories", tab="assigned"))

    # ----------------------------
    # 2) Save changes: 제목 수정 + 추가 학생 배정
    # ----------------------------
    new_title = (request.form.get("worksheet_title") or "").strip()
    extra_user_ids = request.form.getlist("extra_user_ids")

    # 2-1) 제목이 바뀌면, 같은 assignment กลุ่ม 전체 업데이트
    if new_title and new_title != assignment["assignment_title"]:
        # FIX: Update assignments table
        db.execute(
            """
            UPDATE assignments
            SET assignment_title = ?
            WHERE story_id = ?
              AND assignment_type = ?
              AND assignment_title = ?
            """,
            (
                new_title,
                assignment["story_id"],
                assignment_type,
                assignment["assignment_title"],
            ),
        )

    # 이 이후부터는 최신 제목 기준으로 사용
    effective_title = new_title if new_title else assignment["assignment_title"]

    # 2-2) 이미 배정된 학생들 확인 (assignee_id 기준)
    # FIX: Check assignments table for existing assignees
    existing = db.execute(
        """
        SELECT assignee_id
        from assignments
        WHERE story_id = ?
          AND assignment_type = ?
          AND assignment_title = ?
        """,
        (assignment["story_id"], assignment_type, effective_title),
    ).fetchall()
    existing_ids = {row["assignee_id"] for row in existing}

    # 2-3) extra_user_ids 에서 아직 배정 안 된 학생에게만 새 row 생성
    for uid_str in extra_user_ids:
        try:
            uid = int(uid_str)
        except ValueError:
            continue
        if uid in existing_ids:
            continue

        # FIX: Insert new per-user assignment into assignments table
        db.execute(
            """
            INSERT INTO assignments
            (story_id, draft_id, assignee_id, assignment_type,
             questions_json, assigned_by, assignment_title,
             status, score, attempt_count, created_at)
            SELECT
                story_id,
                draft_id,
                ?,               -- assignee_id (새 학생)
                assignment_type,
                questions_json,
                assigned_by,
                ?,               -- 새 제목(또는 기존 제목)
                'assigned',
                NULL,
                0,
                CURRENT_TIMESTAMP
            from assignments
            WHERE id = ? -- Copy template data from the primary assignment row
            """,
            (uid, effective_title, assignment_id),
        )

    db.commit()
    flash("Assignment group has been updated.", "success")
    # Redirect to the correct endpoint. Since the route is '/admin/assignments/<int:assignment_id>/edit',
    # we redirect back to the /admin/stories tab.
    return redirect(url_for("admin_stories", tab="assigned"))


# --- New Helper: make_structured_story ---
# --- Updated make_structured_story function in app.py ---
# --- Helper: make_structured_story ---
def make_structured_story(prompt: str, level: str) -> dict:
    """
    Generates a full story structured into Beginning, Middle, and Ending parts
    using the OpenAI API. Forces JSON output via the system prompt.
    """
    if client is None:
        # Robust fallback for when AI is disabled
        return {"title": "Luna's Lost Star",
                "beginning": "Luna the fox cub woke up to a dark sky. 'Where is my favorite morning star?' she whispered. It was always the first thing she saw. Luna decided she must go find it.",
                "middle": "She climbed the tallest oak tree, but the star was not there. She asked the sleepy owl and the busy squirrel, but nobody had seen a star fall. Luna felt a little sad, but she kept looking.",
                "ending": "Finally, Luna looked down at her feet. The star wasn't in the sky at all! It was a shiny piece of glass left by the pond, reflecting the moon's light. Luna giggled and carefully put the shiny glass next to her bed."}

    # Define the required JSON structure
    json_schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "beginning": {"type": "string",
                          "description": "The introduction and setup (MUST be 2-3 detailed paragraphs)."},
            "middle": {"type": "string",
                       "description": "The conflict and main action (MUST be 2-3 detailed paragraphs)."},
            "ending": {"type": "string",
                       "description": "The resolution and conclusion (MUST be 2-3 detailed paragraphs)."}
        },
        "required": ["title", "beginning", "middle", "ending"]
    }

    # 1. Inject JSON requirement into the system prompt
    system = (
        "You are a children's story generator for phonics & early readers. "
        "You must generate a complete story and divide it clearly into three distinct, cohesive parts: Beginning, Middle, and Ending. "
        "Each part MUST be written as 2 to 3 detailed paragraphs, separated by double newlines. "
        "IMPORTANT: Output ONLY a single JSON object (no markdown, no commentary). The output MUST match the keys: title, beginning, middle, ending. "
        f"Schema hint: {json.dumps(json_schema)}"
    )

    user = f"""
    Target Story Level: {level.title()}
    Target elements: {prompt}
    Generate a complete, simple, child-friendly story of 300-450 words, structured into three parts (Beginning, Middle, Ending). Each part should contain multiple sentences and clear action.
    """

    try:
        resp = client.responses.create(
            model=DEFAULT_MODEL,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.7,
            max_output_tokens=1500,
        )
        raw_json = getattr(resp, "output_text", "") or ""

        # Robust JSON parsing
        raw_json = raw_json.strip()
        if raw_json.startswith("```json"):
            raw_json = raw_json.strip("```json").strip("```").strip()

        return json.loads(raw_json)
    except Exception as e:
        print("Error calling OpenAI for structured story:", e)
        # Ensure a robust fallback is always returned
        return {"title": "Fallback Story",
                "beginning": "A young bear named Barnaby lived in a tall, green forest. He loved honey more than anything. One sunny morning, Barnaby decided he was going to find the biggest, sweetest hive in the whole forest. He packed his empty jar and waved goodbye to his mother.",
                "middle": "Barnaby searched all morning. He climbed over giant mossy logs and waded across a cold, trickling stream. Finally, high up in a maple tree, he saw it: a huge, round beehive! He knew this was the one. But when he reached the first branch, the bees buzzed angrily. Barnaby realized climbing up would be very dangerous. He was stuck.",
                "ending": "Barnaby sat down and thought hard. Instead of climbing up, he decided to wait quietly near the trunk. After a few minutes, the queen bee flew down. Barnaby bowed politely and said, 'Excuse me, I love honey. Could I please trade you a sweet red apple for a little bit?' The queen bee agreed! Barnaby got his honey and the bees got a tasty apple. It was a win-win day, and Barnaby learned that being patient and polite works better than climbing."}


### B. Helper: `create_finish_prompt` (Implements Difficulty)


# --- Helper: create_finish_prompt (Implements Difficulty) ---
def create_finish_prompt(structured_story: dict, assignment_level: str) -> tuple[str, str, str]:
    """
    Creates a partial story prompt by randomly omitting one section.
    Implements level-based difficulty: Beginner favors ending omission.
    Returns: (partial_text_prompt, full_text_answer, missing_part_name)
    """
    import random

    parts = ["beginning", "middle", "ending"]

    # 1. Select the part to be omitted/written by the student based on level
    level_lower = (assignment_level or "beginner").lower()
    if level_lower == 'beginner':
        part_to_omit = 'ending'  # Beginners often find the ending easiest to write
    elif level_lower == 'intermediate':
        # Intermediate: 50% chance Ending, 25% Middle, 25% Beginning
        part_to_omit = random.choice(['ending', 'ending', 'middle', 'beginning'])
    else:  # Advanced
        part_to_omit = random.choice(parts)  # Equal chance for any part

    missing_part_name = part_to_omit.title()

    story_parts = structured_story.copy()

    # The part the student MUST write is replaced by a placeholder in the prompt.
    placeholder = (
        f"\n\n***\n\n"
        f"**[ {missing_part_name.upper()} MISSING! ]**\n\n"
        f"**Your turn! Write the creative {missing_part_name} that happens next.**\n\n"
        f"***"
    )
    story_parts[part_to_omit] = placeholder

    # 2. Join the parts to create the partial prompt
    narrative_sections = []

    # Append/Insert sections ensuring order is maintained
    if part_to_omit != 'beginning':
        narrative_sections.append(f"### The Beginning\n\n{story_parts.get('beginning', '')}")

    if part_to_omit == 'beginning':
        narrative_sections.insert(0, placeholder)

    if part_to_omit != 'middle':
        # If beginning was skipped, middle is section 1 (index 0). If beginning was included, middle is section 1 (index 1).
        current_index = len(narrative_sections) if part_to_omit != 'beginning' else 0
        narrative_sections.insert(current_index, f"### The Middle\n\n{story_parts.get('middle', '')}")
    elif part_to_omit == 'middle':
        # Insert placeholder where middle should be
        current_index = len(narrative_sections)
        narrative_sections.insert(current_index, placeholder)

    if part_to_omit != 'ending':
        narrative_sections.append(f"### The Ending\n\n{story_parts.get('ending', '')}")
    elif part_to_omit == 'ending':
        narrative_sections.append(placeholder)

    # Final cleanup and joining
    partial_narrative = "\n\n---\n\n".join(narrative_sections).strip()

    # 3. Generate the full, complete narrative for the answer key
    full_narrative = (
            (structured_story.get("beginning", "") or "") +
            "\n\n" +
            (structured_story.get("middle", "") or "") +
            "\n\n" +
            (structured_story.get("ending", "") or "")
    ).strip()

    return partial_narrative, full_narrative, missing_part_name


# --- Function: generate_worksheet_payload (Updated) ---
# In app.py, add this new function:

def generate_reading_worksheet(base_text: str, story_level: str) -> dict:
    """
    Generates a comprehensive reading worksheet including MCQ, FIB, and SA questions.
    """
    if client is None:
        print("OpenAI client not configured; cannot generate full reading worksheet.")
        return {"type": "reading", "mcq": [], "fill_in_blank": [], "short_answer": []}

    instructions = """
You are a content generator for a children's English learning platform.
Given a story text, your task is to generate three types of questions:
1. Multiple Choice (MCQ): 5 questions testing comprehension.
2. Fill-in-the-Blank (FIB): 5 sentences directly quoted from the story with one key word removed.
3. Short Answer (SA): 3 questions requiring a short, subjective response.

Rules:
- Output ONLY a single JSON object.
- **FIB ACCURACY FIX:** The sentences for Fill-in-the-Blank MUST be taken VERBATIM from the story text. The missing word should be a critical noun, verb, or adjective, NOT a common article or pronoun.
- The output MUST strictly follow this JSON format:
  {
    "type": "reading",
    "title": "Generated Worksheet Title",
    "mcq": [ {"question": "string", "options": ["A", "B", "C", "D"], "correct_index": 0}, ... ],
    "fill_in_blank": [ {"sentence": "string with ___ placeholder", "answer": "missing word"}, ... ],
    "short_answer": [ {"question": "string", "model_answer": "suggested answer"}, ... ]
  }
"""

    user_prompt = f"""
Story Text:
\"\"\"{base_text}\"\"\"

Student Reading Level: {story_level.title()}.

Please generate:
- 5 Multiple Choice Questions.
- 5 Fill-in-the-Blank sentences (VERBATIM from the story, removing a key word).
- 3 Short Answer Questions.

Ensure the FIB sentences are accurately quoted and the missing word is substantial.
"""

    try:
        resp = client.responses.create(
            model=DEFAULT_MODEL,
            max_output_tokens=1200,
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
        raw = getattr(resp, "output_text", "") or ""

        # Robust JSON extraction
        raw = raw.strip()
        if raw.startswith("```json"):
            raw = raw.strip("```json").strip("```").strip()

        data = json.loads(raw)
        data['type'] = 'reading'  # Ensure type is always set correctly
        return data

    except Exception as e:
        print(f"Error calling OpenAI for Reading Worksheet generation: {e}")
        return {"type": "reading", "mcq": [], "fill_in_blank": [], "short_answer": []}
# In app.py - Replace your existing generate_worksheet_payload function with this:

def generate_worksheet_payload(
        base_text: str,
        worksheet_type: str,
        writing_mode: Optional[str] = "planning",
        writing_level: Optional[str] = "beginner",
) -> dict | None:
    """
    Generates JSON payload exclusively for Writing worksheets (planning or completion).
    Reading assignment generation is handled upstream by generate_reading_worksheet.
    """
    if client is None:
        print("OpenAI client not configured; skipping worksheet generation.")
        return None

    worksheet_type = (worksheet_type or "").strip().lower()
    writing_mode = (writing_mode or "planning").strip().lower()
    writing_level = (writing_level or "beginner").strip().lower()

    if worksheet_type != "writing":
        # Only process if it's explicitly a writing type.
        return None

    if writing_mode == "planning":
        # --- MODE 1: Free Writing Planning Worksheet ---

        return {
            "type": "writing_planning",
            "title": f"{writing_level.title()} Story Planning",
            "sections": [
                {"label": "Beginning Plan", "instruction": f"Plan the start of your {writing_level} story.",
                 "guiding_questions": ["Who is the main character?", "Where and when does the story start?"]},
                {"label": "Middle Plan", "instruction": "What is the main conflict/problem?",
                 "guiding_questions": ["What is the challenge?", "What steps does the character take?"]}
            ],
            "checklist": ["Did I plan a clear problem?", "Did I use descriptive words?"]
        }

    elif writing_mode == "completion":
        # --- MODE 2: Story Completion (Generate Story with Empty Part) ---

        structured_story = make_structured_story(
            prompt=f"A simple story suitable for a {writing_level} reader.",
            level=writing_level
        )

        partial_prompt, full_answer, missing_part = create_finish_prompt(
            structured_story,
            writing_level
        )

        ai_worksheet_data = {
            "instructions": f"Your task is to write the missing **{missing_part}** of the story. Use the clues in the existing text to write a creative and complete section.",
            "guiding_questions": [f"What happened right before the {missing_part}?",
                                  f"How does the {missing_part} resolve the core conflict?",
                                  f"Did I include important characters in the {missing_part}?"],
            "checklist": ["Did I continue the story's tense?", "Does my part connect logically?",
                          "Did I use at least two new descriptive words?"]
        }

        return {
            "type": "writing_completion",
            "story_title": structured_story.get("title", f"Completion Task ({missing_part})"),
            "partial_text_prompt": partial_prompt,
            "full_text_answer": full_answer,
            "missing_part": missing_part,
            "sections": [{"label": f"Plan Your {missing_part}",
                          "guiding_questions": ai_worksheet_data.get("guiding_questions")}],
            "checklist": ai_worksheet_data.get("checklist")
        }

    return None

# -------------------------------------------------------------------
# ADMIN: generate worksheet + assign to students
# -------------------------------------------------------------------
# --- Updated admin_generate_worksheet function in app.py ---
# --- Function: admin_generate_worksheet (Updated) ---
# In app.py - Replace your existing admin_generate_worksheet function with this:

@app.route("/admin/generate-worksheet", methods=["POST"])
@login_required
def admin_generate_worksheet():
    if not current_user_is_admin():
        flash("You do not have permission to create worksheets.", "danger")
        return redirect(url_for("index"))

    db = get_db()

    story_id = request.form.get("story_id", type=int)
    assignment_type = (request.form.get("worksheet_type") or "").strip().lower()
    worksheet_title = (request.form.get("worksheet_title") or "").strip()

    # NEW FIELDS for writing task
    writing_mode = (request.form.get("writing_mode") or "planning").strip().lower()
    writing_level = (request.form.get("writing_level") or "beginner").strip().lower()

    raw_user_ids = request.form.getlist("user_ids") or []
    user_ids = [int(uid) for uid in raw_user_ids if uid]

    # --- Initialization ---
    draft_id = None
    story = None
    questions_json = None

    # --- Basic validation ---
    if assignment_type not in {"writing", "reading"}:
        flash("Please choose a valid worksheet type.", "warning")
        return redirect(url_for("admin_stories", tab="assign"))
    if not user_ids:
        flash("Please choose at least one student to assign the worksheet to.", "warning")
        return redirect(url_for("admin_stories", tab="assign"))

    # --------------------------
    # WRITING ASSIGNMENT SETUP
    # --------------------------
    if assignment_type == "writing":
        # Find or create the dummy story/draft for assignment links
        dummy = db.execute(
            "SELECT id FROM stories WHERE slug = ?", ("free-writing-template",)
        ).fetchone()

        if dummy is None:
            cur = db.execute(
                """INSERT INTO stories (slug, title, language, level, content, created_at)
                   VALUES (?, ?, ?, ?, ?, datetime('now'))""",
                ("free-writing-template", "Creative Writing Practice", "en", "A1", "",),
            )
            story_id = cur.lastrowid
        else:
            story_id = dummy["id"]

        dummy_draft = db.execute(
            "SELECT id FROM finish_drafts WHERE story_id = ? ORDER BY id DESC LIMIT 1",
            (story_id,),
        ).fetchone()

        if dummy_draft is None:
            cur = db.execute(
                """INSERT INTO finish_drafts (story_id, seed_prompt, partial_text, completion_text)
                   VALUES (?, ?, ?, ?)""",
                (story_id, "System generated prompt", "", None),
            )
            draft_id = cur.lastrowid
        else:
            draft_id = dummy_draft["id"]

        # 1. Generate the payload
        worksheet_payload = generate_worksheet_payload(
            base_text="",
            worksheet_type="writing",
            writing_mode=writing_mode,
            writing_level=writing_level
        )

        if not worksheet_payload:
            flash("Failed to generate writing worksheet with AI.", "danger")
            return redirect(url_for("admin_stories", tab="assign"))

        # 2. SPECIAL HANDLING: Completion Mode saves the story content to DB
        if writing_mode == "completion":

            # A. Overwrite the dummy story's content with the *full answer key*
            db.execute(
                "UPDATE stories SET content = ? WHERE id = ?",
                (worksheet_payload["full_text_answer"], story_id)
            )

            # B. Overwrite the dummy draft's text with the *partial prompt* and answer key
            db.execute(
                """
                UPDATE finish_drafts
                SET partial_text = ?, completion_text = ?
                WHERE id = ?
                """,
                (worksheet_payload["partial_text_prompt"], worksheet_payload["full_text_answer"], draft_id)
            )

            # Set the title to reflect the missing part
            if not worksheet_title:
                worksheet_title = worksheet_payload['story_title']

            # Store only the instructions/structure in questions_json for the assignment
            payload_for_db = {
                "type": "writing_completion",
                "sections": worksheet_payload["sections"],
                "checklist": worksheet_payload["checklist"]
            }
            questions_json = json.dumps(payload_for_db, ensure_ascii=False)

        else:  # writing_mode == "planning"
            # Keep original planning logic (no story/draft content updates needed)
            if not worksheet_title:
                worksheet_title = worksheet_payload.get("title") or "Writing Structure Practice"
            questions_json = json.dumps(worksheet_payload, ensure_ascii=False)

    # --------------------------
    # READING ASSIGNMENT SETUP (CALLS generate_reading_worksheet directly)
    # --------------------------
    elif assignment_type == "reading":
        # Check story validity
        if not story_id:
            flash("Please select a story for the reading worksheet.", "warning")
            return redirect(url_for("admin_stories", tab="assign"))

        story = db.execute("SELECT * FROM stories WHERE id = ?", (story_id,)).fetchone()
        draft = db.execute(
            "SELECT id, completion_text FROM finish_drafts WHERE story_id = ? ORDER BY created_at DESC LIMIT 1",
            (story_id,)).fetchone()

        base_text = (draft.get("completion_text") if draft else story.get("content")).strip()
        draft_id = draft["id"] if draft else None

        if not base_text:
            flash("Story text is empty; cannot generate a reading worksheet.", "warning")
            return redirect(url_for("admin_stories", tab="assign"))

        # 1. Generate the FULL structured reading payload (MCQ, FIB, SA)
        worksheet_payload = generate_reading_worksheet(  # <--- CORRECT DIRECT CALL
            base_text=base_text,
            story_level=story["level"]
        )

        if not worksheet_payload or not (
                worksheet_payload.get('mcq') or worksheet_payload.get('fill_in_blank') or worksheet_payload.get(
                'short_answer')):
            flash("AI failed to generate question content for the reading worksheet. Try a different story.", "danger")
            return redirect(url_for("admin_stories", tab="assign"))

        # 2. Save the entire structured payload to the assignment row
        questions_json = json.dumps(worksheet_payload, ensure_ascii=False)

        # 3. Use the generated title if the user didn't provide one
        if not worksheet_title:
            worksheet_title = worksheet_payload.get("title") or f"Reading · {story['title']}"

    # --- INSERT ASSIGNMENTS INTO DB (Common Path) ---
    created = 0
    for uid in user_ids:
        assignee_id = int(uid)
        db.execute(
            """
            INSERT INTO assignments
            (story_id, draft_id, assignee_id, assignment_type,
             questions_json, assigned_by, assignment_title, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'assigned')
            """,
            (
                story_id,
                draft_id,
                assignee_id,
                assignment_type,
                questions_json,
                session.get("user_id"),
                worksheet_title,
            ),
        )
        created += 1

    db.commit()
    flash(f"Worksheet generated and assigned to {created} student(s) successfully.", "success")
    return redirect(url_for("admin_stories", tab="assigned"))

# --- New Route: admin_assignment_group_detail ---

@app.get("/admin/assignments/group/<int:assignment_id>")
@login_required
def admin_assignment_group_detail(assignment_id: int):
    """
    Admin view: Show detail for a specific assignment group (template).
    This retrieves the template information and all student statuses linked to it.
    """
    if not current_user_is_admin():
        flash("Admin access only.", "warning")
        return redirect(url_for("index"))

    db = get_db()

    # 1. Fetch the primary assignment row to get the template key (story, type, title)
    group_template = db.execute(
        """
        SELECT 
            a.story_id, 
            a.assignment_type, 
            a.assignment_title, 
            s.title AS story_title
        FROM assignments a
        JOIN stories s ON a.story_id = s.id
        WHERE a.id = ?
        """,
        (assignment_id,),
    ).fetchone()

    if not group_template:
        flash("Assignment template not found.", "warning")
        return redirect(url_for("admin_stories", tab="assigned"))

    # 2. Fetch ALL assignments matching this template key (all students assigned)
    assignees = db.execute(
        """
        SELECT
            a.id AS assignment_id,
            a.assignee_id,
            a.status,
            a.score,
            a.attempt_count,
            a.created_at,
            u.username,
            u.email
        FROM assignments a
        JOIN users u ON a.assignee_id = u.id
        WHERE a.story_id = ?
          AND a.assignment_type = ?
          AND a.assignment_title = ?
        ORDER BY a.status DESC, u.username ASC
        """,
        (
            group_template["story_id"],
            group_template["assignment_type"],
            group_template["assignment_title"],
        ),
    ).fetchall()

    stats = {
        "total": len(assignees),
        "submitted": sum(1 for a in assignees if a["status"] == "submitted"),
        "graded": sum(1 for a in assignees if a["status"] == "graded"),
        "assigned": sum(1 for a in assignees if a["status"] == "assigned"),
    }

    return render_template(
        "admin_assignment_group_detail.html",
        group_template=group_template,
        assignees=assignees,
        stats=stats,
        assignment_id=assignment_id # Pass original ID for action links
    )


# --- New Route: admin_view_worksheet_content ---

# --- Updated Route: admin_view_worksheet_content in app.py ---

@app.get("/admin/assignments/<int:assignment_id>/content")
@login_required
def admin_view_worksheet_content(assignment_id: int):
    """
    Fetches and formats the questions_json content for a specific assignment template.
    For Writing Completion tasks, it retrieves the story prompt from the linked draft.
    """
    if not current_user_is_admin():
        return jsonify({"error": "Admin access required."}), 403

    db = get_db()

    # 1. Fetch assignment details including the draft_id
    assignment = db.execute(
        """
        SELECT 
            a.assignment_title, 
            a.assignment_type,
            a.questions_json,
            a.draft_id,  -- Retrieve the draft ID
            s.title AS story_title
        FROM assignments a
        JOIN stories s ON a.story_id = s.id
        WHERE a.id = ?
        """,
        (assignment_id,),
    ).fetchone()

    if not assignment:
        return jsonify({"error": "Assignment not found."}), 404

    content_data = {
        "title": assignment["assignment_title"],
        "type": assignment["assignment_type"],
        "story": assignment["story_title"],
        "sections": [],
        "checklist": [],
        "prompt_story": None,  # New field for writing completion prompt
        "raw_text": assignment["questions_json"],
    }

    if assignment["assignment_type"] == 'writing' and assignment["draft_id"]:
        # 2. If writing, fetch the actual story content from the linked draft

        draft = db.execute(
            """
            SELECT partial_text, completion_text 
            FROM finish_drafts 
            WHERE id = ?
            """,
            (assignment["draft_id"],)  # CORRECT: Use the draft_id retrieved from the assignment row
        ).fetchone()

        if draft:
            content_data["prompt_story"] = draft["partial_text"]
            content_data["full_story_answer"] = draft["completion_text"]

            # Check if this is a completion task (based on marker text in the prompt)
            if draft["partial_text"] and 'WRITE THE MISSING' in draft["partial_text"]:
                content_data["type_detail"] = "Story Completion"
            else:
                content_data["type_detail"] = "Free Writing (Planning Only)"

    # 3. Parse and include the worksheet/checklist structure from questions_json
    if assignment["questions_json"]:
        try:
            payload = json.loads(assignment["questions_json"])

            if assignment["assignment_type"] == 'writing':
                # Both planning and completion tasks store sections/checklist here
                content_data["sections"].extend(payload.get("sections", []))
                content_data["checklist"].extend(payload.get("checklist", []))

            elif assignment["assignment_type"] == 'reading':
                # Handles structured reading questions
                content_data["sections"].append({"label": "MCQ Questions", "questions": payload.get("mcq") or []})
                content_data["sections"].append(
                    {"label": "Fill-in-the-Blank", "questions": payload.get("fill_in_blank") or []})
                content_data["sections"].append(
                    {"label": "Short Answer", "questions": payload.get("short_answer") or []})

        except Exception as e:
            content_data["json_parse_error"] = f"Error parsing assignment JSON content: {e}"

    return jsonify(content_data)
# Helper to redirect from assignment ID to the correct submission grading page
# In app.py
@app.get("/admin/assignments/<int:assignment_id>/grade_redirect")
@login_required
def admin_grade_redirect(assignment_id: int):
    """
    Helper route: Finds the latest submission for a given Assignment ID
    and redirects the admin to the dedicated grading page for that Submission ID.
    """
    if not current_user_is_admin():
        flash("Admin access only.", "warning")
        return redirect(url_for("index"))

    db = get_db()

    # Find the latest submission for this specific assignment_id
    # Using created_at based on the previous fix.
    submission = db.execute(
        """
        SELECT id, user_id, assignment_id
        FROM assignment_submissions
        WHERE assignment_id = ?
        ORDER BY datetime(created_at) DESC
        LIMIT 1
        """,
        (assignment_id,),
    ).fetchone()

    if submission:
        return redirect(url_for("admin_grading_page", submission_id=submission["id"]))
    else:
        flash("No submitted work found for this assignment.", "warning")
        return redirect(url_for("assignment_detail", assignment_id=assignment_id))

# In app.py: Add or verify this exact filter definition

@app.template_filter('chr')
def char_filter(value):
    """Makes the Python built-in chr() function available in Jinja."""
    try:
        return chr(value)
    except (TypeError, ValueError):
        return ''@app.context_processor
def inject_global_functions():
    # Expose the Python chr() function directly to the Jinja environment
    return dict(chr=chr)
@app.get("/admin/submissions/<int:submission_id>/grade_page")
@login_required
def admin_grading_page(submission_id: int):
    """
    Admin view: Dedicated page to review and grade a single student submission.
    """
    if not current_user_is_admin():
        flash("Admin access required.", "warning")
        return redirect(url_for("index"))

    db = get_db()

    # --- 1. Fetch Submission Data ---
    submission_row = db.execute(
        """
        SELECT
            sub.id AS submission_id,
            sub.completion_text,
            sub.answers_json,
            sub.score AS current_score,
            sub.comment AS admin_comment,
            sub.created_at AS submitted_at,

            a.id AS assignment_id,
            a.assignment_title,
            a.assignment_type,
            a.questions_json,
            a.draft_id,

            u.username AS assignee_username,
            s.title AS story_title
        FROM assignment_submissions sub
        JOIN assignments a ON sub.assignment_id = a.id
        JOIN users u ON sub.user_id = u.id
        JOIN stories s ON a.story_id = s.id
        WHERE sub.id = ?
        """,
        (submission_id,),
    ).fetchone()

    if not submission_row:
        flash("Submission not found.", "danger")
        return redirect(url_for("admin_stories", tab="submitted"))

    submission = dict(submission_row)

    # Safely load student answers
    try:
        submission['student_answers'] = json.loads(submission['answers_json'])
    except (TypeError, json.JSONDecodeError):
        submission['student_answers'] = {}

    # --- 2. Initialize Assignment Content & Answers for Display ---
    assignment_details = {
        "type_label": submission["assignment_type"].replace('_', ' ').title(),
        "prompt_story": None,
        "sections": [],
        "mcq_questions": [],  # List of MCQ questions for answer cross-reference
    }

    # --- 3. Handle Writing Assignments (to get prompt) ---
    if submission["assignment_type"] == 'writing':
        assignment_details['type_label'] = "Writing & Story Creation"

        # Get story prompt from linked draft
        draft = db.execute(
            "SELECT partial_text FROM finish_drafts WHERE id = ?",
            (submission["draft_id"],)
        ).fetchone()

        if draft:
            assignment_details["prompt_story"] = draft["partial_text"]

    # --- 4. Parse Questions and Structure (Reading/Writing Rubric) ---
    # In app.py - inside the admin_grading_page function, replace your existing parsing block with this:

    # --- 4. Parse Questions and Structure (Reading/Writing Rubric) ---
    if submission["questions_json"]:
        try:
            payload = json.loads(submission["questions_json"])

            # --- FIX FOR DIRECT READING PAYLOAD (like the one provided by the user) ---
            if payload.get("type") == 'reading':

                # Extract lists directly from top-level keys
                mcq_list = payload.get("mcq", [])
                fib_list = payload.get("fill_in_blank", [])
                sa_list = payload.get("short_answer", [])

                # Store the full question lists for cross-referencing and display
                assignment_details["mcq_questions"] = mcq_list

                # Rebuild sections structure for the template display:
                if mcq_list:
                    assignment_details["sections"].append({"label": "MCQ Questions", "questions": mcq_list})
                if fib_list:
                    # Note: The template expects 'sentence' and 'answer' for FIB, which your payload provides.
                    assignment_details["sections"].append({"label": "Fill-in-the-Blank Prompts", "questions": fib_list})
                if sa_list:
                    # Note: The template expects 'question' and 'model_answer' for SA, which your payload provides.
                    assignment_details["sections"].append({"label": "Short Answer Questions", "questions": sa_list})

            elif submission["assignment_type"] == 'reading':
                # This handles cases where the reading questions are nested under a 'sections' key (legacy/alternative structure)

                mcq_list = []
                fib_list = []
                sa_list = []

                for section in payload.get("sections", []):
                    label = section.get('label', '')
                    questions = section.get('questions', [])
                    if 'MCQ' in label:
                        mcq_list.extend(questions)
                    elif 'Fill-in' in label:
                        fib_list.extend(questions)
                    elif 'Short Answer' in label:
                        sa_list.extend(questions)

                assignment_details["mcq_questions"] = mcq_list

                if mcq_list or fib_list or sa_list:
                    # Rebuild sections for the template display using the extracted lists
                    if mcq_list:
                        assignment_details["sections"].append({"label": "MCQ Questions", "questions": mcq_list})
                    if fib_list:
                        assignment_details["sections"].append(
                            {"label": "Fill-in-the-Blank Prompts", "questions": fib_list})
                    if sa_list:
                        assignment_details["sections"].append({"label": "Short Answer Questions", "questions": sa_list})


            elif submission["assignment_type"] == 'writing':
                # Handles standard writing assignment structure (sections for planning/rubric)
                assignment_details["sections"].extend(payload.get("sections", []))

        except Exception as e:
            # Note: Added print for debugging, you can remove this after deployment
            print(f"Error parsing assignment template content: {e}")
            # flash(f"Error parsing assignment template content: {e}", "warning") # Use flash instead of print in production
    # --- 5. Render the dedicated grading page ---
    return render_template(
        "admin_grading_page.html",
        submission=submission,
        assignment_details=assignment_details,
    )
# The original code for admin_review_submission is here.
@app.route("/admin/submissions/<int:submission_id>/review", methods=["POST"])
@login_required
def admin_review_submission(submission_id: int):
    # ... (unchanged logic for permission check, score/comment parsing) ...
    if not current_user_is_admin():
        flash("You do not have permission to review submissions.", "danger")
        return redirect(url_for("index"))

    db = get_db()

    # 점수 파싱
    raw_score = (request.form.get("score") or "").strip()
    comment = (request.form.get("comment") or "").strip()

    try:
        score = int(raw_score)
    except ValueError:
        flash("Score must be an integer between 0 and 100.", "warning")
        # IMPORTANT: Redirect back to the dedicated grading page, not the submission list tab.
        return redirect(url_for("admin_grading_page", submission_id=submission_id))

    if score < 0 or score > 100:
        flash("Score must be between 0 and 100.", "warning")
        # IMPORTANT: Redirect back to the dedicated grading page, not the submission list tab.
        return redirect(url_for("admin_grading_page", submission_id=submission_id))

    # 해당 submission 이 실제로 존재하는지 확인 + assignment_id(user_id) 가져오기
    # FIX: Use assignment_submissions table
    sub = db.execute(
        """
        SELECT assignment_id, user_id
        FROM assignment_submissions
        WHERE id = ?
        """,
        (submission_id,),
    ).fetchone()

    if sub is None:
        flash("Submission not found.", "danger")
        return redirect(url_for("admin_stories", tab="submitted"))

    assignment_id = sub["assignment_id"]
    user_id = sub["user_id"]

    # 1) assignment_submissions 업데이트
    # FIX: Use assignment_submissions table
    db.execute(
        """
        UPDATE assignment_submissions
        SET score       = ?,
            comment     = ?,
            reviewed_at = CURRENT_TIMESTAMP,
            updated_at  = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (score, comment, submission_id),
    )

    # 2) 해당 학생의 assignments 상태도 'graded' 로 변경
    # FIX: Use assignments table
    db.execute(
        """
        UPDATE assignments
        SET status = 'graded', score = ? 
        WHERE id = ?
          AND assignee_id  = ?
        """,
        (score, assignment_id, user_id),
    )

    db.commit()
    flash("Submission graded and feedback saved.", "success")

    # Final redirect back to the Submitted tab, with the accordion expanded.
    return redirect(
        url_for("admin_stories", tab="submitted", fragment=f"submission-{submission_id}")
    )

import re


# In app.py, near other template filters:
# In app.py, near other template filters:

@app.template_filter("clean_whitespace_and_nl")
def clean_whitespace_and_nl(value: str) -> str:
    """
    Cleans excessive newlines, replaces triple+ newlines with double newlines (paragraphs),
    and removes leading/trailing whitespace around each line.
    """
    if not value:
        return ""

    cleaned = re.sub(r'[ \t]+', ' ', value)
    cleaned = re.sub(r'(\s*\n\s*){3,}', '\n\n', cleaned)
    lines = cleaned.split('\n')
    cleaned_lines = [line.strip() for line in lines]

    return '\n'.join(cleaned_lines).strip()


# NOTE: The nl2br filter MUST be present and correctly defined too:
from markupsafe import Markup, escape  # Ensure this import is near the top


@app.template_filter("nl2br")
def nl2br(value: str) -> Markup:
    """Convert newlines to <br> tags, with HTML escaping."""
    if not value:
        return Markup("")
    # Escape HTML, then replace newline chars with <br>
    return Markup(escape(value).replace("\n", Markup("<br>\n")))
@app.get("/admin/submissions/<int:submission_id>/data")
@login_required
def admin_fetch_submission_data(submission_id: int):
    """
    Fetches comprehensive data for a single submission to populate the grading modal.
    """
    if not current_user_is_admin():
        return jsonify({"error": "Admin access required."}), 403

    db = get_db()

    submission = db.execute(
        """
        SELECT
            sub.id AS submission_id,
            sub.completion_text,
            sub.answers_json,
            sub.score AS current_score,
            sub.comment AS admin_comment,
            sub.created_at AS submitted_at,

            a.id AS assignment_id,
            a.assignment_title,
            a.assignment_type,
            a.questions_json,

            u.username AS assignee_username,
            s.title AS story_title
        FROM assignment_submissions sub
        JOIN assignments a ON sub.assignment_id = a.id
        JOIN users u ON sub.user_id = u.id
        JOIN stories s ON a.story_id = s.id
        WHERE sub.id = ?
        """,
        (submission_id,),
    ).fetchone()

    if not submission:
        return jsonify({"error": "Submission not found."}), 404

    # Process question content (for display in modal)
    question_content = {"sections": [], "raw_prompt": None}

    if submission["questions_json"]:
        try:
            payload = json.loads(submission["questions_json"])

            if submission["assignment_type"] == 'writing':
                # For completion tasks, we need the original story prompt from the linked draft.
                # NOTE: This requires fetching the draft again, as the prompt is not in questions_json.
                if 'sections' in payload and payload['sections']:
                    question_content["sections"].extend(payload['sections'])

                # Retrieve story prompt if available (from the *current* draft linked to the assignment)
                draft_info = db.execute(
                    "SELECT partial_text FROM finish_drafts WHERE story_id = ? ORDER BY id DESC LIMIT 1",
                    (submission["story_title"],)
                    # Using story title as placeholder logic needs adjustment for actual ID lookup
                ).fetchone()

                # Assuming simple string replacement for now, if story/draft details are correctly linked.
                # In a robust app, we'd ensure 'a.draft_id' is used here:
                # draft = db.execute("SELECT partial_text FROM finish_drafts WHERE id = ?", (submission["draft_id"],)).fetchone()
                # if draft: question_content["raw_prompt"] = draft["partial_text"]
                question_content[
                    "raw_prompt"] = "Story prompt/context is complex and requires specific lookup."  # Placeholder

            elif submission["assignment_type"] == 'reading':
                # Reading tasks
                question_content["sections"].append({"label": "MCQ Questions", "questions": payload.get("mcq") or []})
                # ... (add FIB/SA logic here if needed for review, similar to admin_view_worksheet_content)

        except Exception:
            question_content["error"] = "Error parsing assignment JSON."

    # Return full data structure
    return jsonify({
        "ok": True,
        "submission": submission,
        "question_content": question_content,
        "student_answers": json.loads(submission["answers_json"]) if submission["answers_json"] else None
    })
# --- End app.py additions ---

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
                "Children's picture-book cover illustration for a story. "
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


@app.get("/admin/users/<int:user_id>/assignments")
@login_required
def admin_user_assignments(user_id: int):
    """
    Admin view: show all assignments for a specific student.
    FIXED: Queries 'assignments' table using 'assignee_id'.
    """
    if not current_user_is_admin():
        flash("Admin access only.", "warning")
        return redirect(url_for("index"))

    db = get_db()

    # Get the student record
    student = db.execute(
        """
        SELECT id, username, email, l1_language, l2_language
        FROM users
        WHERE id = ?
        """,
        (user_id,),
    ).fetchone()

    if not student:
        flash("Student not found.", "warning")
        return redirect(url_for("admin_students"))

    # Fetch this student's assignments
    # FIX: Query assignments table with assignee_id
    assignments = db.execute(
        """
        SELECT
          a.id,
          a.assignment_type, 
          a.status,
          a.score,
          a.attempt_count,
          a.created_at,
          a.assignment_title,
          s.id    AS story_id,
          s.slug  AS story_slug,
          s.title AS story_title
        from assignments a
        JOIN stories s ON a.story_id = s.id
        WHERE a.assignee_id = ? 
        ORDER BY datetime(a.created_at) DESC
        """,
        (user_id,),
    ).fetchall()

    return render_template(
        "admin_user_assignments.html",
        student=student,
        assignments=assignments,
    )


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
      1) Distinct vocab words from stories that were assigned to this user (via assignments)
      2) If not enough, fill with PREPARED_SCRAMBLE_WORDS

    Returns a list of dicts: { "word": "apple" }
    """
    db = get_db()

    # FIX: Use assignments table
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
    easy_words = []  # 4 letters
    medium_words = []  # 5 letters
    hard_words = []  # 6+ letters

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


if __name__ == "__main__":
    with app.app_context():
        init_db()
        upgrade_bookmarks_table()
    app.run(debug=True)