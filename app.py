import os
import re
import json
import sqlite3
import random
from datetime import datetime
from functools import wraps
from urllib.parse import urlparse, urljoin
from typing import Optional
import string
import uuid
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


@app.template_filter('chr')
def char_filter(value):
    try:
        return chr(value)
    except:
        return ''


@app.template_filter('from_json')
def from_json_filter(s):
    if s:
        try:
            if isinstance(s, str): return json.loads(s)
            if isinstance(s, dict): return s
        except:
            return None
    return None


@app.template_filter("nl2br")
def nl2br(value: str) -> Markup:
    if not value: return Markup("")
    return Markup(escape(value).replace("\n", Markup("<br>\n")))


@app.teardown_appcontext
def close_db(exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()


# -------------------------------------------------------------------
# INIT DB (UPDATED for page_images_json)
# -------------------------------------------------------------------
def init_db():
    """Create the tables used by the app and run light migrations."""
    db = get_db()

    # ... (Keep existing CREATE TABLE statements) ...
    db.executescript(
        """
        PRAGMA foreign_keys = ON;
        CREATE TABLE IF NOT EXISTS users (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          username TEXT NOT NULL UNIQUE,
          email TEXT NOT NULL UNIQUE,
          password_hash TEXT NOT NULL,
          role TEXT DEFAULT 'student',
          grade TEXT, school TEXT, subject TEXT,
          l1_language TEXT, l2_language TEXT, age INTEGER, gender TEXT,
          is_english_native INTEGER DEFAULT 0,
          english_exposure_years REAL, english_start_age INTEGER,
          english_learned_where TEXT, english_use_frequency TEXT, english_self_level TEXT,
          created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS stories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            slug TEXT UNIQUE,
            title TEXT,
            prompt TEXT,
            language TEXT DEFAULT 'en',
            level TEXT DEFAULT 'beginner',
            content TEXT,
            visuals TEXT,
            mcq_questions_json TEXT,
            author_name TEXT,
            is_shared_library INTEGER DEFAULT 0,
            shared_class_id INTEGER,
            page_images_json TEXT, -- NEW COLUMN
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS finish_drafts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            story_id INTEGER,
            seed_prompt TEXT,
            partial_text TEXT,
            completion_text TEXT,
            learner_name TEXT,
            language TEXT DEFAULT 'en',
            created_at TEXT DEFAULT (datetime('now'))
        );
        -- ... (Ensure assignment_submissions, assignments, vocab_items, classes, etc. exist) ...
        CREATE TABLE IF NOT EXISTS vocab_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            story_id INTEGER,
            word TEXT,
            definition TEXT, example TEXT,
            definition_ko TEXT, example_ko TEXT,
            picture_url TEXT
        );
        CREATE TABLE IF NOT EXISTS classes (
             id INTEGER PRIMARY KEY AUTOINCREMENT,
             name TEXT, code TEXT UNIQUE, created_by INTEGER,
             created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS class_members (
             class_id INTEGER, user_id INTEGER, role TEXT,
             joined_at TEXT DEFAULT CURRENT_TIMESTAMP,
             PRIMARY KEY(class_id, user_id)
        );
        CREATE TABLE IF NOT EXISTS assignments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            story_id INTEGER, draft_id INTEGER, assignee_id INTEGER,
            assignment_type TEXT, assignment_title TEXT,
            questions_json TEXT, status TEXT DEFAULT 'assigned',
            score REAL, attempt_count INTEGER DEFAULT 0,
            assigned_by INTEGER, class_id INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS assignment_submissions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            assignment_id INTEGER, user_id INTEGER, story_id INTEGER, draft_id INTEGER,
            completion_text TEXT, answers_json TEXT,
            score REAL, comment TEXT,
            story_grammar_json TEXT, story_grammar_total REAL, story_grammar_updated_at TEXT,
            reviewed_at TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP, updated_at TEXT
        );
        CREATE TABLE IF NOT EXISTS library_shares (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          story_id INTEGER NOT NULL,
          class_id INTEGER,
          shared_by INTEGER,
          created_at TEXT DEFAULT CURRENT_TIMESTAMP,
          UNIQUE(story_id, class_id)
        );
        """
    )

    # Migrations for existing tables
    try:
        db.execute("ALTER TABLE stories ADD COLUMN page_images_json TEXT")
    except sqlite3.OperationalError:
        pass

    # ... (Keep existing migrations for other columns) ...
    try:
        db.execute("ALTER TABLE stories ADD COLUMN is_shared_library INTEGER DEFAULT 0")
    except:
        pass
    try:
        db.execute("ALTER TABLE stories ADD COLUMN shared_class_id INTEGER")
    except:
        pass

    db.commit()


with app.app_context():
    init_db()


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

# --- Role helpers (add near your auth helpers) ---

TEACHER_ROLES = {"teacher", "admin", "instructor", "staff"}

def current_user_role() -> str:
    # g.current_user is used throughout your app (e.g., classes routes) :contentReference[oaicite:1]{index=1}
    return (g.current_user.get("role") or "student").strip().lower()

def current_user_is_teacher() -> bool:
    return current_user_role() in TEACHER_ROLES

def current_user_is_student() -> bool:
    return current_user_role() == "student"

def current_user_is_admin() -> bool:
    if session.get("username") == "testtest":
        return True
    return current_user_is_teacher()


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
    role = (data.get("role") or "student").strip().lower()

    l1 = (data.get("l1_language") or "").strip().lower()
    l2 = (data.get("l2_language") or "").strip().lower()
    age_raw = (data.get("age") or "").strip()
    gender = (data.get("gender") or "").strip().lower()

    # Screening (student background)
    exposure_years_raw = (data.get("english_exposure_years") or "").strip()
    learned_where = (data.get("english_learned_where") or "").strip().lower()
    use_freq = (data.get("english_use_frequency") or "").strip().lower()
    self_level = (data.get("english_self_level") or "").strip().lower()
    start_age_raw = (data.get("english_start_age") or "").strip()

    errors = []

    # Required fields
    if role not in {"student", "teacher"}:
        errors.append("Please choose a valid role (Student or Teacher).")

    if not username or not email or not password or not confirm:
        errors.append("Please fill in username, email, and password.")
    if not l1 or not l2 or not age_raw or not gender:
        errors.append("Please fill in L1, L2, age, and gender.")

    # Screening required for students (helps estimate level before the test)
    if role == "student":
        if not exposure_years_raw or not learned_where or not use_freq or not self_level:
            errors.append("Please complete the English background questions (exposure, where you learned, usage, self-level).")

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

    # Screening validation (students only)
    if role == "student":
        # exposure years
        try:
            exposure_years = float(exposure_years_raw)
        except Exception:
            exposure_years = None
        if exposure_years is None or exposure_years < 0 or exposure_years > 60:
            errors.append("English exposure years must be a number between 0 and 60.")

        # start age (optional)
        if start_age_raw:
            if not start_age_raw.isdigit():
                errors.append("English start age must be a number.")
            else:
                start_age = int(start_age_raw)
                if start_age < 0 or start_age > 80:
                    errors.append("English start age must be between 0 and 80.")

        allowed_where = {"school", "academy", "home", "abroad", "online", "other"}
        if learned_where and learned_where not in allowed_where:
            errors.append("Please choose a valid option for where you learned English.")

        allowed_freq = {"never", "rarely", "sometimes", "often", "daily"}
        if use_freq and use_freq not in allowed_freq:
            errors.append("Please choose a valid option for how often you use English.")

        allowed_self = {"beginner", "intermediate", "advanced"}
        if self_level and self_level not in allowed_self:
            errors.append("Please choose a valid self-assessed level.")

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
        def render_fail(msg: str, category: str = "warning"):
            flash(msg, category)
            return render_template(
                "register.html",
                registered=False,
                registered_level_name=None,
                registered_level_score=None,
            )

        # -----------------------------
        # Step 1 fields (shared)
        # -----------------------------
        role = (request.form.get("role") or "").strip().lower() or "student"

        username = (request.form.get("username") or "").strip()
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""
        confirm = request.form.get("confirm") or ""

        l1_language = (request.form.get("l1_language") or "").strip().lower()
        l2_language = (request.form.get("l2_language") or "").strip().lower()
        age_raw = (request.form.get("age") or "").strip()
        gender = (request.form.get("gender") or "").strip().lower()

        # Optional role-specific extras
        grade = (request.form.get("grade") or "").strip().lower()
        school = (request.form.get("school") or "").strip()
        subject = (request.form.get("subject") or "").strip()

        # -----------------------------
        # Screening (student background)
        # -----------------------------
        english_exposure_years_raw = (request.form.get("english_exposure_years") or "").strip()
        english_start_age_raw      = (request.form.get("english_start_age") or "").strip()
        english_learned_where      = (request.form.get("english_learned_where") or "").strip().lower()
        english_use_frequency      = (request.form.get("english_use_frequency") or "").strip().lower()
        english_self_level         = (request.form.get("english_self_level") or "").strip().lower()

        def _to_float_or_none(x: str):
            try:
                return float(x) if x != "" else None
            except ValueError:
                return None

        def _to_int_or_none(x: str):
            try:
                return int(x) if x != "" else None
            except ValueError:
                return None

        english_exposure_years = _to_float_or_none(english_exposure_years_raw)
        english_start_age = _to_int_or_none(english_start_age_raw)

        # -----------------------------
        # Step 2 fields (student only)
        # -----------------------------
        level_score_raw = request.form.get("level_score")
        level_name = (request.form.get("level_name") or "").strip()

        # -----------------------------
        # Validate role
        # -----------------------------
        if role not in {"student", "teacher"}:
            return render_fail("Please select a valid role (Student or Teacher).")

        # -----------------------------
        # Basic required checks
        # -----------------------------
        if not username or not email or not password or not confirm:
            return render_fail("Please fill in username, email, and password.")

        if not l1_language or not l2_language or not age_raw or not gender:
            return render_fail("Please fill in all language and profile fields (L1, L2, age, gender).")

        if password != confirm:
            return render_fail("Passwords do not match.")
        if len(password) < 8:
            return render_fail("Password must be at least 8 characters.")

        if not re.match(r"^[A-Za-z0-9_.-]{3,32}$", username):
            return render_fail("Username must be 3–32 characters and use only letters, numbers, _, ., -.")
        if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
            return render_fail("Please enter a valid email address.")

        allowed_l1 = {"korean", "english", "chinese", "japanese", "spanish", "other"}
        allowed_l2 = {"none", "korean", "english", "chinese", "japanese", "spanish", "other"}
        if l1_language not in allowed_l1:
            return render_fail("Please choose a valid L1 language.")
        if l2_language not in allowed_l2:
            return render_fail("Please choose a valid L2 language.")

        try:
            age = int(age_raw)
        except ValueError:
            return render_fail("Please enter a valid age (number).")
        if age < 5 or age > 120:
            return render_fail("Please enter an age between 5 and 120.")

        allowed_genders = {"female", "male", "nonbinary", "prefer_not"}
        if gender not in allowed_genders:
            return render_fail("Please choose a valid gender option.")

        # -----------------------------
        # Role-specific validation
        # -----------------------------
        level_score = None

        if role == "student":
            # Screening required for students
            if english_exposure_years is None:
                return render_fail("Please enter your English exposure years (number).")
            if english_exposure_years < 0 or english_exposure_years > 60:
                return render_fail("English exposure years must be between 0 and 60.")

            allowed_where = {"school", "academy", "home", "online", "abroad", "other"}
            if english_learned_where not in allowed_where:
                return render_fail("Please choose where you learned English mostly.")

            allowed_freq = {"never", "rarely", "sometimes", "often", "daily"}
            if english_use_frequency not in allowed_freq:
                return render_fail("Please choose how often you use English.")

            allowed_self = {"beginner", "intermediate", "advanced"}
            if english_self_level not in allowed_self:
                return render_fail("Please choose your self-assessed English level.")

            # Level test REQUIRED for students
            if not level_score_raw or not level_name:
                return render_fail("Please complete the level test before creating your account.")

            try:
                level_score = int(level_score_raw)
            except (TypeError, ValueError):
                return render_fail("Level test score is invalid. Please retake the test.")
            if level_score < 0 or level_score > 10:
                return render_fail("Level test score is invalid. Please retake the test.")

            valid_levels = {"Beginner", "Intermediate", "Advanced"}
            if level_name not in valid_levels:
                return render_fail("Level test result is invalid. Please try the test again.")
        else:
            # Teacher: no screening + no level test
            level_score = None
            level_name = None
            english_exposure_years = None
            english_start_age = None
            english_learned_where = None
            english_use_frequency = None
            english_self_level = None

        # -----------------------------
        # Duplicate checks
        # -----------------------------
        existing = db.execute(
            "SELECT id FROM users WHERE lower(username)=?",
            (username.lower(),),
        ).fetchone()
        if existing:
            return render_fail("That username is already taken.", "danger")

        existing = db.execute(
            "SELECT id FROM users WHERE lower(email)=?",
            (email.lower(),),
        ).fetchone()
        if existing:
            return render_fail("An account with that email already exists.", "danger")

        # -----------------------------
        # Insert user
        # -----------------------------
        try:
            pwd_hash = generate_password_hash(password)

            cur = db.execute(
                """
                INSERT INTO users
                  (username, email, password_hash,
                   l1_language, l2_language, age, gender,
                   role, grade, school, subject,
                   english_exposure_years, english_start_age,
                   english_learned_where, english_use_frequency, english_self_level)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    username,
                    email,
                    pwd_hash,
                    l1_language,
                    l2_language,
                    age,
                    gender,
                    role,
                    grade if role == "student" else None,
                    school if role == "teacher" else None,
                    subject if role == "teacher" else None,
                    english_exposure_years,
                    english_start_age,
                    english_learned_where,
                    english_use_frequency,
                    english_self_level,
                ),
            )
            user_id = cur.lastrowid

            # Only students get a level_test_results row
            if role == "student":
                db.execute(
                    """
                    INSERT INTO level_test_results (user_id, score, total, level)
                    VALUES (?, ?, ?, ?)
                    """,
                    (user_id, level_score, 10, level_name),
                )

            db.commit()

            return render_template(
                "register.html",
                registered=True,
                registered_level_name=(level_name if role == "student" else None),
                registered_level_score=(level_score if role == "student" else None),
            )

        except sqlite3.IntegrityError as e:
            db.rollback()
            msg = "Could not create account. Please try again."
            if "users.username" in str(e):
                msg = "That username is already taken."
            elif "users.email" in str(e):
                msg = "An account with that email already exists."
            return render_fail(msg, "danger")

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
# Writing worksheet generator (MAIN story grammar)
# -------------------------------------------------------------------
def generate_main_writing_worksheet(
    story_title: str,
    story_partial_text: str,
    learner_level_hint: str | None = None,
):
    """
    Generate a writing worksheet aligned to MAIN story grammar:
    Character, Setting, Problem/Initiating Event, Actions/Attempts, Resolution.

    Returns a dict (to be JSON-serialized) with:
      {
        "type": "writing_main",
        "sections": [...],
        "checklist": [...],
        "teacher_rubric": {...}
      }

    Notes:
    - This worksheet is used on writing assignments (assignment_type='writing').
    - Output is intentionally UI-friendly: short prompts + sentence starters.
    """
    story_title = (story_title or "").strip() or "Story"
    story_partial_text = (story_partial_text or "").strip()

    # Fallback (no LLM)
    if client is None:
        return {
            "type": "writing_main",
            "version": 1,
            "sections": [
                {
                    "key": "character",
                    "title": "Character",
                    "goal": "Introduce the main character and what they want.",
                    "questions": [
                        "Who is the main character?",
                        "What do they want or care about?",
                        "What is their personality (kind, brave, shy, etc.)?",
                    ],
                    "sentence_starters": [
                        "The main character is ...",
                        "They want to ...",
                        "They feel ... because ...",
                    ],
                },
                {
                    "key": "setting",
                    "title": "Setting",
                    "goal": "Describe where and when the story happens.",
                    "questions": [
                        "Where does the story happen?",
                        "When does it happen (day/night/season)?",
                        "What do you see, hear, or feel in this place?",
                    ],
                    "sentence_starters": [
                        "The story takes place in ...",
                        "It is ... (morning/night/winter).",
                        "The place looks/sounds like ...",
                    ],
                },
                {
                    "key": "problem",
                    "title": "Problem / Initiating Event",
                    "goal": "Explain what goes wrong or what challenge starts the story.",
                    "questions": [
                        "What problem happens?",
                        "Why is it a problem for the character?",
                        "What do they decide to do first?",
                    ],
                    "sentence_starters": [
                        "Suddenly, ...",
                        "This is a problem because ...",
                        "So, the character decides to ...",
                    ],
                },
                {
                    "key": "actions",
                    "title": "Actions / Attempts",
                    "goal": "Write 2–4 attempts the character makes to solve the problem.",
                    "questions": [
                        "What is the first attempt?",
                        "What happens after that?",
                        "Do they try again in a new way?",
                    ],
                    "sentence_starters": [
                        "First, ...",
                        "Then, ...",
                        "After that, ...",
                    ],
                },
                {
                    "key": "resolution",
                    "title": "Resolution",
                    "goal": "Show how the problem ends and what the character learns/feels.",
                    "questions": [
                        "How is the problem solved (or not solved)?",
                        "How does the character feel at the end?",
                        "What did they learn or change?",
                    ],
                    "sentence_starters": [
                        "In the end, ...",
                        "Finally, ...",
                        "The character learned that ...",
                    ],
                },
            ],
            "checklist": [
                "I clearly introduced my main character.",
                "I described where and when the story happens.",
                "I explained the main problem or initiating event.",
                "I wrote several actions/attempts in a clear order (first/then/after).",
                "My ending resolves the problem (or explains why it cannot be solved).",
                "My story is easy to follow and has complete sentences.",
            ],
            "teacher_rubric": {
                "scale": "0–2",
                "character": {"0": "missing", "1": "basic", "2": "clear + detailed"},
                "setting": {"0": "missing", "1": "basic", "2": "clear + sensory details"},
                "problem": {"0": "missing", "1": "basic", "2": "clear cause + stakes"},
                "actions": {"0": "missing", "1": "some attempts", "2": "logical sequence + effort"},
                "resolution": {"0": "missing", "1": "basic", "2": "clear outcome + reflection"},
            },
        }

    level_hint = (learner_level_hint or "").strip() or "elementary/ESL"
    instructions = """
You are an English writing teacher assistant.
Create a short writing worksheet aligned with MAIN story grammar:
Character, Setting, Problem/Initiating Event, Actions/Attempts, Resolution.

Output ONLY valid JSON using this exact top-level schema:
{
  "type": "writing_main",
  "version": 1,
  "sections": [
    {
      "key": "character|setting|problem|actions|resolution",
      "title": "string",
      "goal": "string (1 sentence)",
      "questions": ["string", "..."],
      "sentence_starters": ["string", "..."]
    }
  ],
  "checklist": ["string", "..."],
  "teacher_rubric": {
    "scale": "0–2",
    "character": {"0":"...", "1":"...", "2":"..."},
    "setting": {"0":"...", "1":"...", "2":"..."},
    "problem": {"0":"...", "1":"...", "2":"..."},
    "actions": {"0":"...", "1":"...", "2":"..."},
    "resolution": {"0":"...", "1":"...", "2":"..."}
  }
}

Rules:
- EXACTLY 5 sections, in the order: character, setting, problem, actions, resolution.
- Each section: 3–5 questions and 2–3 sentence starters.
- Keep student language simple and short (A2–B1). Avoid jargon.
- Make prompts fit the specific story context provided.
- Do NOT include markdown, commentary, or extra keys.
"""
    user_prompt = f"""
Story title: {story_title}
Student level hint: {level_hint}

Story excerpt (what the student saw so far):
\"\"\"{story_partial_text}\"\"\"

Task:
Create the worksheet JSON that helps the student write the rest of the story.
"""

    try:
        resp = client.responses.create(
            model=DEFAULT_MODEL,
            max_output_tokens=900,
            temperature=0.4,
            instructions=instructions.strip(),
            input=[{
                "role": "user",
                "content": [{"type": "input_text", "text": user_prompt.strip()}],
            }],
        )
        raw = (getattr(resp, "output_text", "") or "").strip()
        if not raw:
            return generate_main_writing_worksheet(story_title, story_partial_text, learner_level_hint=None)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                return generate_main_writing_worksheet(story_title, story_partial_text, learner_level_hint=None)
            data = json.loads(match.group(0))

        # Minimal validation / normalization
        if not isinstance(data, dict):
            return generate_main_writing_worksheet(story_title, story_partial_text, learner_level_hint=None)

        if data.get("type") != "writing_main":
            data["type"] = "writing_main"
        if data.get("version") is None:
            data["version"] = 1

        sections = data.get("sections") or []
        if not (isinstance(sections, list) and len(sections) == 5):
            return generate_main_writing_worksheet(story_title, story_partial_text, learner_level_hint=None)

        # Ensure keys order
        wanted = ["character", "setting", "problem", "actions", "resolution"]
        fixed = []
        by_key = {str(s.get("key")).strip().lower(): s for s in sections if isinstance(s, dict)}
        for k in wanted:
            s = by_key.get(k) or {"key": k, "title": k.title(), "goal": "", "questions": [], "sentence_starters": []}
            # coerce lists
            s["questions"] = [str(x).strip() for x in (s.get("questions") or []) if str(x).strip()]
            s["sentence_starters"] = [str(x).strip() for x in (s.get("sentence_starters") or []) if str(x).strip()]
            fixed.append(s)
        data["sections"] = fixed

        checklist = data.get("checklist") or []
        data["checklist"] = [str(x).strip() for x in checklist if str(x).strip()][:14]

        return data

    except Exception as e:
        log_input("writing_worksheet_llm_error", {"error": str(e), "title": story_title})
        return generate_main_writing_worksheet(story_title, story_partial_text, learner_level_hint=None)

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
        class_id = request.form.get("class_id")
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
        # Prepare questions_json per assignment type
        # - writing: generate MAIN story-grammar worksheet via GPT
        # - reading: require pre-generated MCQ/worksheet JSON on the story
        # -------------------------------------------------
        questions_json = None

        if assignment_type == "writing":
            partial_text = ""
            try:
                partial_text = (draft.get("partial_text") or "").strip()
            except Exception:
                partial_text = ""

            worksheet_payload = generate_main_writing_worksheet(
                story_title=story.get("title") or "Story",
                story_partial_text=partial_text,
                learner_level_hint=None,
            )
            questions_json = json.dumps(worksheet_payload, ensure_ascii=False)

        elif assignment_type == "reading":
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
                return redirect(url_for("admin_story_detail", slug=slug))# Prevent duplicate assignments for same story+type+student
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
                 questions_json, assigned_by, assignment_title,class_id)
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
                    class_id
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

        # MAIN story grammar analytics (from writing submissions)
        grammar_rows = db.execute(
            """
            SELECT story_grammar_json
            FROM assignment_submissions
            WHERE user_id = ? AND story_grammar_json IS NOT NULL
            ORDER BY datetime(updated_at) DESC
            """,
            (selected_user_id,),
        ).fetchall()

        grammar_scores_sum = {"character": 0, "setting": 0, "problem": 0, "actions": 0, "resolution": 0}
        grammar_n = 0
        for r in grammar_rows:
            try:
                payload = json.loads(r["story_grammar_json"]) if r["story_grammar_json"] else None
                scores = (payload or {}).get("scores") or {}
                if not isinstance(scores, dict):
                    continue
                for k in grammar_scores_sum.keys():
                    grammar_scores_sum[k] += int(scores.get(k, 0) or 0)
                grammar_n += 1
            except Exception:
                continue

        grammar_avg = None
        if grammar_n > 0:
            grammar_avg = {k: round(v / float(grammar_n), 2) for k, v in grammar_scores_sum.items()}

        user_data = {
            "user": selected_user,
            "stats": user_stats,
            "current_level": current_level,
            "story_grammar_avg": grammar_avg,
            "story_grammar_count": grammar_n,
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
    user_id = g.current_user["id"]

    # classes user belongs to (for the dropdown)
    my_classes = db.execute(
        """
        SELECT c.id, c.name
        FROM classes c
        JOIN class_members cm ON cm.class_id = c.id
        WHERE cm.user_id = ?
        ORDER BY c.name COLLATE NOCASE
        """,
        (user_id,),
    ).fetchall()

    selected_class_id = request.args.get("class_id", type=int)

    # If user selected a class, ensure membership
    if selected_class_id:
        ok = db.execute(
            "SELECT 1 FROM class_members WHERE user_id=? AND class_id=?",
            (user_id, selected_class_id),
        ).fetchone()
        if not ok:
            flash("You don’t have access to that class.", "warning")
            return redirect(url_for("assignments_list"))

    where_extra = ""
    params = [user_id]

    if selected_class_id:
        where_extra = " AND a.class_id = ? "
        params.append(selected_class_id)

    rows = db.execute(
        f"""
        SELECT
          a.id,
          a.assignment_type,
          a.status,
          a.score,
          a.attempt_count,
          a.created_at,
          a.draft_id,
          a.assignment_title,
          a.class_id,
          c.name AS class_name,
          s.id    AS story_id,
          s.slug  AS story_slug,
          s.title AS story_title
        FROM assignments a
        JOIN stories s ON a.story_id = s.id
        LEFT JOIN classes c ON c.id = a.class_id
        WHERE a.assignee_id = ?
        {where_extra}
        ORDER BY datetime(a.created_at) DESC
        """,
        tuple(params),
    ).fetchall()

    return render_template(
        "assignments.html",
        assignments=rows,
        my_classes=my_classes,
        selected_class_id=selected_class_id,
    )


@app.route("/assignments/<int:assignment_id>", methods=["GET", "POST"])
@login_required
def assignment_detail(assignment_id: int):
    db = get_db()
    user_id = g.current_user["id"]

    # 1. Fetch Assignment & Story
    assignment = db.execute(
        "SELECT * from assignments WHERE id = ? AND assignee_id = ?",
        (assignment_id, user_id),
    ).fetchone()

    if not assignment:
        flash("Assignment not found.", "warning")
        return redirect(url_for("assignments_list"))

    story = db.execute(
        "SELECT * FROM stories WHERE id = ?",
        (assignment["story_id"],),
    ).fetchone()

    draft = None
    if assignment.get("draft_id"):
        draft = db.execute(
            "SELECT * FROM finish_drafts WHERE id = ?",
            (assignment["draft_id"],),
        ).fetchone()

    submission = db.execute(
        "SELECT * from assignment_submissions WHERE assignment_id = ? AND user_id = ? ORDER BY datetime(updated_at) DESC LIMIT 1",
        (assignment_id, user_id),
    ).fetchone()

    # ------------------------------------------------------------------
    # TYPE A: WRITING ASSIGNMENT
    # ------------------------------------------------------------------
    if assignment["assignment_type"] == "writing":
        writing_sections = []
        writing_checklist = []
        if assignment.get("questions_json"):
            try:
                wd = json.loads(assignment["questions_json"])
                writing_sections = wd.get("sections") or []
                writing_checklist = wd.get("checklist") or []
            except Exception:
                pass

        if request.method == "POST":
            completion_text = (request.form.get("completion_text") or "").strip()
            if not completion_text:
                flash("Please write your ending before submitting.", "warning")
                return redirect(request.url)

            # Basic grammar check placeholder
            grammar = analyze_main_story_grammar(completion_text)
            grammar_json = json.dumps(grammar, ensure_ascii=False) if grammar else None

            grammar_total = None
            if grammar and isinstance(grammar.get("scores"), dict):
                try:
                    grammar_total = float(sum(int(v) for v in grammar["scores"].values()))
                except Exception:
                    grammar_total = None

            now = datetime.utcnow().isoformat(timespec="seconds")

            if submission:
                db.execute(
                    """
                    UPDATE assignment_submissions
                    SET completion_text = ?, story_grammar_json = ?, story_grammar_total = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (completion_text, grammar_json, grammar_total, now, submission["id"]),
                )
            else:
                db.execute(
                    """
                    INSERT INTO assignment_submissions
                    (assignment_id, user_id, story_id, draft_id, completion_text, story_grammar_json, story_grammar_total, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (assignment_id, user_id, assignment["story_id"], assignment["draft_id"], completion_text,
                     grammar_json, grammar_total, now, now),
                )

            db.execute(
                "UPDATE assignments SET status='submitted', attempt_count=COALESCE(attempt_count, 0)+1 WHERE id=?",
                (assignment_id,))
            db.commit()

            flash("Story submitted successfully!", "success")
            return redirect(url_for("assignments_list"))

        return render_template(
            "assignment_finish.html",
            assignment=assignment, story=story, draft=draft, submission=submission,
            writing_sections=writing_sections, writing_checklist=writing_checklist
        )

    # ------------------------------------------------------------------
    # TYPE B: READING ASSIGNMENT (FIXED FOR V3)
    # ------------------------------------------------------------------
    elif assignment["assignment_type"] == "reading":
        mcq_questions = []
        fill_in_blank_questions = []
        short_answer_questions = []

        # 1. Parse Questions
        if assignment.get("questions_json"):
            try:
                payload = json.loads(assignment["questions_json"])
                q_type = payload.get("type")

                # Strategy A: V3 Structure (comprehension list)
                if q_type == "reading_v3" or "comprehension" in payload:
                    comp_list = payload.get("comprehension") or []
                    # Sort V3 questions into buckets
                    for q in comp_list:
                        fmt = q.get("format", "").lower()
                        if fmt == "mcq":
                            mcq_questions.append(q)
                        elif fmt == "fill_in_blank":
                            fill_in_blank_questions.append(q)
                        else:
                            short_answer_questions.append(q)

                    # Add expression questions to Short Answer list
                    expr_list = payload.get("expression") or []
                    short_answer_questions.extend(expr_list)

                # Strategy B: Direct Lists (Standard Reading Generator)
                elif "mcq" in payload or "short_answer" in payload:
                    mcq_questions = payload.get("mcq") or []
                    fill_in_blank_questions = payload.get("fill_in_blank") or []
                    short_answer_questions = payload.get("short_answer") or []

                # Strategy C: Legacy Flat List
                elif isinstance(payload, list):
                    mcq_questions = payload

            except Exception as e:
                print(f"Reading parse error: {e}")

        # 2. Handle Submission
        if request.method == "POST":
            answers = []
            correct_count = 0
            mcq_breakdown = {"factual": {"correct": 0, "total": 0}, "inference": {"correct": 0, "total": 0},
                             "other": {"correct": 0, "total": 0}}

            # A. Process MCQ Answers
            if mcq_questions:
                for idx, q in enumerate(mcq_questions):
                    ans_raw = request.form.get(f"q{idx}")

                    # Score Categorization
                    q_cat = (q.get("question_type") or q.get("category") or "other").lower()
                    bucket = "other"
                    if "fact" in q_cat:
                        bucket = "factual"
                    elif "infer" in q_cat:
                        bucket = "inference"
                    mcq_breakdown[bucket]["total"] += 1

                    try:
                        ans_idx = int(ans_raw)
                    except (TypeError, ValueError):
                        ans_idx = None

                    # SAVE ANSWER INDEX
                    answers.append(ans_idx)

                    # Grading
                    try:
                        correct_index = int(q.get("correct_index", -1))
                    except (ValueError, TypeError):
                        correct_index = -1

                    if ans_idx is not None and ans_idx == correct_index:
                        correct_count += 1
                        mcq_breakdown[bucket]["correct"] += 1

                total_mcqs = len(mcq_questions)
                score = (correct_count / total_mcqs) * 100.0 if total_mcqs > 0 else 0.0
            else:
                score = 0.0  # Score is 0 if only short answers exist (pending grading)

            # B. Process Text Answers
            short_responses = []
            for i in range(len(short_answer_questions)):
                short_responses.append(request.form.get(f"short{i}", ""))

            fill_responses = []
            for i in range(len(fill_in_blank_questions)):
                fill_responses.append(request.form.get(f"fill{i}", ""))

            # C. Capture Legacy/Custom Answers
            custom_answers = {}
            for key, val in request.form.items():
                if key.startswith("custom_answer_"):
                    # key like "custom_answer_0" -> store "0": "value"
                    idx_str = key.replace("custom_answer_", "")
                    custom_answers[idx_str] = val

            all_answers = {
                "mcq_answers": answers,
                "mcq_breakdown": mcq_breakdown,
                "fill_in_blank_responses": fill_responses,
                "short_answer_responses": short_responses,
                "custom_answers": custom_answers
            }

            now = datetime.utcnow().isoformat(timespec="seconds")

            # Update or Insert Submission
            if submission:
                db.execute(
                    "UPDATE assignment_submissions SET answers_json=?, score=?, updated_at=? WHERE id=?",
                    (json.dumps(all_answers), score, now, submission["id"]),
                )
            else:
                db.execute(
                    "INSERT INTO assignment_submissions (assignment_id, user_id, story_id, draft_id, answers_json, score, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (assignment_id, user_id, assignment["story_id"], assignment["draft_id"], json.dumps(all_answers),
                     score, now, now),
                )

            # Mark assignment as submitted
            db.execute(
                "UPDATE assignments SET status='submitted', score=?, attempt_count=COALESCE(attempt_count, 0)+1 WHERE id=?",
                (score, assignment_id)
            )
            db.commit()

            # AJAX Response
            if request.headers.get("X-Requested-With") == "XMLHttpRequest":
                return jsonify({"success": True, "score": score})

            return redirect(url_for('assignment_detail', assignment_id=assignment_id))

        # GET Request
        return render_template(
            "assignment_mcq.html",
            assignment=assignment,
            story=story,
            mcq_questions=mcq_questions,
            fill_in_blank_questions=fill_in_blank_questions,
            short_answer_questions=short_answer_questions,
            submission=submission,
            just_submitted=False
        )

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


@app.get("/api/assignment_review/<int:assignment_id>")
@login_required
def api_get_assignment_review(assignment_id):
    """Fetch the score and teacher comment for a specific assignment."""
    db = get_db()
    user_id = g.current_user["id"]

    # We look for the latest submission for this assignment and user
    row = db.execute(
        """
        SELECT score, comment 
        FROM assignment_submissions 
        WHERE assignment_id = ? AND user_id = ?
        ORDER BY created_at DESC LIMIT 1
        """,
        (assignment_id, user_id),
    ).fetchone()

    if not row:
        return jsonify({"error": "No submission found"}), 404

    return jsonify({
        "score": row["score"] if row["score"] is not None else 0,
        "comment": row["comment"] or "The teacher hasn't left a comment yet, but you've been graded!"
    })

@app.route("/story/new", methods=["GET", "POST"])
@login_required
def story_new():
    db = get_db()
    user_id = g.current_user["id"]

    my_classes = db.execute(
        "SELECT c.id, c.name FROM classes c JOIN class_members cm ON cm.class_id = c.id WHERE cm.user_id = ?",
        (user_id,),
    ).fetchall()

    if request.method == "POST":
        title = (request.form.get("title") or "").strip() or "My Story"
        prompt = (request.form.get("prompt") or "").strip()
        base_author = (request.form.get("author_name") or "").strip()
        english_level = (request.form.get("english_level") or "beginner").strip().lower()
        share_class_id = request.form.get("share_class_id")

        # NEW: Capture the multi-image generation mode
        gen_images_mode = request.form.get("gen_images_mode") == "all"

        # ... (keep your existing meta_bits and gen_prompt logic) ...
        meta_bits = [f"Level: {english_level}", f"Title: {title}"]
        gen_prompt = prompt + "\n\n" + " ".join(meta_bits)

        try:
            profile = get_learner_profile()
            content = llm_story_from_prompt(gen_prompt, "en", "", base_author, learner_profile=profile)
        except Exception as e:
            content = naive_story_from_prompt(prompt, "en")

        # --- IMAGE GENERATION LOGIC ---
        visuals_data_url = None  # Cover
        page_images_list = []  # Per-page images

        if client is not None:
            # 1. Generate Cover Image (Visuals)
            try:
                cover_prompt = f"Children's book cover illustration, {title}. High quality, no text."
                img_resp = client.images.generate(model="gpt-image-1", prompt=cover_prompt, n=1)
                visuals_data_url = img_resp.data[0].url
            except Exception:
                pass

            # 2. Generate Multi-Page Images (if requested)
            if gen_images_mode:
                paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
                # Limit to 5 pages to manage API costs/time
                for para in paragraphs[:5]:
                    try:
                        page_resp = client.images.generate(
                            model="dall-e-2",
                            prompt=f"Children's book illustration for: {para[:300]}",
                            size="512x512", n=1
                        )
                        page_images_list.append(page_resp.data[0].url)
                    except Exception:
                        page_images_list.append(None)  # Keep list indices aligned

        # Save to DB
        # Save to DB
        cur = db.execute(
            """INSERT INTO stories (title, slug, prompt, language,english_level , content, visuals, page_images_json, author_name, shared_class_id, is_shared_library)
               VALUES (?, ?, ?, ?,?, ?, ?, ?, ?, ?, 1)""",
            (
                title,
                f"{slugify(title)}-{uuid.uuid4().hex[:6]}",
                prompt,
                "en",
                english_level,
                content,
                visuals_data_url,
                json.dumps(page_images_list),
                base_author,
                share_class_id
            )
        )
        story_id = cur.lastrowid  # Get the ID of the story we just made
        db.execute(
            """
            INSERT INTO finish_drafts (story_id, seed_prompt, partial_text, completion_text, learner_name, language)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                story_id,
                prompt,
                content[:200] + "...",  # A preview
                content,  # The full text
                base_author or "AI",
                "en"
            )
        )
        db.commit()
        return redirect(url_for("library"))

    return render_template("story_new.html", my_classes=my_classes)


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

def table_exists(db, name: str) -> bool:
    row = db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (name,),
    ).fetchone()
    return row is not None


def column_exists(db, table: str, column: str) -> bool:
    cols = db.execute(f"PRAGMA table_info({table})").fetchall()
    return any(c["name"] == column for c in cols)


def ensure_library_schema():
    """
    Backward-compatible schema + backfill for Library sharing.
    Also normalizes legacy shared_class_id=0 to NULL (meaning "All").
    """
    db = get_db()

    # Ensure columns on stories
    if table_exists(db, "stories"):
        if not column_exists(db, "stories", "is_shared_library"):
            db.execute("ALTER TABLE stories ADD COLUMN is_shared_library INTEGER DEFAULT 0")
        if not column_exists(db, "stories", "shared_class_id"):
            db.execute("ALTER TABLE stories ADD COLUMN shared_class_id INTEGER")

        # Normalize legacy "0 means ALL" into NULL
        db.execute("UPDATE stories SET shared_class_id = NULL WHERE shared_class_id = 0")

    # Ensure library_shares table (share rows: class_id NULL => ALL)
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS library_shares (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          story_id INTEGER NOT NULL,
          class_id INTEGER,
          created_at TEXT DEFAULT CURRENT_TIMESTAMP,
          UNIQUE(story_id, class_id)
        )
        """
    )

    # Normalize legacy 0 there too (if it ever happened)
    db.execute("UPDATE library_shares SET class_id = NULL WHERE class_id = 0")

    # Backfill: if a story is shared but has no share row, create it
    # Use stories.shared_class_id (already normalized so 0 => NULL)
    db.execute(
        """
        INSERT OR IGNORE INTO library_shares (story_id, class_id)
        SELECT s.id, s.shared_class_id
        FROM stories s
        WHERE COALESCE(s.is_shared_library, 0) = 1
        """
    )

    db.commit()
@app.get("/library")
@login_required
def library():
    db = get_db()
    user_id = g.current_user["id"]

    ensure_library_schema()

    # Classes user belongs to (for tabs)
    my_classes = db.execute(
        """
        SELECT c.id, c.name
        FROM classes c
        JOIN class_members cm ON cm.class_id = c.id
        WHERE cm.user_id = ?
        ORDER BY c.name COLLATE NOCASE
        """,
        (user_id,),
    ).fetchall()
    my_class_ids = [c["id"] for c in my_classes]

    # Template uses selected_class_id
    class_id_raw = (request.args.get("class_id") or "").strip()
    selected_class_id = int(class_id_raw) if class_id_raw.isdigit() else None

    # Security: if class tab, user must belong (unless teacher/admin)
    if selected_class_id and not (current_user_is_admin() or current_user_is_teacher()):
        ok = db.execute(
            "SELECT 1 FROM class_members WHERE user_id=? AND class_id=? LIMIT 1",
            (user_id, selected_class_id),
        ).fetchone()
        if not ok:
            flash("You don’t have access to that class.", "warning")
            return redirect(url_for("library"))

    # Treat "0 as ALL" at query-time too (extra safety):
    share_expr = "NULLIF(COALESCE(ls.class_id, s.shared_class_id), 0)"

    where = ["COALESCE(s.is_shared_library, 0) = 1"]
    params = []

    if selected_class_id:
        # Only that class
        where.append(f"{share_expr} = ?")
        params.append(selected_class_id)
    else:
        # ALL tab: global (NULL) OR any class user is in
        if my_class_ids:
            placeholders = ",".join("?" for _ in my_class_ids)
            where.append(f"({share_expr} IS NULL OR {share_expr} IN ({placeholders}))")
            params.extend(my_class_ids)
        else:
            where.append(f"{share_expr} IS NULL")

    stories = db.execute(
        f"""
        SELECT
          s.id,
          s.slug,
          s.title,
          s.language,
          s.level,
          s.visuals,
          s.author_name,
          s.created_at AS story_created_at,
          (
            SELECT fd2.id
            FROM finish_drafts fd2
            WHERE fd2.story_id = s.id
            ORDER BY datetime(fd2.created_at) DESC, fd2.id DESC
            LIMIT 1
          ) AS draft_id
        FROM stories s
        LEFT JOIN library_shares ls ON ls.story_id = s.id
        WHERE {" AND ".join(where)}
        GROUP BY s.id
        ORDER BY datetime(s.created_at) DESC, s.id DESC
        """,
        tuple(params),
    ).fetchall()

    # If a story has no draft, the card link will break; hide those
    stories = [r for r in stories if r["draft_id"] is not None]

    return render_template(
        "library.html",
        stories=stories,
        my_classes=my_classes,
        selected_class_id=selected_class_id,
        is_admin=current_user_is_admin(),
        is_teacher=current_user_is_teacher(),
    )


# --- app2.py addition ---

# --- app2.py addition  ---
@app.get("/book/<int:draft_id>")
@login_required
def book_view(draft_id: int):
    db = get_db()
    draft = db.execute("SELECT * FROM finish_drafts WHERE id = ?", (draft_id,)).fetchone()
    story = db.execute("SELECT * FROM stories WHERE id = ?", (draft["story_id"],)).fetchone()

    full_content = (draft.get("completion_text") or story.get("content") or "").strip()
    pages = [p.strip() for p in full_content.split("\n\n") if p.strip()]

    # NEW: Extract the per-page images
    page_images = []
    if story.get("page_images_json"):
        try:
            page_images = json.loads(story["page_images_json"])
        except:
            pass

    return render_template(
        "book_view.html",
        story=story,
        pages=pages,
        page_images=page_images,  # Pass to JS
        vocab=db.execute("SELECT * FROM vocab_items WHERE story_id=?", (story["id"],)).fetchall()
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

    # -------------------------------
    # NEW: Class selector
    # -------------------------------
    classes = db.execute(
        """
        SELECT c.id, c.name
        FROM classes c
        ORDER BY c.name COLLATE NOCASE
        """
    ).fetchall()

    selected_class_id = request.args.get("class_id", type=int)

    # -------------------------------
    # TAB 1: Story list
    # -------------------------------
    stories = db.execute(
        """
        SELECT s.id, s.slug, s.title, s.language, s.level, s.created_at
        FROM stories s
        ORDER BY s.created_at DESC
        """
    ).fetchall()

    # -------------------------------
    # NEW: Users list filtered by class
    # - only students in selected class
    # -------------------------------
    if selected_class_id:
        users = db.execute(
            """
            SELECT u.id, u.username, u.email
            FROM users u
            JOIN class_members cm ON cm.user_id = u.id
            WHERE cm.class_id = ?
              AND (cm.role = 'student' OR cm.role IS NULL)
            ORDER BY u.username COLLATE NOCASE
            """,
            (selected_class_id,),
        ).fetchall()
    else:
        # If no class selected, show none (force admin to select a class)
        users = []

    # -------------------------------
    # TAB 2: Assigned Worksheets (grouped)
    # -------------------------------
    params = []
    where_class = ""
    if selected_class_id:
        where_class = "WHERE a.class_id = ?"
        params.append(selected_class_id)

    assignment_rows = db.execute(
        f"""
        SELECT
            a.id               AS assignment_id,
            a.story_id         AS story_id,
            a.assignment_type  AS assignment_type,
            a.assignment_title AS assignment_title,
            a.created_at       AS assignment_created_at,
            a.class_id         AS class_id,

            s.title            AS story_title,
            s.language         AS language,
            s.level            AS level,

            a.assignee_id      AS assignee_id,
            a.status           AS assignee_status,
            a.score            AS assignee_score,
            a.attempt_count    AS assignee_attempts,

            u.username         AS assignee_username,
            u.email            AS assignee_email,

            c.name             AS class_name
        FROM assignments a
        JOIN stories s ON a.story_id = s.id
        JOIN users u   ON a.assignee_id = u.id
        LEFT JOIN classes c ON c.id = a.class_id
        {where_class}
        ORDER BY a.created_at DESC, a.id DESC
        """,
        tuple(params),
    ).fetchall()

    assignment_groups = {}
    for row in assignment_rows:
        # NEW: include class_id in grouping key to avoid mixing classes
        key = (row["class_id"], row["story_id"], row["assignment_type"], row["assignment_title"])

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
                "primary_assignment_id": row["assignment_id"],
                "class_id": row["class_id"],
                "class_name": row["class_name"],
                "assignees": [],
                "count_assigned": 0,
                "count_submitted": 0,
                "count_graded": 0,
            }

        g = assignment_groups[key]
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

    assignments = sorted(
        assignment_groups.values(),
        key=lambda x: x["created_at"] or "",
        reverse=True,
    )

    # -------------------------------
    # TAB 3: Submitted Work filtered by class
    # -------------------------------
    sub_params = []
    sub_where = ""
    if selected_class_id:
        sub_where = "AND a.class_id = ?"
        sub_params.append(selected_class_id)

    submissions_to_review = db.execute(
        f"""
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
        JOIN assignments a ON ws.assignment_id = a.id
        JOIN stories s ON a.story_id = s.id
        JOIN users u ON ws.user_id = u.id
        WHERE (ws.completion_text IS NOT NULL OR ws.answers_json IS NOT NULL)
        {sub_where}
        ORDER BY ws.created_at DESC
        """,
        tuple(sub_params),
    ).fetchall()

    return render_template(
        "admin_stories.html",
        stories=stories,
        users=users,
        assignments=assignments,
        submissions_to_review=submissions_to_review,
        active_tab=active_tab,
        fragment=fragment,
        classes=classes,  # NEW
        selected_class_id=selected_class_id,  # NEW
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

def get_openai_client():
    """
    Minimal OpenAI client factory.
    Uses OPENAI_API_KEY from environment.
    """
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY environment variable.")
    return OpenAI(api_key=api_key)

# --- New Helper: make_structured_story ---
# --- Updated make_structured_story function in app.py ---
# --- Helper: make_structured_story ---
def make_structured_story(prompt: str, level: str) -> dict | None:
    """
    Generate a story as JSON with keys: title, beginning, middle, ending.

    IMPORTANT:
    - The *values* for beginning/middle/ending must be plain story text.
    - They must NOT include labels like "Beginning:", "Middle:", "End:" inside the text.
    - Returns None if GPT is unavailable or parsing fails (no fallback).
    """
    import json
    import re

    def _strip_leading_labels(s: str) -> str:
        if not s:
            return ""
        s = s.strip()
        # Remove accidental leading labels if GPT still adds them
        s = re.sub(r'^(beginning|middle|ending|end)\s*[:\-–]\s*', '', s, flags=re.IGNORECASE).strip()
        s = re.sub(r'^(beginning|middle|ending|end)\s+', '', s, flags=re.IGNORECASE).strip()
        return s

    if client is None:
        print("[GPT][story] ERROR: OpenAI client is not initialized. Story generation requires GPT (no fallback).")
        return None

    # Define required JSON structure (kept loose; we validate keys ourselves)
    json_schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "beginning": {"type": "string"},
            "middle": {"type": "string"},
            "ending": {"type": "string"}
        },
        "required": ["title", "beginning", "middle", "ending"]
    }

    system_instructions = (
        "You are a helpful children's story writer. "
        "Return ONLY valid JSON. No markdown. No extra commentary."
    )

    user_instructions = f"""
Write a short story suitable for English learners at level: {level}.

Prompt/theme:
{prompt}

Return ONLY valid JSON matching:
{{
  "title": "...",
  "beginning": "...",
  "middle": "...",
  "ending": "..."
}}

Rules:
- The values for beginning/middle/ending must be plain story text.
- Do NOT include section labels like "Beginning", "Middle", "End", or headings inside the values.
- Keep each section concise and coherent (beginning introduces characters/setting, middle introduces problem/attempts, ending resolves).
"""

    try:
        print(f"[GPT][story] Calling GPT for structured story. level={level}")
        # Use the same client style already used in the app
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.8,
            messages=[
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": user_instructions},
            ],
            response_format={"type": "json_object"},
        )
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            print("[GPT][story] ERROR: Empty response from GPT.")
            return None

        data = json.loads(text)

        # Validate keys
        title = (data.get("title") or "").strip()
        beginning = _strip_leading_labels(data.get("beginning", ""))
        middle = _strip_leading_labels(data.get("middle", ""))
        ending = _strip_leading_labels(data.get("ending", ""))

        if not (title and beginning and middle and ending):
            print("[GPT][story] ERROR: Missing one or more required fields after parsing.")
            return None

        # Extra guard: if GPT still injected labels inside content, strip common first-line headings
        beginning = beginning.replace("### The Beginning", "").strip()
        middle = middle.replace("### The Middle", "").strip()
        ending = ending.replace("### The Ending", "").strip()

        print("[GPT][story] SUCCESS: Structured story JSON parsed.")
        return {"title": title, "beginning": beginning, "middle": middle, "ending": ending}

    except Exception as e:
        print(f"[GPT][story] ERROR: {e}")
        return None

def create_finish_prompt(structured_story: dict, assignment_level: str) -> tuple[str, str, str]:
    """
    Creates a partial story prompt by omitting one section (beginning/middle/ending).
    Returns:
      (partial_text_prompt, full_text_answer, missing_part_name)

    partial_text_prompt format is compatible with finish_view.html:
      ### The Beginning ---\n<text>
      ### The Middle ---\n<text or MISSING...>
      ### The Ending ---\n<text>
    """
    import random

    parts = ["beginning", "middle", "ending"]

    level_lower = (assignment_level or "beginner").lower()
    if level_lower == "beginner":
        part_to_omit = "ending"
    elif level_lower == "intermediate":
        part_to_omit = random.choice(["ending", "ending", "middle", "beginning"])
    else:
        part_to_omit = random.choice(parts)

    missing_part_name = part_to_omit.title()  # Beginning/Middle/Ending

    # Placeholder shown to student
    placeholder = f"MISSING: Write the {missing_part_name.upper()} here."

    story_parts = structured_story.copy()
    story_parts[part_to_omit] = placeholder

    blocks = []

    # Keep stable order: Beginning -> Middle -> Ending
    blocks.append("### The Beginning ---\n" + (story_parts.get("beginning", "") or "").strip())
    blocks.append("### The Middle ---\n" + (story_parts.get("middle", "") or "").strip())
    blocks.append("### The Ending ---\n" + (story_parts.get("ending", "") or "").strip())

    partial_narrative = "\n\n".join(blocks).strip()

    full_narrative = (
        (structured_story.get("beginning", "") or "").strip()
        + "\n\n"
        + (structured_story.get("middle", "") or "").strip()
        + "\n\n"
        + (structured_story.get("ending", "") or "").strip()
    ).strip()

    return partial_narrative, full_narrative, missing_part_name



# --- Function: generate_worksheet_payload (Updated) ---
# In app.py, add this new function:
# -------------------------------------------------------------------
#  Reading Worksheet Generator (Updated for Factual/Inferential/Critical)
# -------------------------------------------------------------------
# In app.py

def generate_reading_worksheet(base_text: str, story_level: str, worksheet_level: str | None = None) -> dict:
    """
    Generates a reading worksheet with a longer story and questions classified by type.
    Returns JSON with 'mcq' list and 'short_answer' list.
    """
    # 1. Fallback if OpenAI client is missing
    if client is None:
        return {
            "story_text": base_text or "A simple default story.",
            "mcq": [
                {"question": "What is this story about?", "options": ["A", "B", "C", "D"], "correct_index": 0}
            ],
            "short_answer": [
                {"question": "Why did the character do that?", "category": "Inferential"}
            ]
        }

    # 2. Configure Length & Complexity based on Level
    current_level = worksheet_level or story_level or "beginner"

    length_instruction = "Write about 100-150 words."
    if "intermediate" in current_level.lower():
        length_instruction = "Write about 150-200 words."
    elif "advanced" in current_level.lower():
        length_instruction = "Write about 200-250 words."

    instructions = f"""
You are an expert English reading comprehension test generator for young students.

**Task:**
1. **Story**: Write an original, engaging story in English suitable for a {current_level} learner.
   - {length_instruction}
   - Use simple, clear sentences but an interesting plot.
   - Topic: {base_text or "A surprise adventure"}

2. **Questions**: Create exactly 12 questions.
   - **Questions 1-8 (either Factual/Detail/Inferential/Critical)**: Multiple Choice (MCQ). Provide 4 options each.
   - **Questions 9-12 (either Factual/Detail/Inferential/Critical)**: Short Answer (Open-ended). No options.

**Output Format**:
Return ONLY valid JSON with this structure:
{{
  "story_text": "Full story text here...",
  "mcq": [
    {{
      "category": "Factual",
      "question": "Where did the story take place?",
      "options": ["Option A", "Option B", "Option C", "Option D"], 
      "correct_index": 0
    }},
    
  ],
  "short_answer": [
    {{
      "category": "Inferential",
      "question": "Why did the main character feel sad?",
      "model_answer": "Because he lost his toy."
    }},
  ]
}}
"""

    try:
        resp = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": instructions},
            ],
            temperature=0.7,
            response_format={"type": "json_object"},
        )

        raw = (resp.choices[0].message.content or "").strip()
        data = json.loads(raw)

        # Validate structure
        if "mcq" not in data: data["mcq"] = []
        if "short_answer" not in data: data["short_answer"] = []

        return data

    except Exception as e:
        print(f"Error generating worksheet: {e}")
        return {
            "story_text": base_text,
            "mcq": [],
            "short_answer": [
                {"category": "Critical", "question": "Write a summary of the story."}
            ]
        }

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
            prompt=(
                f"A simple story titled '{base_text.strip()}' suitable for a {writing_level} reader. "
                "Make the story clearly match the title/theme."
                if (base_text or '').strip()
                else f"A simple story suitable for a {writing_level} reader."
            ),
            level=writing_level
        )
        if not structured_story:
            print("[GPT][worksheet][completion] ERROR: structured story generation failed (no fallback).")
            return None

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
# MAIN-based story grammar analytics (Character/Setting/Problem/Actions/Resolution)
# -------------------------------------------------------------------
def analyze_main_story_grammar(story_text: str) -> Optional[dict]:
    """Return MAIN-style story grammar analysis as JSON.

    Scores are 0/1/2 per component:
      0 = missing, 1 = partial/unclear, 2 = clear and complete.
    """
    if client is None:
        return None

    text = (story_text or "").strip()
    if not text:
        return None

    instructions = """
You are an assessor for a children's English writing platform.
Assess the student's story using MAIN-style story grammar components:
1) Character (who)
2) Setting (where/when)
3) Problem / Initiating Event (what starts the issue)
4) Actions / Attempts (what the character does)
5) Resolution (how it ends)

Return ONLY JSON with this schema:
{
  "scores": {
    "character": 0|1|2,
    "setting": 0|1|2,
    "problem": 0|1|2,
    "actions": 0|1|2,
    "resolution": 0|1|2
  },
  "evidence": {
    "character": "short quote or phrase",
    "setting": "short quote or phrase",
    "problem": "short quote or phrase",
    "actions": "short quote or phrase",
    "resolution": "short quote or phrase"
  },
  "tips": {
    "character": "one improvement suggestion",
    "setting": "one improvement suggestion",
    "problem": "one improvement suggestion",
    "actions": "one improvement suggestion",
    "resolution": "one improvement suggestion"
  },
  "summary": "1-2 sentence overall feedback"
}

Rules:
- Keep evidence short (<= 12 words each). If missing, set evidence to "".
- Be generous for beginner writers: if implied, score 1.
"""

    user_prompt = f"Student story text:\n\n{text}"

    try:
        resp = client.responses.create(
            model=DEFAULT_MODEL,
            max_output_tokens=650,
            temperature=0.2,
            instructions=instructions.strip(),
            input=[
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_prompt.strip()}],
                }
            ],
        )
        raw = (getattr(resp, "output_text", "") or "").strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?", "", raw).strip()
            raw = raw.strip("`").strip()

        # Robust JSON extraction
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if not m:
                return None
            data = json.loads(m.group(0))

        # Normalize scores
        scores = data.get("scores") or {}
        norm = {}
        for k in ["character", "setting", "problem", "actions", "resolution"]:
            v = scores.get(k, 0)
            try:
                v = int(v)
            except Exception:
                v = 0
            if v < 0:
                v = 0
            if v > 2:
                v = 2
            norm[k] = v
        data["scores"] = norm
        return data
    except Exception as e:
        print("MAIN story grammar analysis failed:", e)
        return None

# -------------------------------------------------------------------
# ADMIN: generate worksheet + assign to students
# -------------------------------------------------------------------
# --- Updated admin_generate_worksheet function in app.py ---
# --- Function: admin_generate_worksheet (Updated) ---
# In app.py - Replace your existing admin_generate_worksheet function with this:
def _reading_level_guidance(level: str) -> str:
    lvl = (level or "").strip().lower()
    if lvl in {"beginner", "a1"}:
        return "Beginner: short sentences, simple vocabulary, clear events. ~120–180 words."
    if lvl in {"intermediate", "a2"}:
        return "Intermediate: slightly longer sentences, more details, gentle inference. ~180–260 words."
    return "Advanced: richer details, varied sentences, mild challenge but kid-friendly. ~260–360 words."
def safe_json_load(raw_text: str):
    if not raw_text:
        return None

    text = raw_text.strip()

    # Remove markdown code fences if present
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```", "", text)
    text = re.sub(r"```$", "", text)

    # Try direct load first
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try extracting first JSON object using regex
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None

    return None
def generate_reading_story_and_questions_v3(
    worksheet_title: str,
    worksheet_level: str = "beginner",
    language: str = "en",
) -> dict:
    """
    Creates a NEW reading assignment payload:
      - Generates a short story with GPT
      - Comprehension questions: 사실적(factual) / 추론적(inference) / 비판적(critical)
      - Expression questions: MAIN story grammar rubric
      - Includes analytics schema placeholders so grading can compute strengths by category
    Returns dict payload ready to json.dumps().
    """
    # Fallback when OpenAI client is not available
    if client is None:
        story_text = (
            "Mina found a small key under a tree near the school garden. "
            "She wondered what it opened, so she asked her friend Joon to help. "
            "They searched the garden shed and discovered a tiny locked box. "
            "Inside, they found a note that said, 'Be curious, be kind.' "
            "Mina smiled and decided to return the key to the school office."
        )
        return {
            "type": "reading_v3",
            "title": worksheet_title or "Reading Worksheet",
            "language": language,
            "story": {"title": worksheet_title or "A Curious Key", "text": story_text},
            "comprehension": [
                {
                    "id": "C1",
                    "question_type": "factual",
                    "question_type_ko": "사실적",
                    "format": "mcq",
                    "question": "Where did Mina find the key?",
                    "options": ["Under a tree", "In her backpack", "On a bus seat", "In the classroom"],
                    "correct_index": 0,
                    "rationale": "The story says Mina found the key under a tree."
                },
                {
                    "id": "C2",
                    "question_type": "inference",
                    "question_type_ko": "추론적",
                    "format": "short",
                    "question": "Why did Mina ask Joon for help? (1–2 sentences)",
                    "model_answer": "She was curious about what the key opened and wanted help searching for the lock."
                },
                {
                    "id": "C3",
                    "question_type": "critical",
                    "question_type_ko": "비판적",
                    "format": "short",
                    "question": "Do you think Mina made a good choice by returning the key? Explain your opinion (2–3 sentences).",
                    "model_answer": "Yes, because returning the key is responsible and prevents problems. "
                                    "It also shows honesty and respect for school property."
                },
            ],
            "expression": [
                {
                    "id": "E1",
                    "grammar_label": "Character",
                    "grammar_label_ko": "인물",
                    "question": "Who is the main character? Describe them using 2 details from the story.",
                    "model_answer": "Mina is the main character. She is curious and responsible because she searches for answers and returns the key."
                },
                {
                    "id": "E2",
                    "grammar_label": "Setting",
                    "grammar_label_ko": "배경",
                    "question": "Where and when does the story happen? Use words from the story.",
                    "model_answer": "It happens near the school garden and around the garden shed during a normal school day."
                },
                {
                    "id": "E3",
                    "grammar_label": "Problem/Initiating Event",
                    "grammar_label_ko": "문제/시작 사건",
                    "question": "What event starts the story’s problem?",
                    "model_answer": "Mina finds a key and doesn’t know what it opens."
                },
                {
                    "id": "E4",
                    "grammar_label": "Actions/Attempts",
                    "grammar_label_ko": "시도/행동",
                    "question": "What actions do the characters take to solve the problem? List 2 actions.",
                    "model_answer": "She asks Joon for help and they search the garden shed to find what the key opens."
                },
                {
                    "id": "E5",
                    "grammar_label": "Resolution",
                    "grammar_label_ko": "해결",
                    "question": "How is the problem resolved at the end?",
                    "model_answer": "Mina discovers the note in the box and decides to return the key to the school office."
                },
            ],
            "analytics_schema": {
                "comprehension": {
                    "by_type": {
                        "factual": {"label_ko": "사실적", "strength_rule": "more correct factual answers", "weakness_rule": "many incorrect factual answers"},
                        "inference": {"label_ko": "추론적", "strength_rule": "answers show reasonable inference supported by text", "weakness_rule": "answers not supported by text"},
                        "critical": {"label_ko": "비판적", "strength_rule": "clear opinion + reasoning connected to story", "weakness_rule": "opinion without reasons or unrelated"},
                    },
                    "report_note": "Compute category accuracy/quality during grading and summarize strengths/weaknesses (focus on factual vs inference as requested).",
                },
                "expression": {
                    "main_story_grammar": {
                        "Character": {"label_ko": "인물"},
                        "Setting": {"label_ko": "배경"},
                        "Problem/Initiating Event": {"label_ko": "문제/시작 사건"},
                        "Actions/Attempts": {"label_ko": "시도/행동"},
                        "Resolution": {"label_ko": "해결"},
                    },
                    "report_note": "During grading, mark which MAIN elements are complete/clear and summarize the student’s strongest area.",
                },
            },
        }

    # GPT path
    lvl_hint = _reading_level_guidance(worksheet_level)

    instructions = f"""
You are generating a READING WORKSHEET JSON for kids.
Return ONLY valid JSON (no markdown, no extra text).

Requirements:
1) First create an original short story in {language}. Make it kid-friendly and coherent.
   - {lvl_hint}
2) Create Comprehension questions with EXACTLY these 3 types:
   - factual (사실적): answer is explicitly stated in the text (use MCQ)
   - inference (추론적): answer requires reasoning from the text (short answer)
   - critical (비판적): opinion/argument question connected to the story (short answer)
3) Create Expression questions based on MAIN story grammar:
   Character, Setting, Problem/Initiating Event, Actions/Attempts, Resolution
4) Include analytics_schema so the grader can compute strengths/weaknesses:
   - emphasize factual vs inference strengths/weaknesses
   - include MAIN category labels

JSON shape:
{{
  "type": "reading_v3",
  "title": "...",
  "language": "{language}",
  "story": {{"title": "...", "text": "..."}},
  "comprehension": [
    {{
      "id": "C1",
      "question_type": "factual",
      "question_type_ko": "사실적",
      "format": "mcq",
      "question": "...",
      "options": ["...","...","...","..."],
      "correct_index": 0,
      "rationale": "..."
    }},
    {{
      "id": "C2",
      "question_type": "inference",
      "question_type_ko": "추론적",
      "format": "short",
      "question": "...",
      "model_answer": "..."
    }},
    {{
      "id": "C3",
      "question_type": "critical",
      "question_type_ko": "비판적",
      "format": "short",
      "question": "...",
      "model_answer": "..."
    }}
  ],
  "expression": [
    {{
      "id": "E1",
      "grammar_label": "Character",
      "grammar_label_ko": "인물",
      "question": "...",
      "model_answer": "..."
    }},
    ...
  ],
  "analytics_schema": {{
    "comprehension": {{
      "by_type": {{
        "factual": {{"label_ko":"사실적", "strength_rule":"...", "weakness_rule":"..."}},
        "inference": {{"label_ko":"추론적", "strength_rule":"...", "weakness_rule":"..."}},
        "critical": {{"label_ko":"비판적", "strength_rule":"...", "weakness_rule":"..."}}
      }},
      "report_note": "..."
    }},
    "expression": {{
      "main_story_grammar": {{
        "Character": {{"label_ko":"인물"}},
        "Setting": {{"label_ko":"배경"}},
        "Problem/Initiating Event": {{"label_ko":"문제/시작 사건"}},
        "Actions/Attempts": {{"label_ko":"시도/행동"}},
        "Resolution": {{"label_ko":"해결"}}
      }},
      "report_note": "..."
    }}
  }}
}}
"""

    user_prompt = f"""
Worksheet title/theme: {worksheet_title or "Reading Worksheet"}
Generate the JSON now.
"""

    resp = client.responses.create(
        model=DEFAULT_MODEL,
        max_output_tokens=1600,
        temperature=0.5,
        instructions=instructions.strip(),
        input=[{"role": "user", "content": [{"type": "input_text", "text": user_prompt.strip()}]}],
    )

    raw = getattr(resp, "output_text", "") or ""
    data = safe_json_load(raw)

    if not isinstance(data, dict):
        raise ValueError("Reading v3: model did not return a JSON object.")

    # Hard-guard defaults
    data["type"] = "reading_v3"
    data.setdefault("title", worksheet_title or "Reading Worksheet")
    data.setdefault("language", language)
    data.setdefault("story", {})
    data.setdefault("comprehension", [])
    data.setdefault("expression", [])
    data.setdefault("analytics_schema", {})

    return data


def _extract_json_object(text: str) -> str:
    """Best-effort: extract the first top-level JSON object from a model response."""
    if not text:
        return ""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return ""
    return text[start : end + 1]


def gpt_json(client, prompt: str, model: str = None, default=None):
    """
    Returns dict parsed from GPT output.
    Assumes you have an OpenAI client already (e.g., OpenAI()).
    Works even if model sometimes wraps JSON in extra text.
    """
    if default is None:
        default = {}
    try:
        # Use whichever you already use elsewhere in your app.
        # If you are using Responses API:
        # resp = client.responses.create(model=model or "gpt-4.1-mini", input=prompt)
        # text = resp.output_text

        # If you are using Chat Completions:
        resp = client.chat.completions.create(
            model=model or "gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Return ONLY valid JSON. No markdown, no extra text."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )
        text = (resp.choices[0].message.content or "").strip()

        j = safe_json_load(text, None)
        if isinstance(j, dict):
            return j

        # fallback: strip wrappers
        j2 = safe_json_load(_extract_json_object(text), None)
        if isinstance(j2, dict):
            return j2
        return default
    except Exception:
        return default


def generate_reading_story_payload_gpt(
    client,
    worksheet_level: str,
    topic: str = "",
    genre: str = "",
    length: str = "short",
):
    """
    GPT generates the reading story (title + content) for the worksheet.
    """
    worksheet_level = (worksheet_level or "beginner").strip().lower()
    topic = (topic or "").strip()
    genre = (genre or "").strip()
    length = (length or "short").strip().lower()

    # Simple length control
    if length == "long":
        sentence_hint = "12–16 sentences"
    elif length == "medium":
        sentence_hint = "8–12 sentences"
    else:
        sentence_hint = "5–8 sentences"

    # Level control
    if worksheet_level == "advanced":
        vocab_hint = "slightly richer vocabulary, varied sentence structures, but still kid-friendly"
    elif worksheet_level == "intermediate":
        vocab_hint = "simple vocabulary with a few new words, kid-friendly"
    else:
        vocab_hint = "very simple vocabulary, short sentences, kid-friendly"

    prompt = f"""
Create an original children's reading passage for an English worksheet.

Constraints:
- Language: English
- Difficulty: {worksheet_level}
- Length: {sentence_hint}
- Style: {vocab_hint}
- Must be a complete story with: Character, Setting, Problem/Initiating Event, Actions/Attempts, Resolution
- Avoid violence, politics, or scary content.

Optional guidance (use if provided):
- Topic/Theme: {topic if topic else "auto"}
- Genre: {genre if genre else "auto"}

Return ONLY JSON in this exact structure:
{{
  "title": "string",
  "language": "en",
  "level": "{worksheet_level}",
  "content": "full story text"
}}
""".strip()

    data = gpt_json(client, prompt, default={})
    title = (data.get("title") or "").strip()
    content = (data.get("content") or "").strip()

    if not title:
        title = (topic.title() if topic else "Reading Story")
    if not content:
        # very small fallback story so route never breaks
        content = "Mina found a lost notebook at school. She looked for the owner and asked her teacher for help. Soon, she returned it to Jun, who smiled and said thank you."

    return {
        "title": title,
        "language": "en",
        "level": worksheet_level,
        "content": content,
    }


def generate_expression_rubric_gpt(client, base_text: str):
    """
    GPT generates MAIN-based story grammar rubric prompts + analytics keys.
    Stored into questions_json under 'expression' section.
    """
    prompt = f"""
You are an English literacy teacher. Based on the story below, produce a MAIN-style story grammar evaluation rubric.

Story:
\"\"\"{base_text}\"\"\"

Return ONLY JSON:
{{
  "story_grammar": [
    {{"key":"character","label":"Character","what_to_check":"...","student_prompt":"..."}},
    {{"key":"setting","label":"Setting","what_to_check":"...","student_prompt":"..."}},
    {{"key":"problem","label":"Problem/Initiating Event","what_to_check":"...","student_prompt":"..."}},
    {{"key":"actions","label":"Actions/Attempts","what_to_check":"...","student_prompt":"..."}},
    {{"key":"resolution","label":"Resolution","what_to_check":"...","student_prompt":"..."}}
  ],
  "analytics_keys": ["character","setting","problem","actions","resolution"]
}}
""".strip()

    data = gpt_json(client, prompt, default={})
    sg = data.get("story_grammar") if isinstance(data.get("story_grammar"), list) else []
    keys = data.get("analytics_keys") if isinstance(data.get("analytics_keys"), list) else ["character","setting","problem","actions","resolution"]

    if not sg:
        sg = [
            {"key":"character","label":"Character","what_to_check":"Who is the main character? What are they like?","student_prompt":"Write 1–2 sentences describing the main character."},
            {"key":"setting","label":"Setting","what_to_check":"Where and when does the story happen?","student_prompt":"Write 1 sentence telling where the story happens."},
            {"key":"problem","label":"Problem/Initiating Event","what_to_check":"What problem starts the story?","student_prompt":"Write 1 sentence explaining the problem."},
            {"key":"actions","label":"Actions/Attempts","what_to_check":"What does the character try to do?","student_prompt":"List 2 actions the character takes."},
            {"key":"resolution","label":"Resolution","what_to_check":"How does the story end? Is the problem solved?","student_prompt":"Write 1 sentence explaining the ending."},
        ]

    return {"story_grammar": sg, "analytics_keys": keys}


@app.route("/admin/generate-worksheet", methods=["POST"])
@login_required
def admin_generate_worksheet():
    if not current_user_is_admin():
        flash("You do not have permission to create worksheets.", "danger")
        return redirect(url_for("index"))

    db = get_db()

    assignment_type = (request.form.get("worksheet_type") or "").strip().lower()
    worksheet_title = (request.form.get("worksheet_title") or "").strip()

    # writing fields
    writing_mode = (request.form.get("writing_mode") or "planning").strip().lower()
    writing_level = (
                request.form.get("writing_level") or request.form.get("worksheet_level") or "beginner").strip().lower()

    # reading fields
    worksheet_level = (request.form.get("worksheet_level") or "beginner").strip().lower()
    reading_topic = (request.form.get("reading_topic") or "").strip()
    reading_genre = (request.form.get("reading_genre") or "").strip()
    reading_length = (request.form.get("reading_length") or "short").strip().lower()

    raw_user_ids = request.form.getlist("user_ids") or []
    user_ids = [int(uid) for uid in raw_user_ids if uid]

    draft_id = None
    story_id = None
    questions_json = None

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
        worksheet_payload = generate_worksheet_payload(
            base_text=worksheet_title,
            worksheet_type="writing",
            writing_mode=writing_mode,
            writing_level=writing_level
        )

        if not worksheet_payload:
            flash("Failed to generate writing worksheet with AI.", "danger")
            return redirect(url_for("admin_stories", tab="assign"))

        level_map = {"beginner": "A1", "intermediate": "A2", "advanced": "B1"}
        story_level = level_map.get(writing_level, "beginner")

        if writing_mode == "completion":
            generated_title = (worksheet_payload.get("story_title") or "").strip() or "Story Completion"
            slug = f"writing-completion-{uuid.uuid4().hex[:12]}"

            if not worksheet_title:
                worksheet_title = generated_title

            cur = db.execute(
                """
                INSERT INTO stories (slug, title, language, level, content, created_at)
                VALUES (?, ?, ?, ?, ?, datetime('now'))
                """,
                (slug, worksheet_title or generated_title, "en", story_level,
                 worksheet_payload.get("full_text_answer", ""))
            )
            story_id = cur.lastrowid

            cur = db.execute(
                """
                INSERT INTO finish_drafts (story_id, seed_prompt, partial_text, completion_text)
                VALUES (?, ?, ?, ?)
                """,
                (
                    story_id,
                    "System generated completion worksheet",
                    worksheet_payload.get("partial_text_prompt", ""),
                    worksheet_payload.get("full_text_answer", "")
                )
            )
            draft_id = cur.lastrowid

            payload_for_db = {
                "type": "writing_completion",
                "sections": worksheet_payload.get("sections", []),
                "checklist": worksheet_payload.get("checklist", [])
            }
            questions_json = json.dumps(payload_for_db, ensure_ascii=False)

        else:
            dummy = db.execute(
                "SELECT id FROM stories WHERE slug = ?",
                ("free-writing-template",)
            ).fetchone()

            if dummy is None:
                cur = db.execute(
                    """
                    INSERT INTO stories (slug, title, language, level, content, created_at)
                    VALUES (?, ?, ?, ?, ?, datetime('now'))
                    """,
                    ("free-writing-template", "Creative Writing Practice", "en", "beginner", "")
                )
                story_id = cur.lastrowid
            else:
                story_id = dummy["id"]

            dummy_draft = db.execute(
                "SELECT id FROM finish_drafts WHERE story_id = ? ORDER BY id DESC LIMIT 1",
                (story_id,)
            ).fetchone()

            if dummy_draft is None:
                cur = db.execute(
                    """
                    INSERT INTO finish_drafts (story_id, seed_prompt, partial_text, completion_text)
                    VALUES (?, ?, ?, ?)
                    """,
                    (story_id, "System generated prompt", "", None)
                )
                draft_id = cur.lastrowid
            else:
                draft_id = dummy_draft["id"]

            if not worksheet_title:
                worksheet_title = worksheet_payload.get("title") or "Writing Structure Practice"

            questions_json = json.dumps(worksheet_payload, ensure_ascii=False)

    # --------------------------
    # READING ASSIGNMENT SETUP (UPDATED)
    # --------------------------
    elif assignment_type == "reading":
        # 1) Use the TOPIC as the base_text input
        # generate_reading_worksheet will now handle story generation AND question generation

        # Ensure we have a topic, even if generic
        topic_prompt = reading_topic if reading_topic else "A generic story for reading practice"

        worksheet_payload = generate_reading_worksheet(
            base_text=topic_prompt,
            story_level=worksheet_level,
            worksheet_level=worksheet_level,
        ) or {}

        # 2) Extract the generated story text from the payload
        # (The updated function returns 'story_text' in the dict)
        story_text = worksheet_payload.get("story_text") or worksheet_payload.get("story", {}).get("text") or ""

        # Fallback if generation failed entirely
        if not story_text:
            story_text = f"Story generation failed for topic: {reading_topic}. Please try again."

        # 3) Determine Title (User provided or Auto-generated format)
        if not worksheet_title:
            if reading_topic:
                worksheet_title = f"Reading · {reading_topic.title()}"
            else:
                worksheet_title = "Reading Worksheet"

        slug = f"reading-auto-{uuid.uuid4().hex[:12]}"

        # 4) Insert the generated story into DB
        cur = db.execute(
            """
            INSERT INTO stories (slug, title, language, level, content, created_at)
            VALUES (?, ?, ?, ?, ?, datetime('now'))
            """,
            (slug, worksheet_title, "en", worksheet_level, story_text)
        )
        story_id = cur.lastrowid

        # 5) Ensure payload has the correct lists (mcq, short_answer)
        worksheet_payload.setdefault("mcq", [])
        worksheet_payload.setdefault("short_answer", [])

        # (Optional) Generate Expression Rubric based on the new text
        client = get_openai_client()
        expression = generate_expression_rubric_gpt(client, story_text)
        worksheet_payload["expression"] = expression

        questions_json = json.dumps(worksheet_payload, ensure_ascii=False)

    # --------------------------
    # INSERT ASSIGNMENTS (Common Path)
    # --------------------------
    created = 0
    for uid in user_ids:
        assignee_id = int(uid)
        class_id = request.form.get("class_id")
        db.execute(
            """
            INSERT INTO assignments
            (story_id, draft_id, assignee_id, assignment_type,
             questions_json, assigned_by, assignment_title, status, class_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'assigned',?)
            """,
            (
                story_id,
                draft_id,
                assignee_id,
                assignment_type,
                questions_json,
                session.get("user_id"),
                worksheet_title,
                class_id
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
    if submission["questions_json"]:
        try:
            payload = json.loads(submission["questions_json"])

            # Detect Reading Assignment Types
            if submission["assignment_type"] == 'reading':
                mcq_list = []
                fib_list = []
                sa_list = []

                # Strategy A: Check for V3 Structure (comprehension list)
                if payload.get("type") == "reading_v3" or "comprehension" in payload:
                    comp_list = payload.get("comprehension") or []
                    # Sort V3 questions into buckets
                    for q in comp_list:
                        fmt = q.get("format", "").lower()
                        if fmt == "mcq":
                            mcq_list.append(q)
                        elif fmt == "fill_in_blank":
                            fib_list.append(q)
                        else:
                            sa_list.append(q)
                    # Add expression questions to Short Answer
                    expr_list = payload.get("expression") or []
                    sa_list.extend(expr_list)

                # Strategy B: Check for Direct Lists (Standard Reading Generator)
                elif "mcq" in payload or "short_answer" in payload:
                    mcq_list = payload.get("mcq") or []
                    fib_list = payload.get("fill_in_blank") or []
                    sa_list = payload.get("short_answer") or []

                # Strategy C: Check for Legacy "sections" Structure
                elif "sections" in payload:
                    for section in payload.get("sections", []):
                        label = section.get('label', '')
                        questions = section.get('questions', [])
                        if 'MCQ' in label:
                            mcq_list.extend(questions)
                        elif 'Fill-in' in label:
                            fib_list.extend(questions)
                        elif 'Short Answer' in label:
                            sa_list.extend(questions)

                # --- POPULATE VIEW DATA ---
                assignment_details["mcq_questions"] = mcq_list

                if mcq_list:
                    assignment_details["sections"].append({"label": "MCQ Questions", "questions": mcq_list})
                if fib_list:
                    assignment_details["sections"].append({"label": "Fill-in-the-Blank Prompts", "questions": fib_list})
                if sa_list:
                    assignment_details["sections"].append({"label": "Short Answer Questions", "questions": sa_list})

            elif submission["assignment_type"] == 'writing':
                # Handles standard writing assignment structure (sections for planning/rubric)
                assignment_details["sections"].extend(payload.get("sections", []))

        except Exception as e:
            print(f"Error parsing assignment template content: {e}")

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
    if not current_user_is_admin():
        flash("Admin access only.", "warning")
        return redirect(url_for("index"))

    db = get_db()

    story = db.execute(
        "SELECT id, title, language, level, prompt, visuals FROM stories WHERE slug = ?",
        (slug,),
    ).fetchone()

    if not story:
        flash("Story not found.", "warning")
        return redirect(url_for("admin_stories"))

    # optional: class_id from form (empty => share to ALL)
    raw_class_id = (request.form.get("class_id") or "").strip()
    class_id = int(raw_class_id) if raw_class_id.isdigit() else None

    # (your existing cover generation logic stays the same...)

    # mark shared
    db.execute("UPDATE stories SET is_shared_library = 1 WHERE id = ?", (story["id"],))

    # NEW: record where it was shared
    db.execute(
        """
        INSERT OR IGNORE INTO library_shares (story_id, class_id, shared_by)
        VALUES (?, ?, ?)
        """,
        (story["id"], class_id, g.current_user["id"]),
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
def _gen_class_code(n=6):
    # Simple readable code like: A9K2QF
    alphabet = string.ascii_uppercase + string.digits
    return "".join(random.choice(alphabet) for _ in range(n))

@app.get("/classes")
@login_required
def classes_page():
    db = get_db()
    user_id = g.current_user["id"]

    my_classes = db.execute(
        """
        SELECT c.id, c.name, c.code, cm.role, cm.joined_at
        FROM classes c
        JOIN class_members cm ON cm.class_id = c.id
        WHERE cm.user_id = ?
        ORDER BY c.name COLLATE NOCASE
        """,
        (user_id,),
    ).fetchall()

    return render_template(
        "classes.html",
        my_classes=my_classes,
        role=current_user_role(),
        is_teacher=current_user_is_teacher(),
    )
@app.post("/classes/join")
@login_required
def join_class():
    # Teachers should not join via code (your UI will hide it, but backend must enforce too)
    if current_user_is_teacher():
        flash("Teachers can’t join classes with a code. Please create a class instead.", "warning")
        return redirect(url_for("classes_page"))

    db = get_db()
    user_id = g.current_user["id"]

    code = (request.form.get("class_code") or "").strip().upper().replace(" ", "")
    if not code:
        flash("Please enter a class code.", "warning")
        return redirect(url_for("classes_page"))

    klass = db.execute("SELECT * FROM classes WHERE code = ?", (code,)).fetchone()
    if not klass:
        flash("Invalid class code. Please check and try again.", "danger")
        return redirect(url_for("classes_page"))

    try:
        db.execute(
            """
            INSERT INTO class_members (class_id, user_id, role)
            VALUES (?, ?, ?)
            """,
            (klass["id"], user_id, "student"),
        )
        db.commit()
        flash(f"You joined “{klass['name']}”.", "success")
    except sqlite3.IntegrityError:
        flash("You are already in this class.", "info")

    return redirect(url_for("classes_page"))


@app.post("/admin/classes/create")
@login_required
def admin_create_class():
    # Only teachers/admin/staff can create classes
    if not current_user_is_teacher():
        flash("Only teachers can create classes.", "danger")
        return redirect(url_for("classes_page"))

    db = get_db()
    name = (request.form.get("class_name") or "").strip()
    if not name:
        flash("Please enter a class name.", "warning")
        return redirect(url_for("classes_page"))

    # Retry a few times in case code collides
    for _ in range(10):
        code = _gen_class_code(6)
        try:
            cur = db.execute(
                "INSERT INTO classes (name, code, created_by) VALUES (?, ?, ?)",
                (name, code, g.current_user["id"]),
            )
            class_id = cur.lastrowid

            # creator becomes teacher member
            db.execute(
                "INSERT INTO class_members (class_id, user_id, role) VALUES (?, ?, 'teacher')",
                (class_id, g.current_user["id"]),
            )
            db.commit()

            flash(f"Class created! Code: {code}", "success")
            return redirect(url_for("classes_page"))

        except sqlite3.IntegrityError:
            continue

    flash("Could not generate a unique class code. Try again.", "danger")
    return redirect(url_for("classes_page"))

@app.post("/assignments/<int:assignment_id>/one_idea")
@login_required
def assignment_one_idea(assignment_id: int):
    db = get_db()
    user_id = g.current_user["id"]

    assignment = db.execute(
        "SELECT * FROM assignments WHERE id=? AND assignee_id=?",
        (assignment_id, user_id),
    ).fetchone()
    if not assignment:
        return jsonify({"error": "not_found"}), 404

    story = db.execute("SELECT * FROM stories WHERE id=?", (assignment["story_id"],)).fetchone()
    draft = None
    if assignment.get("draft_id"):
        draft = db.execute("SELECT * FROM finish_drafts WHERE id=?", (assignment["draft_id"],)).fetchone()

    title = (assignment.get("assignment_title") or "")
    ins = (assignment.get("instructions") or "")  # might be missing in your table; safe
    blob = (title + " " + ins).lower()

    part = "middle"
    if "beginning" in blob:
        part = "beginning"
    elif "middle" in blob:
        part = "middle"
    elif "ending" in blob or " end " in blob or blob.endswith(" end"):
        part = "end"

    partial = ""
    if draft:
        partial = (draft.get("partial_text") or "").strip()

    # Fallback if OpenAI is not configured
    if client is None:
        fallback = {
            "beginning": "One day, Tim decided to start his adventure by saying hello to someone nearby.",
            "middle": "Then something surprising happened, and Tim had to try a new idea with his friends.",
            "end": "In the end, Tim solved the problem and felt proud of his new friends.",
        }[part]
        return jsonify({"idea": fallback})

    # GPT one-sentence prompt
    sys = "You write for 7–8 year olds. Output exactly ONE sentence. No quotes. No extra text."
    user = f"""
Story so far:
\"\"\"{partial}\"\"\"

Task:
Give ONE sentence of idea and suggestion for helping young student to complete the {part} of the story.
Make it simple, clear, and connected to the story.
"""

    try:
        resp = client.responses.create(
            model=DEFAULT_MODEL,
            max_output_tokens=60,
            temperature=0.6,
            input=[
                {"role": "system", "content": sys},
                {"role": "user", "content": user.strip()},
            ],
        )
        idea = (getattr(resp, "output_text", "") or "").strip()
        # Safety: keep only first line / first sentence-ish
        idea = idea.splitlines()[0].strip()
        return jsonify({"idea": idea or "Then Tim had an idea and tried it with a smile."})
    except Exception as e:
        log_input("one_idea_error", {"error": str(e), "assignment_id": assignment_id})
        return jsonify({"idea": "Then Tim had an idea and tried it with a smile."})

if __name__ == "__main__":
    with app.app_context():
        init_db()
        upgrade_bookmarks_table()
    app.run(debug=True)