import os
import io
import re
import csv
import json
import sqlite3
from datetime import datetime
from functools import wraps
from urllib.parse import urlparse, urljoin
import random
from slugify import slugify
from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, send_file, g, session
)
from werkzeug.security import generate_password_hash, check_password_hash

# --- OpenAI SDK (env-based; do NOT hardcode secrets) ---
from openai import OpenAI
import requests
import math
import html
import nltk
from nltk.corpus import wordnet as wn

DEFAULT_MODEL = os.getenv("LINGOPLAY_MODEL", "gpt-4.1-mini")

# ------------------------------------------------------------------------------------
# App setup
# ------------------------------------------------------------------------------------
app = Flask(__name__, instance_relative_config=True)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "dev-secret")

# Ensure instance folder exists (DB goes here)
os.makedirs(app.instance_path, exist_ok=True)
DB_PATH = os.path.join(app.instance_path, "lingoplay.db")

# NOTE: Keeping your existing client initialization exactly as provided

# ------------------------------------------------------------------------------------
# Helpers: DB, JSON, datetime
# ------------------------------------------------------------------------------------
def dict_factory(cursor, row):
    """Return rows as dicts instead of tuples."""
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d

@app.template_filter('dt')
def format_dt(value, fmt="%Y-%m-%d %H:%M"):
    """
    Safely format SQLite DATETIME strings or datetime objects.
    """
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

@app.template_filter("load_json")
def load_json_filter_template(s):
    try:
        return json.loads(s) if s else {}
    except Exception:
        return {}
def get_learner_profile():
    """
    Returns a normalized, safe dict about the current learner.
    Keys: age (int|None), is_english_native (bool|None), gender ('female'|'male'|'nonbinary'|'prefer_not'|None)
    """
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
        "username": u.get("username") or "guest"
    }
def _reading_prefs_from_profile(profile: dict, explicit_level: str | None):
    """
    Decide level + difficulty notes from age + native flag.
    Returns (level, bullets[str]) where bullets is appended to prompt.
    """
    level = (explicit_level or "phonics").strip().lower()

    age = profile.get("age")
    is_native = profile.get("is_english_native")

    # Infer level only if user didn't override with 'custom'
    if explicit_level not in ("custom",):
        if age is not None:
            if age <= 7:
                level = "phonics"
            elif 8 <= age <= 10:
                level = "early-reader"
            else:
                level = level  # keep user's choice

    notes = []
    # Simplicity knobs for non-native readers
    if is_native is False:
        notes += [
            "Prefer high-frequency, decodable words; avoid idioms and slang.",
            "Keep sentences short (≤10–12 words) and concrete.",
            "Rephrase rare words with simpler synonyms."
        ]
    # Age-tailored structure hints
    if age is not None and age <= 7:
        notes += [
            "Use clear repetition and predictable patterns.",
            "One action per sentence; present tense preferred."
        ]
    elif age is not None and 8 <= age <= 10:
        notes += [
            "Keep sentences simple (8–12 words) with occasional compound sentences.",
            "Use concrete details and gentle cause-effect."
        ]

    return level, notes

def get_db():
    """Get a cached connection for the current request context."""
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
    """Create tables if they don't exist (incl. users)."""
    db = get_db()
    db.executescript(
        """
        PRAGMA foreign_keys = ON;

        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT DEFAULT (DATETIME('now'))
        );

        CREATE TABLE IF NOT EXISTS stories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            slug TEXT NOT NULL UNIQUE,
            prompt TEXT NOT NULL,
            language TEXT DEFAULT 'en',
            level TEXT DEFAULT 'phonics',
            content TEXT NOT NULL,
            visuals TEXT,
            created_at TEXT DEFAULT (DATETIME('now')),
            author_name TEXT DEFAULT 'guest'
        );

        CREATE TABLE IF NOT EXISTS quiz_questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            story_id INTEGER NOT NULL,
            question TEXT NOT NULL,
            choices_json TEXT,
            correct_index INTEGER,
            /* NEW: type + free-text fields */
            qtype TEXT DEFAULT 'mcq',
            answer_text TEXT,
            rubric TEXT,
            FOREIGN KEY (story_id) REFERENCES stories(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS quiz_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            story_id INTEGER NOT NULL,
            taker_name TEXT DEFAULT 'guest',
            score INTEGER DEFAULT 0,
            total INTEGER DEFAULT 0,
            detail_json TEXT,
            created_at TEXT DEFAULT (DATETIME('now')),
            FOREIGN KEY (story_id) REFERENCES stories(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS finish_drafts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            seed_prompt TEXT NOT NULL,
            partial_text TEXT NOT NULL,
            learner_name TEXT DEFAULT 'guest',
            completion_text TEXT,
            language TEXT DEFAULT 'en',
            created_at TEXT DEFAULT (DATETIME('now'))
        );

        CREATE TABLE IF NOT EXISTS input_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            action TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            created_at TEXT DEFAULT (DATETIME('now'))
        );

        CREATE TABLE IF NOT EXISTS vocab_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            story_id INTEGER NOT NULL,
            word TEXT NOT NULL,
            definition TEXT,
            example TEXT,
            picture_url TEXT,
            created_at TEXT DEFAULT (DATETIME('now')),
            definition_ko TEXT,
            example_ko TEXT,
            FOREIGN KEY (story_id) REFERENCES stories(id) ON DELETE CASCADE
        );
        """
    )
    db.executescript(
        """
        PRAGMA foreign_keys = ON;

        CREATE TABLE IF NOT EXISTS finish_comments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            draft_id INTEGER NOT NULL,
            author_name TEXT NOT NULL,
            body TEXT NOT NULL,
            created_at TEXT DEFAULT (DATETIME('now')),
            FOREIGN KEY (draft_id) REFERENCES finish_drafts(id) ON DELETE CASCADE
        );
        """
    )
    db.executescript(
        """
        PRAGMA foreign_keys = ON;

        CREATE TABLE IF NOT EXISTS classrooms (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            code TEXT NOT NULL UNIQUE,
            owner_id INTEGER NOT NULL,
            created_at TEXT DEFAULT (DATETIME('now')),
            FOREIGN KEY (owner_id) REFERENCES users(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS classroom_members (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            classroom_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            role TEXT DEFAULT 'student', -- 'teacher' or 'student'
            joined_at TEXT DEFAULT (DATETIME('now')),
            UNIQUE (classroom_id, user_id),
            FOREIGN KEY (classroom_id) REFERENCES classrooms(id) ON DELETE CASCADE,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS classroom_assignments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            classroom_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            description TEXT,
            story_id INTEGER,  -- seed story created by teacher (optional)
            due_at TEXT,
            created_at TEXT DEFAULT (DATETIME('now')),
            FOREIGN KEY (classroom_id) REFERENCES classrooms(id) ON DELETE CASCADE,
            FOREIGN KEY (story_id) REFERENCES stories(id) ON DELETE SET NULL
        );

        CREATE TABLE IF NOT EXISTS assignment_submissions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            assignment_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            draft_id INTEGER NOT NULL,  -- link to finish_drafts.id
            status TEXT DEFAULT 'in_progress', -- 'in_progress' / 'submitted'
            created_at TEXT DEFAULT (DATETIME('now')),
            updated_at TEXT DEFAULT (DATETIME('now')),
            UNIQUE (assignment_id, user_id),
            FOREIGN KEY (assignment_id) REFERENCES classroom_assignments(id) ON DELETE CASCADE,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY (draft_id) REFERENCES finish_drafts(id) ON DELETE CASCADE
        );
        """
    )

    # Idempotent ALTERs for older DBs
    try:
        db.execute("ALTER TABLE users ADD COLUMN is_admin INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass

    try:
        db.execute("ALTER TABLE quiz_questions ADD COLUMN qtype TEXT DEFAULT 'mcq'")
    except sqlite3.OperationalError:
        pass
    try:
        db.execute("ALTER TABLE quiz_questions ADD COLUMN answer_text TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        db.execute("ALTER TABLE quiz_questions ADD COLUMN rubric TEXT")
    except sqlite3.OperationalError:
        pass

    db.execute("UPDATE users SET is_admin = 1 WHERE lower(username) = lower(?)", ("testtest",))
    db.commit()

# ------- Batch resolver (call this from story_new) -------
from nltk.stem import WordNetLemmatizer

_lemmatizer = WordNetLemmatizer()
def _insert_quiz_question(db, story_id: int, item: dict):
    """
    Normalizes and inserts a quiz question row.
    Ensures choices_json is always a JSON string (never NULL).
    """
    t = (item.get("type") or "mcq").strip().lower()
    q = (item.get("question") or "").strip()
    if not q:
        return  # skip empties

    if t == "mcq":
        choices = [str(c) for c in (item.get("choices") or [])][:4]
        if len(choices) < 4:
            choices += ["—"] * (4 - len(choices))
        try:
            ci = int(item.get("correct_index", 0))
            if ci not in (0, 1, 2, 3):
                ci = 0
        except Exception:
            ci = 0

        db.execute(
            """INSERT INTO quiz_questions
               (story_id, qtype, question, choices_json, correct_index, answer_text, rubric)
               VALUES (?, 'mcq', ?, ?, ?, NULL, NULL)""",
            (story_id, q, json.dumps(choices, ensure_ascii=False), ci)
        )
    elif t == "short":
        ans = (item.get("answer") or "").strip()
        rub = (item.get("rubric") or "").strip() or "Short, story-consistent answer."
        db.execute(
            """INSERT INTO quiz_questions
               (story_id, qtype, question, choices_json, correct_index, answer_text, rubric)
               VALUES (?, 'short', ?, '[]', -1, ?, ?)""",
            (story_id, q, ans, rub)
        )
    else:  # long (or anything else)
        rub = (item.get("rubric") or "").strip() or "Clear structure; uses story details; coherent."
        db.execute(
            """INSERT INTO quiz_questions
               (story_id, qtype, question, choices_json, correct_index, answer_text, rubric)
               VALUES (?, 'long', ?, '[]', -1, NULL, ?)""",
            (story_id, q, rub)
        )

def _lemmatize_en(token: str) -> str:
    """
    Light lemmatization for English to reduce lookup misses.
    Try noun -> verb -> adj -> adv order.
    """
    w = token
    for pos in ("n", "v", "a", "r"):
        candidate = nltk.corpus.wordnet.morphy(w, pos=pos)
        if candidate:
            w = candidate
            break
    # WordNetLemmatizer as last pass
    w = _lemmatizer.lemmatize(w)
    return w
# ------- Batch resolver (call this from story_new) -------
from nltk.stem import WordNetLemmatizer

_lemmatizer = WordNetLemmatizer()

def _lemmatize_en(token: str) -> str:
    """
    Light lemmatization for English to reduce lookup misses.
    Try noun -> verb -> adj -> adv order.
    """
    w = token
    for pos in ("n", "v", "a", "r"):
        candidate = nltk.corpus.wordnet.morphy(w, pos=pos)
        if candidate:
            w = candidate
            break
    # WordNetLemmatizer as last pass
    w = _lemmatizer.lemmatize(w)
    return w

def bulk_resolve_kid_definitions_bilingual(words: list[str]) -> list[dict]:
    """
    Clean tokens, lemmatize English where helpful, resolve EN+KO kid-friendly defs.
    Returns a list like:
    [{word, definition_en, example_en, definition_ko, example_ko}, ...]
    """
    if not words:
        return []

    clean: list[str] = []
    seen = set()

    for raw in words:
        w = (raw or "").strip().lower()
        if not w:
            continue

        # Accept simple EN or KO tokens only
        if re.match(r"^[a-z']+$", w):
            # strip surrounding apostrophes (e.g., children's -> children’s already cleaned upstream)
            w = w.strip("'")
            # basic length filter
            if not (2 <= len(w) <= 24):
                continue
            # lemmatize to reduce misses (cats -> cat, running -> run)
            w = _lemmatize_en(w)
        elif re.match(r"^[가-힣]+$", w):
            # short heuristic length filter for KO
            if not (1 <= len(w) <= 8):
                continue
        else:
            # skip mixed/complex tokens
            continue

        if w and w not in seen:
            seen.add(w)
            clean.append(w)

    out: list[dict] = []
    for w in clean:
        try:
            item = resolve_kid_definition_bilingual(w)
        except Exception:
            # absolute safety net: never crash vocab build
            de, ee = kid_def_fallback(w, "en")
            dk, ek = kid_def_fallback(w, "ko")
            item = {
                "word": w,
                "definition_en": de, "example_en": ee,
                "definition_ko": dk, "example_ko": ek,
            }
        out.append(item)

    return out



# Create DB on first run
with app.app_context():

    init_db()

# ------------------------------------------------------------------------------------
# Auth helpers + current_user injection
# ------------------------------------------------------------------------------------
def get_user_by_identifier(identifier: str):
    """Allow login with either email or username (case-insensitive)."""
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

# ------------------------------------------------------------------------------------
# NEW: login_required decorator
# ------------------------------------------------------------------------------------
def is_safe_url(target: str) -> bool:
    try:
        ref = urlparse(request.host_url)
        test = urlparse(urljoin(request.host_url, target))
        return test.scheme in ("http", "https") and ref.netloc == test.netloc
    except Exception:
        return False
def _gpt_bilingual_kid_defs(words: list[str]) -> dict[str, dict]:
    """
    Return {word: {definition_en, example_en, definition_ko, example_ko}} for each word,
    using strict JSON. Examples must be natural usages (no meta talk like "I can use...").
    """
    if client is None or not words:
        return {}

    schema = {
        "name": "kid_bilingual_defs",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "word": {"type": "string", "minLength": 1},
                            "definition_en": {"type": "string", "minLength": 1},
                            "example_en": {"type": "string", "minLength": 1},
                            "definition_ko": {"type": "string", "minLength": 1},
                            "example_ko": {"type": "string", "minLength": 1}
                        },
                        "required": ["word","definition_en","example_en","definition_ko","example_ko"],
                        "additionalProperties": False
                    },
                    "minItems": 1
                }
            },
            "required": ["items"],
            "additionalProperties": False
        }
    }

    instr = (
        "You are a children's dictionary. For each WORD, produce:\n"
        "- definition_en: ≤16 words, simple, concrete, natural English.\n"
        "- example_en: ≤12 words, NATURAL sentence using the word in context (no meta talk).\n"
        "- definition_ko: ≤16 words, natural Korean for kids (use 쉬운 말, avoid loanwords if possible).\n"
        "- example_ko: ≤12 words, NATURAL sentence using the word in context (no meta talk).\n"
        "Do not define with circular wording; avoid dictionary jargon; no quotes around the word.\n"
        "Examples must read like normal sentences (e.g., “The lemonade feels cool.” / “레몬에이드는 시원했어.”)."
    )

    out: dict[str, dict] = {}
    BATCH = 20
    chunks = [words[i:i+BATCH] for i in range(0, len(words), BATCH)]
    for chunk in chunks:
        user_msg = "WORDS:\n" + "\n".join(f"- {w}" for w in chunk)
        try:
            resp = client.responses.create(
                model=os.getenv("LINGOPLAY_MODEL","gpt-4o-mini"),
                input=[{"role":"system","content": instr},
                       {"role":"user","content": user_msg}],
                temperature=0.2,
                max_output_tokens=1500,
                response_format={"type": "json_schema", "json_schema": schema},
            )
            raw = (getattr(resp, "output_text", "") or "").strip()
            data = json.loads(raw) if raw else {}
            items = data.get("items") or []
            for it in items:
                w = (it.get("word") or "").strip().lower()
                if not w:
                    continue
                out[w] = {
                    "word": w,
                    "definition_en": (it.get("definition_en") or "").strip(),
                    "example_en": (it.get("example_en") or "").strip(),
                    "definition_ko": (it.get("definition_ko") or "").strip(),
                    "example_ko": (it.get("example_ko") or "").strip(),
                }
        except Exception:
            # fall through; caller will provide fallback
            pass
    return out

def login_required(view_func):
    @wraps(view_func)
    def _wrapped(*args, **kwargs):
        if not (g.get("current_user") and g.current_user.get("id")):
            next_url = request.full_path if request.query_string else request.path
            return redirect(url_for("login", next=next_url))
        return view_func(*args, **kwargs)
    return _wrapped

# ------------------------------------------------------------------------------------
# Generators & utilities (unchanged logic)
# ------------------------------------------------------------------------------------
def naive_story_from_prompt(prompt: str, language: str = "en") -> str:
    if language == "ko":
        return (f"{prompt} 단어를 활용한 이야기를 길게 써 볼게요. 주인공은 쉬운 말과 짧은 문장으로 생각을 나누고, "
                "친구들과 소리를 연습하며 장면이 천천히 이어집니다. 날씨가 바뀌고, 작은 실수가 나오고, "
                "다시 시도하며 표현이 점점 또렷해집니다. 마지막에는 스스로 읽고 말하며 오늘 배운 소리를 "
                "일상에서 써 보기로 다짐합니다.")
    elif language == "en-ko":
        return ("We will write a longer story using your words. The hero practices new sounds, makes small mistakes, "
                "and tries again with patient friends. Scenes change slowly and clearly, so early readers can follow. "
                "At the end, the hero uses today’s sounds in real life and smiles. "
                "우리는 당신의 단어로 좀 더 긴 이야기를 써요. 주인공은 소리를 연습하고, 작은 실수를 하지만, "
                "다시 시도하며 차분히 나아가요. 마지막에 오늘 배운 소리를 생활에서 써 보고 미소를 짓습니다.")
    else:
        return ("We’ll write a longer, simple story using your words. The hero practices sounds with friends, "
                "tries again after small mistakes, and speaks more clearly with each step. The day changes, "
                "little goals appear, and confidence grows. In the end, the hero uses today’s sounds in real life.")
def llm_story_from_prompt(prompt: str, language: str, level: str, author: str, learner_profile: dict | None = None) -> str:
    if client is None:
        return naive_story_from_prompt(prompt, language)

    profile = learner_profile or {}
    level, pref_notes = _reading_prefs_from_profile(profile, level)

    # Gender is NOT used to stereotype; we only allow neutral/inclusive pronouns guidance.
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
        "custom": "Neutral elementary reading level unless the prompt implies otherwise."
    }.get(level, "Use simple sentences.")

    lang_note = (
        "Write entirely in Korean." if language == "ko"
        else "Write each sentence in English, then the Korean translation on the next line." if language == "en-ko"
        else "Write entirely in English."
    )

    # Extra scaffolding for non-native readers with EN output
    extra_scaffold = ""
    if language == "en" and profile.get("is_english_native") is False:
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
        "Length: 180–260 words (or Korean equivalent).\n"
        "Keep sentences short and decodable; use gentle repetition; warm tone; hopeful ending.\n"
        "Personalization notes:\n- " + "\n- ".join(pref_notes)
    )

    resp = client.responses.create(
        model=DEFAULT_MODEL,
        input=[{"role": "system", "content": system},{"role": "user", "content": user}],
        temperature=0.4,
        max_output_tokens=800,
    )
    text = getattr(resp, "output_text", "") or ""
    return text.strip() or naive_story_from_prompt(prompt, language)


def generate_quiz_via_gpt(story_text: str, language: str = "en", level: str = "phonics") -> list[dict]:
    TARGET_QS = 10
    if client is None:
        base = [
            {"question": "What did the hero practice?", "choices": ["Math", "New sounds", "Running", "Cooking"], "correct_index": 1},
            {"question": "Who supported the hero?", "choices": ["Friends", "A doctor", "A pilot", "A chef"], "correct_index": 0},
            {"question": "Where did reading happen?", "choices": ["At a park", "In a cave", "On a boat", "In a plane"], "correct_index": 0},
            {"question": "What is the main idea?", "choices": ["Cooking", "Reading together", "Racing", "Sleeping"], "correct_index": 1},
            {"question": "What happens near the start?", "choices": ["They sleep", "They share words", "They cook", "They race"], "correct_index": 1},
        ]
        while len(base) < TARGET_QS:
            base.append({"question":"Which choice matches the story?", "choices":["Option A","Option B","Option C","Option D"], "correct_index":0})
        return base[:TARGET_QS]
    lang_note = "Write questions in English." if language != "ko" else "Write questions in Korean."
    level_note = {
        "phonics": "Use very simple, concrete wording; early-reader friendly; decodable vocabulary.",
        "early-reader": "Simple sentences (8–12 words) and concrete vocabulary.",
        "custom": "General elementary level."
    }.get(level, "Simple sentences.")
    system = (
        "You create multiple-choice questions for short children's stories.\n"
        "Return STRICT JSON ONLY (no markdown, no commentary):\n"
        "{ \"questions\": [ {\"question\": str, \"choices\": [str, str, str, str], \"correct_index\": int} ] }"
    )
    user = (
        f"{lang_note}\n{level_note}\n"
        "Make EXACTLY 10 questions aimed at early learners:\n"
        "- 5 vocabulary-in-context; - 3 detail; - 1 sequence; - 1 main idea.\n"
        "Rules: 4 choices, one correct; short, unambiguous; reference only the story.\n"
        "STORY:\n" + story_text
    )
    resp = client.responses.create(
        model=DEFAULT_MODEL,
        input=[{"role": "system", "content": system},
               {"role": "user", "content": user}],
        temperature=0.35,
        max_output_tokens=1200,
    )
    raw = (getattr(resp, "output_text", "") or "").strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I | re.M)
    try:
        data = json.loads(raw)
    except Exception:
        data = {"questions": []}
    out: list[dict] = []
    for q in (data.get("questions") or []):
        question = str(q.get("question", "")).strip()
        choices = [str(c) for c in (q.get("choices") or [])][:4]
        while len(choices) < 4:
            choices.append("—")
        try:
            ci = int(q.get("correct_index"))
            if not (0 <= ci <= 3):
                ci = 0
        except Exception:
            ci = 0
        if question:
            out.append({"question": question, "choices": choices[:4], "correct_index": ci})
    while len(out) < 10:
        out.append({"question": "Which choice best matches the story?","choices": ["A","B","C","D"],"correct_index": 0})
    return out[:10]

def simple_questions_from_story(text: str):
    return [
        ("What happens near the start?", ["They sleep", "They share words", "They cook", "They race"], 1),
        ("What does the hero practice?", ["Sports", "Math", "New sounds", "Nothing"], 2),
        ("How does the story end?", ["Sad ending", "No ending", "Hopeful note", "A storm"], 2),
    ]

def parse_vocab_from_prompt(prompt: str) -> list[str]:
    raw = re.split(r"[,\n;]+", prompt)
    words = []
    for tok in raw:
        t = re.sub(r"[^A-Za-z가-힣0-9' _-]+", "", tok).strip()
        if not t:
            continue
        if len(t) <= 1 and not re.search(r"[A-Za-z]{2,}", t):
            continue
        words.append(t.lower())
    seen, out = set(), []
    for w in words:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out[:12]

def kid_def_fallback(word: str, language: str) -> tuple[str, str]:
    gl = {
        "sun": ("a big hot star we see in the sky", "The sun makes the day bright."),
        "shop": ("a place where people buy things", "We go to the shop to get milk."),
        "tree": ("a tall plant with a trunk and leaves", "The bird sits in the tree.")
    }
    if language == "ko":
        d = "이야기에서 쓰인 쉬운 단어예요."
        e = f"나는 이야기에서 '{word}' 단어를 읽을 수 있어요."
        return d, e
    if word in gl:
        d, e = gl[word]
    else:
        d = "a simple word used in this story"
        e = f"I can read the word '{word}' in the story."
    return d, e
def _llm_json_kid_defs(words: list[str], language: str = "en") -> list[dict]:
    """
    Calls OpenAI Responses with strict JSON Schema to get
    [{word, definition, example}...] in the requested language.
    Returns a best-effort validated list; never raises.
    """
    if client is None or not words:
        # local fallback path
        out = []
        for w in words:
            d, e = kid_def_fallback(w, language)
            out.append({"word": w, "definition": d, "example": e})
        return out

    # Keep batches small (models tend to be more consistent)
    BATCH = 20
    chunks = [words[i:i+BATCH] for i in range(0, len(words), BATCH)]
    results: list[dict] = []

    # JSON schema to strictly enforce shape & non-empty strings
    schema = {
        "name": "kid_defs",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "word": {"type": "string", "minLength": 1},
                            "definition": {"type": "string", "minLength": 1},
                            "example": {"type": "string", "minLength": 1}
                        },
                        "required": ["word", "definition", "example"],
                        "additionalProperties": False
                    },
                    "minItems": 1
                }
            },
            "required": ["items"],
            "additionalProperties": False
        }
    }

    lang_note = "Write in English only." if language != "ko" else "Write in Korean only."
    instruction = (
        f"{lang_note} For each WORD, return a one-sentence kid-friendly DEFINITION (≤16 words) "
        f"and a short EXAMPLE sentence (≤12 words). Keep it concrete and early-reader friendly."
    )

    for chunk in chunks:
        # Build a plain list; keep words in lower-case to match your storage
        chunk = [w.lower() for w in chunk if w]
        if not chunk:
            continue

        # Compose the prompt
        user_msg = instruction + "\n\nWORDS:\n" + "\n".join(f"- {w}" for w in chunk)

        try:
            resp = client.responses.create(
                model=DEFAULT_MODEL,
                input=[{"role": "user", "content": user_msg}],
                temperature=0.1,
                max_output_tokens=800,
                response_format={"type": "json_schema", "json_schema": schema},
            )
            raw = (getattr(resp, "output_text", "") or "").strip()
            data = {}
            try:
                data = json.loads(raw)
            except Exception:
                # If somehow not JSON, force fallback for this batch
                data = {}

            items = data.get("items") if isinstance(data, dict) else None
            if not isinstance(items, list) or not items:
                # strict fallback: fill this whole chunk locally
                for w in chunk:
                    d, e = kid_def_fallback(w, language)
                    results.append({"word": w, "definition": d, "example": e})
                continue

            # Validate each item and fill any gap with fallback
            seen = set()
            for it in items:
                w = (it.get("word") or "").strip().lower()
                d = (it.get("definition") or "").strip()
                e = (it.get("example") or "").strip()
                if not w or w not in chunk or w in seen or not d or not e:
                    continue
                seen.add(w)
                results.append({"word": w, "definition": d, "example": e})

            # Any missing words in this chunk -> fallback individually
            for w in chunk:
                if w not in {x["word"] for x in results}:
                    d, e = kid_def_fallback(w, language)
                    results.append({"word": w, "definition": d, "example": e})

        except Exception:
            # Total failure for this batch -> fallback for the whole chunk
            for w in chunk:
                d, e = kid_def_fallback(w, language)
                results.append({"word": w, "definition": d, "example": e})

    return results


def generate_kid_definitions(words: list[str], language: str = "en") -> list[dict]:
    """
    Public helper used across the app. Wraps the strict JSON call + local fallback.
    Output shape: [{"word", "definition", "example"}...].
    """
    # Clean + dedupe in a stable order
    clean = []
    seen = set()
    for w in (words or []):
        lw = (w or "").strip().lower()
        if not lw or lw in seen:
            continue
        # Keep simple tokens: english letters or korean blocks
        if not (re.match(r"^[a-z']+$", lw) or re.match(r"^[가-힣]+$", lw)):
            continue
        seen.add(lw)
        clean.append(lw)

    if not clean:
        return []

    return _llm_json_kid_defs(clean, language=language)
# replace your current make_partial_from_story with this:
def make_partial_from_story(full_text: str) -> str:
    sentences = re.split(r'(?<=[.!?。！？])\s+', full_text.strip())
    if len(sentences) < 4:
        keep = sentences[:max(1, int(len(sentences)*0.75))]
    else:
        keep = sentences[:max(3, int(len(sentences)*0.8))]
    partial = " ".join(keep).strip()
    if partial and not partial.endswith(('.', '!', '?', '。', '！', '？')):
        partial += "."
    # exact style you asked for:
    partial += "\n\n"
    return partial

def log_input(action: str, payload: dict):
    db = get_db()
    db.execute(
        "INSERT INTO input_logs (action, payload_json) VALUES (?, ?)",
        (action, json.dumps(payload)),
    )
    db.commit()
def generate_content_questions_via_gpt(story_text, language="en", level="phonics", n=5):
    """
    Use GPT to generate creative, mixed-type comprehension questions from a story.
    About half of them will be story-based, and half vocabulary- or meaning-based.
    """
    prompt = f"""
You are an expert children's reading educator. Create {n} unique comprehension questions
for the following story written for {level}-level learners.

Story:
\"\"\"{story_text}\"\"\"

Please mix question types creatively:
- ~50% should focus on story events, emotions, reasoning, or predictions.
- ~50% should relate to the meanings or usage of key words from the story.
- Include a mix of multiple-choice, "what if", and sequencing questions.
- Each question must have 4 answer options and indicate the correct choice.

Return the output in strict JSON format:
[
  {{
    "question": "...",
    "choices": ["A", "B", "C", "D"],
    "correct_index": 0
  }},
  ...
]
    """

    r = client.responses.create(
        model="gpt-4o-mini",
        input=prompt,
        temperature=0.9,  # encourage creative variation
    )

    try:
        data = json.loads(r.output_text.strip())
        return data if isinstance(data, list) else []
    except Exception:
        return []
def generate_definition_questions_from_vocab(db, story_id: int, language: str = "en", count: int = 5) -> list[dict]:
    """
    Builds 'what does WORD mean?' questions from precomputed vocab_items for this story.
    Uses EN defs unless language == 'ko' -> uses KO defs.
    """
    rows = db.execute(
        "SELECT word, definition, definition_ko FROM vocab_items WHERE story_id = ? ORDER BY id ASC",
        (story_id,)
    ).fetchall()

    if not rows:
        return []

    # Choose definition field
    use_ko = (language == "ko")
    items = []
    for r in rows:
        w = (r.get("word") or "").strip().lower()
        d_en = (r.get("definition") or "").strip()
        d_ko = (r.get("definition_ko") or "").strip()
        # fallback if missing
        if use_ko and not d_ko:
            d_ko, _ = kid_def_fallback(w, "ko")
        if (not use_ko) and not d_en:
            d_en, _ = kid_def_fallback(w, "en")
        items.append({"word": w, "def": (d_ko if use_ko else d_en)})

    # filter out empties
    items = [it for it in items if it["word"] and it["def"]]
    if not items:
        return []

    # Build distractor pool
    defs_pool = [it["def"] for it in items]

    qs = []
    for idx, it in enumerate(items):
        if len(qs) >= count:
            break
        correct = it["def"]

        # pick 3 distractors from other defs
        dists = []
        for d in defs_pool:
            if d != correct and d not in dists:
                dists.append(d)
            if len(dists) == 3:
                break
        # If pool too small, backfill generic distractors
        generic = [
            "A kind of animal.",
            "A place to buy things.",
            "A feeling you have.",
            "A tool people use."
        ]
        while len(dists) < 3:
            dists.append(generic[len(dists) % len(generic)])

        # Compose choices; put correct in a stable position (index 1)
        choices = [dists[0], correct, dists[1], dists[2]]
        if use_ko:
            qtext = f"‘{it['word']}’의 뜻으로 가장 알맞은 것은?"
        else:
            qtext = f"What does “{it['word']}” mean?"

        qs.append({
            "question": qtext,
            "choices": choices,
            "correct_index": 1
        })

    # pad if fewer than requested
    while len(qs) < count:
        qs.append({
            "question": "Choose the best meaning.",
            "choices": ["Thing", "Place", "Feeling", "Animal"],
            "correct_index": 0
        })
    return qs[:count]
def _resp_to_text(resp) -> str:
    """
    Extract best-effort text from OpenAI SDK Responses objects across versions.
    """
    # Newer SDKs often expose .output_text
    if hasattr(resp, "output_text") and resp.output_text:
        return resp.output_text

    # Fallback: walk .output -> .content -> .text (object-y variants)
    try:
        parts = []
        for out in getattr(resp, "output", []) or []:
            content = getattr(out, "content", []) or []
            for c in content:
                # some SDKs use dicts, others small objs
                if isinstance(c, dict):
                    t = c.get("text")
                else:
                    t = getattr(c, "text", None)
                if t:
                    parts.append(t)
        if parts:
            return "".join(parts)
    except Exception:
        pass

    # Very old fallback
    return str(resp or "").strip()


def _strip_md_fences(s: str) -> str:
    """
    Remove single leading/trailing triple-backtick fences (``` or ```json).
    Keeps everything else untouched.
    """
    if not s:
        return s
    return re.sub(r'^```(?:json)?\s*|\s*```$', '', s.strip(), flags=re.I | re.M)


def _extract_first_json(s: str) -> str | None:
    """
    Return the first top-level JSON object/array substring ({...} or [...]) from s, or None.
    Pure Python, no recursive regex (compatible with Python 3.13).
    Strategy:
      1) Strip code fences.
      2) Scan for the first '{' or '['.
      3) At each candidate, try JSONDecoder.raw_decode; on success, return the slice.
    """
    if not s:
        return None
    s = _strip_md_fences(s)

    dec = json.JSONDecoder()
    # Find earliest plausible JSON start
    starts = [i for i, ch in enumerate(s) if ch in "{["]
    for i in starts:
        try:
            _, end = dec.raw_decode(s[i:])
            return s[i:i + end]
        except json.JSONDecodeError:
            continue
    return None

def generate_mixed_questions_via_gpt(
    story_text: str,
    language: str = "en",
    level: str = "phonics",
    target_total: int = 10,
    breakdown: dict | None = None
) -> list[dict]:
    if client is None or not story_text:
        base = [
            {"type":"mcq","question":"What does the hero practice?","choices":["Math","New sounds","Cooking","Running"],"correct_index":1},
            {"type":"mcq","question":"Who helps the hero?","choices":["Chef","Friends","Pilot","Doctor"],"correct_index":1},
            {"type":"mcq","question":"How does it end?","choices":["Sadly","No ending","Hopeful","Stormy"],"correct_index":2},
        ]
        while len(base) < target_total:
            base.append({"type":"mcq","question":"Which choice best matches the story?","choices":["A","B","C","D"],"correct_index":0})
        random.shuffle(base)
        return base[:target_total]

    breakdown = breakdown or {"mcq": 6, "short": 3, "long": 1}
    # Be defensive instead of assert (prod-safe)
    total_requested = sum(int(breakdown.get(k, 0)) for k in ("mcq","short","long"))
    if total_requested != target_total:
        # normalize to target_total with default mix
        breakdown = {"mcq": min(target_total, 6), "short": 3 if target_total >= 9 else 1, "long": 1}
        # final clamp
        s = sum(breakdown.values())
        if s != target_total:
            breakdown["mcq"] = max(0, target_total - (breakdown.get("short",0)+breakdown.get("long",0)))

    lang_note = "Write in English." if language != "ko" else "Write in Korean."
    level_note = {
        "phonics": "Use very simple, decodable wording and concrete ideas.",
        "early-reader": "Use simple sentences (8–12 words) and concrete vocabulary.",
        "custom": "General elementary level."
    }.get(level, "Simple sentences.")

    import time
    nonce = f"[regen_nonce:{time.time()}]"

    instructions = f"""
{lang_note} {level_note}
Make EXACTLY {target_total} questions from the STORY with this mix:
- {breakdown.get('mcq',0)} multiple-choice (4 options, one correct; clear, unambiguous).
- {breakdown.get('short',0)} short-answer (include concise expected "answer" and brief "rubric").
- {breakdown.get('long',0)} long-answer (no "answer"; include a short "rubric" with 2–3 grading points).

Keep questions concrete and tied ONLY to the story content. Avoid repeating the same wording.

Return STRICT JSON as a single object like:
{{ "items": [
  {{"type":"mcq","question":"...","choices":["A","B","C","D"],"correct_index":0}},
  {{"type":"short","question":"...","answer":"...","rubric":"..."}},
  {{"type":"long","question":"...","rubric":"..."}}
] }}

STORY:
{story_text}

{nonce}
""".strip()

    raw = ""
    try:
        resp = client.responses.create(
            model=os.getenv("LINGOPLAY_MODEL","gpt-4o-mini"),
            input=[{"role":"user","content": instructions}],
            temperature=0.6,
            max_output_tokens=1500
        )
        raw = _resp_to_text(resp)
        blob = _extract_first_json(raw) or "{}"
        data = json.loads(blob)
        items = data.get("items") or []
    except Exception as e:
        log_input("gpt_questions_error", {"error": str(e), "raw": raw[:1000]})
        items = []

    out = []
    for it in items:
        t = (it.get("type") or "").strip().lower()
        q = (it.get("question") or "").strip()
        if not q:
            continue
        if t == "mcq":
            choices = [str(c) for c in (it.get("choices") or [])][:4]
            if len(choices) < 4:
                choices += ["—"]*(4-len(choices))
            try:
                ci = int(it.get("correct_index", 0))
                if ci not in (0,1,2,3):
                    ci = 0
            except Exception:
                ci = 0
            out.append({"type":"mcq","question":q,"choices":choices,"correct_index":ci})
            continue

        if t == "short":
            ans = (it.get("answer") or "").strip()
            rub = (it.get("rubric") or "").strip() or "Short, story-consistent answer."
            out.append({"type":"short","question":q,"answer":ans,"rubric":rub})
            continue

        # long (or unknown -> treat as long)
        rub = (it.get("rubric") or "").strip() or "Clear structure; uses story details; coherent."
        out.append({"type":"long","question":q,"rubric":rub})

    while len(out) < target_total:
        out.append({"type":"mcq","question":"Which choice best matches the story?","choices":["A","B","C","D"],"correct_index":0})

    random.shuffle(out)
    return out[:target_total]

# ------------------------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------------------------
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

# ------------------------- AUTH: REGISTER / LOGIN / LOGOUT ---------------------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        db = get_db()
        username = (request.form.get("username") or "").strip()
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""
        confirm = request.form.get("confirm") or ""

        # New optional fields
        # radio: 'yes' / 'no' -> 1 / 0 / None
        native_en_raw = (request.form.get("is_english_native") or "").strip().lower()
        if native_en_raw not in ("yes", "no", ""):
            flash("Invalid value for 'Is English your first language?'", "warning")
            return redirect(url_for("register"))
        is_english_native = 1 if native_en_raw == "yes" else (0 if native_en_raw == "no" else None)

        # age is optional
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

        # gender is optional
        gender = (request.form.get("gender") or "").strip().lower() or None
        if gender and gender not in ("female", "male", "nonbinary", "prefer_not"):
            flash("Please choose a valid gender option.", "warning")
            return redirect(url_for("register"))

        # Existing validations
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
                (username, email, pwd_hash, is_english_native, age, gender)
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
                (username,)
            ).fetchone()

            if not user or not check_password_hash(user["password_hash"], password):
                error = "Wrong username or password."
            else:
                session["user_id"] = user["id"]
                flash(f"Welcome back, {user['username']}!", "success")
                if next_url and is_safe_url(next_url):
                    if next_url.endswith("?"):
                        next_url = next_url[:-1]
                    return redirect(next_url)
                return redirect(url_for("index"))

    return render_template("login.html", next=next_url or "", error=error)

def current_user_is_admin() -> bool:
    try:
        return bool(g.get("current_user") and g.current_user.get("is_admin"))
    except Exception:
        return False

@app.post("/logout")
def logout():
    session.pop("user_id", None)
    flash("You have been signed out.", "info")
    return redirect(url_for("index"))
def extract_vocab_candidates(text: str, language: str = "en", exclude_names: set[str] | None = None, max_words: int = 25) -> list[str]:
    """
    Heuristic vocab extractor for early readers.
    - Filters pronouns/particles/stopwords.
    - Filters likely human names (capitalized tokens that never appear lowercase).
    - Keeps short, decodable tokens. EN + basic KO handling.
    - Returns up to `max_words` sorted by frequency (desc), then alpha.
    """
    if not text:
        return []
    exclude_names = {w.lower() for w in (exclude_names or set())}

    # --- Tokenization
    en_tokens = re.findall(r"[A-Za-z']+", text)
    ko_tokens = re.findall(r"[가-힣]+", text)

    # --- Stopwords (compact)
    en_stop = {
        # articles/conjunctions/aux
        "a","an","and","the","to","in","on","at","of","for","from","with","by","as","is","are","was","were",
        "be","been","being","or","but","so","if","then","than","that","this","these","those","there","here",
        "up","down","out","over","under","again","once","just","not","no","do","did","does","have","has","had",
        # pronouns/dets
        "i","you","he","she","we","they","it","me","him","her","us","them",
        "my","your","his","her","our","their","mine","yours","hers","ours","theirs",
        "himself","herself","itself","ourselves","themselves","yourself","yourselves",
        # common time words that add little for early readers
        "day","today","yesterday","tomorrow"
    }
    ko_stop = {"은","는","이","가","을","를","에","에서","으로","와","과","도","만","보다","처럼","요","다","의","한","하고","그","이것","저것","것",
               "나","너","그","그녀","우리","너희","그들","내","네","우리의","너희의","그들의"}

    # --- Name filter (English):
    # Names: capitalized words that NEVER appear lowercase anywhere in text
    # Build sets for lowercase occurrences and titlecase occurrences
    en_lower_occurs = set([t.lower().strip("'") for t in en_tokens if t and t[0].islower()])
    en_title_occurs = set([t for t in en_tokens if t and t[0].isupper() and t[1:].islower() and len(t) >= 3])
    likely_names = {t for t in en_title_occurs if t.lower() not in en_lower_occurs}
    # Exclusion set includes detected names + explicitly provided character names
    exclude_proper = {t.lower() for t in likely_names} | exclude_names

    # --- Frequency
    freq: dict[str,int] = {}

    for t in en_tokens:
        w = t.lower().strip("'")
        if not w or w in en_stop or w in exclude_proper:
            continue
        if not re.match(r"^[a-z]+$", w):
            continue
        if not (2 <= len(w) <= 12):
            continue
        freq[w] = freq.get(w, 0) + 1

    for t in ko_tokens:
        w = t.strip()
        if not w or w in ko_stop or w in exclude_proper:
            continue
        if not (1 <= len(w) <= 6):
            continue
        freq[w] = freq.get(w, 0) + 1

    items = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    out = [w for (w, c) in items if w]  # all lowercase for EN, original for KO (already fine)
    return out[:max_words]
@app.route("/story/new", methods=["GET", "POST"])
@login_required
def story_new():
    db = get_db()
    if request.method == "POST":
        # Core fields
        title = (request.form.get("title") or "").strip() or "My Story"
        prompt = (request.form.get("prompt") or "").strip()
        language = (request.form.get("language") or "en").strip().lower()
        level = (request.form.get("level") or "phonics").strip().lower()

        # "Author" as entered / current user (used for logging + student-finish mode)
        base_author = (g.current_user["username"] if (g.current_user and g.current_user.get("username")) else None) \
                      or (request.form.get("author_name") or "guest")

        want_image = bool(request.form.get("gen_image"))

        theme = (request.form.get("theme") or "").strip()
        characters = (request.form.get("characters") or "").strip()
        tone = (request.form.get("tone") or "").strip()
        bme = bool(request.form.get("bme"))
        # NEW: who finishes the ending?
        student_finish = bool(request.form.get("student_finish"))

        if not prompt:
            flash("Please provide phonics letters or vocabulary.", "warning")
            return redirect(url_for("story_new"))

        # Build a richer prompt for the LLM
        meta_bits = []
        if theme:
            meta_bits.append(f"Theme: {theme}")
        if characters:
            meta_bits.append(f"Main characters: {characters} (keep names consistent).")
        if tone:
            meta_bits.append(f"Tone: {tone}")
        if bme:
            meta_bits.append("Follow a clear Beginning–Middle–End arc.")
        gen_prompt = prompt if not meta_bits else (prompt + "\n\n" + " ".join(meta_bits))

        # 1) Generate story
        try:
            profile = get_learner_profile()
            content = llm_story_from_prompt(gen_prompt, language, level, base_author, learner_profile=profile)
        except Exception as e:
            content = naive_story_from_prompt(prompt, language)
            flash("AI generator had an issue; used a fallback story.", "warning")
            log_input("generate_story_error", {"error": str(e)})

        # 2) Optional cover image
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

        # Decide the author name stored in DB
        # - If students will finish: keep the human / learner name
        # - If AI fully finishes: author is "EduWeaver AI"
        story_author = base_author
        if not student_finish:
            story_author = "EduWeaver AI"

        # 3) Persist story
        slug_base = slugify(title) or "story"
        ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        slug = f"{slug_base}-{ts}"

        db.execute(
            """INSERT INTO stories (title, slug, prompt, language, level, content, visuals, author_name)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (title, slug, prompt, language, level, content, visuals_data_url, story_author),
        )
        story_id = db.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]
        db.commit()

        # 4) Precompute & cache bilingual vocab (skip names/pronouns)
        # 4a) Exclude explicit character names (if provided)
        exclude_names = set()
        if characters:
            raw_names = re.split(r"[,\n;]+", characters)
            for n in raw_names:
                n = re.sub(r"[^A-Za-z가-힣'-]+", " ", n).strip()
                if not n:
                    continue
                for tok in re.findall(r"[A-Za-z]+|[가-힣]+", n):
                    exclude_names.add(tok.lower())

        # 4b) Auto-extract
        auto_candidates = extract_vocab_candidates(
            text=content,
            language=language,
            exclude_names=exclude_names,
            max_words=25
        )

        # 4c) Merge with prompt-derived words (dedup)
        prompt_words = parse_vocab_from_prompt(prompt)
        merged_words = []
        seen = set()
        for w in (prompt_words + auto_candidates):
            lw = (w or "").strip().lower()
            if not lw or lw in seen:
                continue
            seen.add(lw)
            merged_words.append(lw)

        # 4d) Resolve bilingual kid-friendly definitions from multiple sources
        defs_bi = bulk_resolve_kid_definitions_bilingual(merged_words)

        # 4e) Save to DB
        for item in defs_bi:
            db.execute(
                """INSERT INTO vocab_items (story_id, word, definition, example, definition_ko, example_ko, picture_url)
                   VALUES (?, ?, ?, ?, ?, ?, NULL)""",
                (
                    story_id,
                    item["word"].lower(),
                    item.get("definition_en") or "",
                    item.get("example_en") or "",
                    item.get("definition_ko") or "",
                    item.get("example_ko") or "",
                )
            )
        db.commit()

        # Log story generation (keep both who triggered and stored author)
        log_input("generate_story", {
            "prompt": prompt,
            "language": language,
            "level": level,
            "request_user": base_author,
            "db_author": story_author,
            "model": DEFAULT_MODEL,
            "vocab_count": len(defs_bi),
            "with_image": want_image,
            "theme": theme,
            "characters": characters,
            "tone": tone,
            "bme": bme,
            "student_finish": student_finish,
        })

        # 5) Finish-draft behavior depends on student_finish flag
        if student_finish:
            # Teacher wants students to finish the ending later:
            # create an UNFINISHED draft (no completion_text yet)
            partial = make_partial_from_story(content)
            db.execute(
                """INSERT INTO finish_drafts (seed_prompt, partial_text, learner_name, language)
                   VALUES (?, ?, ?, ?)""",
                (prompt, partial, story_author, language),
            )
            draft_id = db.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]
            db.commit()

            log_input("finish_seed_auto_from_story_new", {
                "story_id": story_id,
                "slug": slug,
                "draft_id": draft_id
            })

            # Redirect back with query params to show success modal & deep link to this draft
            return redirect(url_for(
                "story_new",
                generated=1,
                finish_url=url_for("finish_view", draft_id=draft_id)
            ))
        else:
            # Teacher wants GPT to fully finish the story:
            # create a COMPLETED finish_draft with author "EduWeaver AI"
            partial = make_partial_from_story(content)
            db.execute(
                """INSERT INTO finish_drafts (seed_prompt, partial_text, learner_name, language, completion_text)
                   VALUES (?, ?, ?, ?, ?)""",
                (prompt, partial, story_author, language, content),
            )
            draft_id = db.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]
            db.commit()

            log_input("finish_seed_auto_from_story_new_full_ai", {
                "story_id": story_id,
                "slug": slug,
                "draft_id": draft_id,
                "auto_completed": True,
                "db_author": story_author,
            })

            # In Library:
            # - library() shows ALL finish_drafts with completion_text
            # - finish_view() will NOT allow edit because learner_name = "EduWeaver AI"
            flash("Story generated by EduWeaver AI and added to the Library.", "success")
            return redirect(url_for("library"))

    # GET
    return render_template("story_new.html")




# ------- EXTRA IMPORTS (ensure at top of file) -------
import requests
import html
import nltk
from nltk.corpus import wordnet as wn

# One-time WordNet download on first run
def _ensure_wordnet():
    try:
        wn.synsets("tree")
    except LookupError:
        nltk.download("wordnet")
        nltk.download("omw-1.4")

# ------- Kid-friendly format helpers -------
def _truncate_words(s: str, limit: int) -> str:
    s = re.sub(r'\s+', ' ', (s or '').strip())
    if not s:
        return s
    toks = s.split(' ')
    if len(toks) <= limit:
        return s
    return ' '.join(toks[:limit]).rstrip(",;: ") + "."

def _kidify_en(defn: str) -> str:
    defn = re.sub(r"\(.*?\)", "", defn or "")
    defn = re.sub(r";.*", "", defn)
    defn = defn.strip()
    if defn and defn[0].isupper():
        defn = defn[0].lower() + defn[1:]
    return _truncate_words(defn, 16)

def _kidify_example_en(word: str) -> str:
    return _truncate_words(f"I can use the word '{word}' in a sentence.", 12)

def _kidify_ko(defn_ko: str) -> str:
    defn_ko = re.sub(r"\(.*?\)", "", defn_ko or "")
    defn_ko = re.sub(r";.*", "", defn_ko)
    defn_ko = defn_ko.strip()
    return _truncate_words(defn_ko, 16)

def _kidify_example_ko(word: str) -> str:
    return _truncate_words(f"나는 '{word}'라는 말을 문장에 쓸 수 있어요.", 12)

# ------- Sources: WordNet (offline), Datamuse (free), Wiktionary (free) -------
def _wordnet_def(word: str) -> str | None:
    _ensure_wordnet()
    for pos in ('n', 'v', 'a', 'r'):
        syns = wn.synsets(word, pos=pos)
        if syns:
            gloss = syns[0].definition()
            if gloss:
                return gloss
    syns = wn.synsets(word)
    if syns:
        return syns[0].definition()
    return None

def _datamuse_defs(word: str, timeout=4) -> list[str]:
    try:
        r = requests.get(
            "https://api.datamuse.com/words",
            params={"sp": word, "md": "d", "max": 1},
            timeout=timeout,
        )
        if r.status_code != 200:
            return []
        arr = r.json() or []
        if not arr:
            return []
        defs = arr[0].get("defs") or []
        out = []
        for d in defs:
            parts = d.split("\t", 1)
            out.append(parts[1] if len(parts) > 1 else d)
        return out
    except Exception:
        return []

_WIKI_ENDPOINT = "https://en.wiktionary.org/w/api.php"

def _wiktionary_first_def(word: str, lang_header="English", timeout=6) -> str | None:
    """
    Parse raw wikitext; return first '# ' definition under ==Language== section.
    """
    try:
        r = requests.get(_WIKI_ENDPOINT, params={
            "action": "parse",
            "page": word,
            "prop": "wikitext",
            "format": "json",
            "redirects": 1
        }, timeout=timeout)
        if r.status_code != 200:
            return None
        data = r.json()
        wikitext = data.get("parse", {}).get("wikitext", {}).get("*", "")
        if not wikitext:
            return None

        patt = rf"==\s*{re.escape(lang_header)}\s*==(.+?)(\n==|$)"
        m = re.search(patt, wikitext, flags=re.S)
        if not m:
            return None
        block = m.group(1)

        for line in block.splitlines():
            line = line.strip()
            if line.startswith("# ") and not line.startswith("# {{"):
                # strip templates/links
                line = re.sub(r"\{\{.*?\}\}", "", line)
                line = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", line)
                return html.unescape(line[2:].strip())
        return None
    except Exception:
        return None

# ------- Optional translators (set env vars to enable) -------
NAVER_CLIENT_ID  = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")
GOOGLE_API_KEY = "AIzaSyCG14vrQaBjCyidFq_xZKClZCe1U7CdkWA"

def _translate_word_en_to_ko(word: str) -> str | None:
    """
    Prefer Papago/Google to translate the WORD itself to a concise Korean headword.
    Returns None if unavailable.
    """
    # Try Papago first (usually crisper dictionary headwords)
    t = _papago_translate_en_to_ko(word)
    if t:
        return t.strip()

    # Fallback: Google Translate the *word* (not the definition)
    t = _google_translate_en_to_ko(word)
    return t.strip() if t else None


def _papago_translate_en_to_ko(text: str, timeout=6) -> str | None:
    if not (NAVER_CLIENT_ID and NAVER_CLIENT_SECRET):
        return None
    try:
        r = requests.post(
            "https://openapi.naver.com/v1/papago/n2mt",
            headers={
                "X-Naver-Client-Id": NAVER_CLIENT_ID,
                "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
            },
            data={"source": "en", "target": "ko", "text": text},
            timeout=timeout
        )
        if r.status_code != 200:
            return None
        data = r.json()
        return data.get("message", {}).get("result", {}).get("translatedText")
    except Exception:
        return None
# --- dictionary output style toggles ---
DICT_MODE_MINIMAL = True       # no example sentences
KO_HEADWORD_MODE = True        # ko = translate the WORD itself (not the en definition)

def _google_translate_en_to_ko(text: str, timeout=6) -> str | None:
    if not GOOGLE_API_KEY:
        return None
    try:
        r = requests.post(
            f"https://translation.googleapis.com/language/translate/v2?key={GOOGLE_API_KEY}",
            json={"q": text, "source": "en", "target": "ko", "format": "text"},
            timeout=timeout
        )
        if r.status_code != 200:
            return None
        data = r.json()
        arr = data.get("data", {}).get("translations", [])
        print("no eerror")
        if arr:
            return arr[0].get("translatedText")

        return None
    except Exception:
        print("error")
        return None
def resolve_kid_definition_bilingual(word: str) -> dict:
    """
    Returns dict with EN definition + KO headword-style 'definition'.
    - EN: short kid-friendly definition (your existing chain), examples optional.
    - KO: DO NOT translate the EN definition. Instead, map the WORD -> concise Korean headword.
          If the word is already Korean, keep it as-is.
    - If DICT_MODE_MINIMAL, blank out example fields.
    """
    w = (word or "").strip().lower()
    out = {
        "word": w,
        "definition_en": "",
        "example_en": "",
        "definition_ko": "",
        "example_ko": "",
    }
    if not w:
        return out

    # ---------- English definition (same chain as before) ----------
    en_def = _wordnet_def(w)
    if not en_def:
        dm_defs = _datamuse_defs(w)
        if dm_defs:
            en_def = dm_defs[0]
    if not en_def:
        en_def = _wiktionary_first_def(w, lang_header="English")
    if not en_def:
        en_def, _ = kid_def_fallback(w, "en")

    en_def = _kidify_en(en_def)
    out["definition_en"] = en_def

    # Examples (disable if minimal)
    if DICT_MODE_MINIMAL:
        out["example_en"] = ""
    else:
        out["example_en"] = _kidify_example_en(w)

    # ---------- Korean "definition" as headword ----------
    if re.match(r"^[가-힣]+$", w):
        # already Korean; keep the headword as the 'definition'
        ko_def = w
    else:
        if KO_HEADWORD_MODE:
            # translate the WORD itself (not the English definition)
            ko_def = _translate_word_en_to_ko(w)
        else:
            # legacy path: translate EN definition -> KO
            ko_def = _papago_translate_en_to_ko(en_def) or _google_translate_en_to_ko(en_def)

        if not ko_def:
            # last-resort fallback: keep Korean fallback phrase short
            # (You can customize this further if you like)
            ko_def, _ = kid_def_fallback(w, "ko")

    # keep it concise (no examples in minimal mode)
    out["definition_ko"] = _kidify_ko(ko_def)
    out["example_ko"] = "" if DICT_MODE_MINIMAL else _kidify_example_ko(w)

    return out


# ------- Batch resolver (call this from story_new) -------
from nltk.stem import WordNetLemmatizer

_lemmatizer = WordNetLemmatizer()

def _lemmatize_en(token: str) -> str:
    """
    Light lemmatization for English to reduce lookup misses.
    Try noun -> verb -> adj -> adv order.
    """
    w = token
    for pos in ("n", "v", "a", "r"):
        candidate = nltk.corpus.wordnet.morphy(w, pos=pos)
        if candidate:
            w = candidate
            break
    # WordNetLemmatizer as last pass
    w = _lemmatizer.lemmatize(w)
    return w

def bulk_resolve_kid_definitions_bilingual(words: list[str]) -> list[dict]:
    """
    Clean tokens, lemmatize English where helpful, resolve EN+KO kid-friendly defs.
    Returns a list like:
    [{word, definition_en, example_en, definition_ko, example_ko}, ...]
    """
    if not words:
        return []

    clean: list[str] = []
    seen = set()

    for raw in words:
        w = (raw or "").strip().lower()
        if not w:
            continue

        # Accept simple EN or KO tokens only
        if re.match(r"^[a-z']+$", w):
            # strip surrounding apostrophes (e.g., children's -> children’s already cleaned upstream)
            w = w.strip("'")
            # basic length filter
            if not (2 <= len(w) <= 24):
                continue
            # lemmatize to reduce misses (cats -> cat, running -> run)
            w = _lemmatize_en(w)
        elif re.match(r"^[가-힣]+$", w):
            # short heuristic length filter for KO
            if not (1 <= len(w) <= 8):
                continue
        else:
            # skip mixed/complex tokens
            continue

        if w and w not in seen:
            seen.add(w)
            clean.append(w)

    out: list[dict] = []
    for w in clean:
        try:
            item = resolve_kid_definition_bilingual(w)
        except Exception:
            # absolute safety net: never crash vocab build
            de, ee = kid_def_fallback(w, "en")
            dk, ek = kid_def_fallback(w, "ko")
            item = {
                "word": w,
                "definition_en": de, "example_en": ee,
                "definition_ko": dk, "example_ko": ek,
            }
        out.append(item)

    return out


@app.get("/story/<slug>")
@login_required
def story_view(slug):
    db = get_db()
    s = db.execute("SELECT * FROM stories WHERE slug = ?", (slug,)).fetchone()
    if not s:
        return ("Story not found", 404)
    qcount = db.execute(
        "SELECT COUNT(*) AS cnt FROM quiz_questions WHERE story_id = ?", (s["id"],)
    ).fetchone()["cnt"]
    has_quiz = (qcount > 0)

    vocab_items = db.execute(
        "SELECT * FROM vocab_items WHERE story_id = ? ORDER BY id ASC", (s["id"],)
    ).fetchall()

    return render_template("story_view.html", story=s, has_quiz=has_quiz, vocab_items=vocab_items)
@app.post("/story/<slug>/build-worksheet-ai")
@login_required
def build_worksheet_ai(slug):
    import time
    db = get_db()
    s = db.execute("SELECT * FROM stories WHERE slug = ?", (slug,)).fetchone()
    if not s:
        return ("Story not found", 404)

    try:
        questions = generate_mixed_questions_via_gpt(
            story_text=s["content"],
            language=s["language"],
            level=s["level"],
            target_total=10,
            breakdown={"mcq":6,"short":3,"long":1}
        )
    except Exception as e:
        log_input("build_worksheet_ai_gpt_only_error", {"story_id": s["id"], "error": str(e)})
        flash("AI had an issue generating the quiz. Please try again.", "danger")
        return redirect(url_for("story_view", slug=slug))

    db.execute("DELETE FROM quiz_questions WHERE story_id = ?", (s["id"],))
    for item in questions:
        _insert_quiz_question(db, s["id"], item)
    db.commit()

    log_input("build_worksheet_ai_gpt_only", {"story_id": s["id"], "slug": s["slug"], "count": len(questions)})
    flash("Worksheet generated (GPT-only).", "success")
    return redirect(url_for("quiz_take", slug=slug, _=int(time.time())))


@app.route("/quiz/<slug>", methods=["GET", "POST"])
@login_required
def quiz_take(slug):
    db = get_db()
    s = db.execute("SELECT * FROM stories WHERE slug = ?", (slug,)).fetchone()
    if not s:
        return ("Story not found", 404)

    # Draft link (if any) to show on result page
    draft_row = db.execute(
        """
        SELECT id
        FROM finish_drafts
        WHERE seed_prompt = ?
          AND learner_name = ?
          AND language = ?
        ORDER BY datetime(created_at) DESC
        LIMIT 1
        """,
        (s["prompt"], s["author_name"], s["language"])
    ).fetchone()
    draft_id = draft_row["id"] if draft_row else None

    qs = db.execute(
        "SELECT * FROM quiz_questions WHERE story_id = ? ORDER BY id ASC",
        (s["id"],)
    ).fetchall()

    if not qs:
        flash("No quiz yet. Build the worksheet first.", "warning")
        return redirect(url_for("story_view", slug=slug))

    # Ensure qtype defaults for older rows
    for q in qs:
        if not q.get("qtype"):
            q["qtype"] = "mcq"

    if request.method == "POST":
        taker = request.form.get(
            "taker_name",
            g.current_user["username"] if (g.current_user and g.current_user.get("username")) else "guest"
        )

        total, score = len(qs), 0
        details, chosen_map = [], {}
        short_long_answers = {}  # qid -> user text

        for q in qs:
            qtype = (q.get("qtype") or "mcq").lower()
            field = f"q{q['id']}"
            raw = request.form.get(field)

            if qtype == "mcq":
                try:
                    chosen = int(raw)
                except Exception:
                    chosen = -1
                correct = (chosen == q["correct_index"])
                if correct:
                    score += 1
                details.append({
                    "qid": q["id"],
                    "type": "mcq",
                    "chosen": chosen,
                    "correct": q["correct_index"]
                })
                chosen_map[q["id"]] = chosen

            elif qtype == "short":
                ans = (raw or "").strip()
                details.append({
                    "qid": q["id"],
                    "type": "short",
                    "answer": ans,
                    "ref_answer": q.get("answer_text") or "",
                    "rubric": q.get("rubric") or ""
                })
                short_long_answers[q["id"]] = ans

            else:  # long
                essay = (raw or "").strip()
                details.append({
                    "qid": q["id"],
                    "type": "long",
                    "answer": essay,
                    "rubric": q.get("rubric") or ""
                })
                short_long_answers[q["id"]] = essay

        db.execute(
            """INSERT INTO quiz_attempts (story_id, taker_name, score, total, detail_json)
               VALUES (?, ?, ?, ?, ?)""",
            (s["id"], taker, score, total, json.dumps(details)),
        )
        db.commit()

        pct = round((score / total) * 100) if total else 0
        log_input("take_quiz", {"story_id": s["id"], "taker": taker, "score": score, "total": total, "percent": pct})

        # Build view model
        q_for_view = []
        for q in qs:
            qtype = (q.get("qtype") or "mcq").lower()
            row = {
                "id": q["id"],
                "type": qtype,
                "question": q["question"],
                "rubric": q.get("rubric") or ""
            }
            if qtype == "mcq":
                choices = q["choices_json"]
                if isinstance(choices, str):
                    try:
                        choices = json.loads(choices) if choices else []
                    except Exception:
                        choices = []
                row.update({
                    "choices": choices,
                    "correct_index": q["correct_index"],
                    "chosen_index": chosen_map.get(q["id"], -1)
                })
            elif qtype == "short":
                row.update({
                    "user_answer": short_long_answers.get(q["id"], ""),
                    "ref_answer": q.get("answer_text") or ""
                })
            else:  # long
                row.update({
                    "user_answer": short_long_answers.get(q["id"], "")
                })
            q_for_view.append(row)

        # Explanations for MCQs only (optional)
        mcq_items_for_explain = [
            {"question": x["question"], "choices": x.get("choices") or [], "correct_index": x.get("correct_index", 0)}
            for x in q_for_view if x["type"] == "mcq"
        ]
        explanations = []
        if mcq_items_for_explain:
            explanations = explain_answers_via_gpt(
                story_text=s["content"],
                questions=mcq_items_for_explain
            )

        return render_template(
            "quiz_result.html",
            story=s, taker=taker, score=score, total=total, percent=pct,
            questions=q_for_view, explanations=explanations, draft_id=draft_id
        )

    # GET: parse MCQ choices for the template
    for q in qs:
        if q["qtype"] == "mcq":
            ch = q["choices_json"]
            if isinstance(ch, str):
                try:
                    q["choices"] = json.loads(ch) if ch else []
                except Exception:
                    q["choices"] = []
            else:
                q["choices"] = ch or []
        else:
            q["choices"] = []

    return render_template("quiz_take.html", story=s, questions=qs, draft_id=draft_id)

def explain_answers_via_gpt(story_text: str, questions: list[dict]) -> list[str]:
    fallback = [
        "This answer matches a clear detail stated in the story.",
        "The story describes this event directly in that order.",
        "Vocabulary meaning fits how the word is used in context.",
        "This is the main idea repeated across the story.",
        "Sequence is supported by the order of events."
    ]
    if client is None or not questions:
        return (fallback * ((len(questions) // len(fallback)) + 1))[:len(questions)]
    qpack = []
    for i, q in enumerate(questions):
        qpack.append({
            "index": i,
            "question": q["question"],
            "choices": q.get("choices") or json.loads(q.get("choices_json") or "[]"),
            "correct_index": q["correct_index"],
        })
    system = (
        "You explain quiz answers for a short children's story. "
        "Return STRICT JSON ONLY (no markdown): "
        "{ \"explanations\": [\"...\", \"...\", ...] } with one short reason per question, "
        "each ≤18 words, concrete and child-friendly."
    )
    user = "STORY:\n" + story_text + "\n\nQUESTIONS:\n" + json.dumps(qpack, ensure_ascii=False) + "\n\nGive one simple reason per question in order."
    try:
        resp = client.responses.create(
            model=DEFAULT_MODEL,
            input=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.3,
            max_output_tokens=600,
        )
        raw = (getattr(resp, "output_text", "") or "").strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I | re.M)
        data = json.loads(raw)
        exps = data.get("explanations") or []
        exps = [str(x).strip() for x in exps]
        if not exps or len(exps) < len(questions):
            exps += (fallback * 5)
        return exps[:len(questions)]
    except Exception:
        return (fallback * ((len(questions) // len(fallback)) + 1))[:len(questions)]

@app.post("/story/<slug>/to-finish")
@login_required
def story_to_finish(slug):
    db = get_db()
    s = db.execute("SELECT * FROM stories WHERE slug = ?", (slug,)).fetchone()
    if not s:
        return ("Story not found", 404)

    # Only the original author can start a finish draft
    current_username = g.current_user["username"]
    if s["author_name"] != current_username:
        flash("Only the story’s author can finish this story.", "danger")
        return redirect(url_for("story_view", slug=slug))

    partial = make_partial_from_story(s["content"])
    db.execute(
        """INSERT INTO finish_drafts (seed_prompt, partial_text, learner_name, language)
           VALUES (?, ?, ?, ?)""",
        (s["prompt"], partial, current_username, s["language"]),
    )
    draft_id = db.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]
    db.commit()

    log_input("finish_seed_from_story", {"story_id": s["id"], "draft_id": draft_id})
    return redirect(url_for("finish_view", draft_id=draft_id))
def admin_required(view_func):
    @wraps(view_func)
    def _wrapped(*args, **kwargs):
        u = g.get("current_user")
        if not (u and u.get("is_admin")):
            flash("Admins only.", "danger")
            return redirect(url_for("index"))
        return view_func(*args, **kwargs)
    return _wrapped
# ------------------------ ADMIN ANALYTICS UI ------------------------
@app.get("/admin/analytics")
@login_required
@admin_required
def admin_analytics():
    return render_template("admin_analytics.html")

from collections import defaultdict

def _parse_dt(s: str):
    try:
        s = (s or "").replace("Z","")
        if "." in s: s = s.split(".",1)[0]
        return datetime.fromisoformat(s)
    except Exception:
        return None

@app.get("/api/admin/overview")
@login_required
@admin_required
def api_admin_overview():
    """
    Totals + 30-day time series for stories, finishes, quiz attempts; mean quiz score.
    """
    db = get_db()

    total_users = db.execute("SELECT COUNT(*) AS c FROM users").fetchone()["c"]
    total_stories = db.execute("SELECT COUNT(*) AS c FROM stories").fetchone()["c"]
    total_finishes = db.execute("""
        SELECT COUNT(*) AS c FROM finish_drafts
        WHERE completion_text IS NOT NULL AND TRIM(completion_text) <> ''
    """).fetchone()["c"]
    total_attempts = db.execute("SELECT COUNT(*) AS c FROM quiz_attempts").fetchone()["c"]
    avg_score_row = db.execute("""
        SELECT AVG(CAST(score AS FLOAT)/NULLIF(total,0)) AS p
        FROM quiz_attempts WHERE total > 0
    """).fetchone()
    avg_score = round((avg_score_row["p"] or 0)*100, 1)

    # 30-day buckets (UTC naive)
    def daily_counts(query):
        rows = db.execute(query).fetchall()
        buckets = defaultdict(int)
        for r in rows:
            t = _parse_dt(r["created_at"])
            if not t: continue
            d = t.date().isoformat()
            buckets[d] += 1
        return buckets

    s_b = daily_counts("SELECT created_at FROM stories WHERE created_at >= DATETIME('now','-30 day')")
    f_b = daily_counts("""
        SELECT created_at FROM finish_drafts
        WHERE completion_text IS NOT NULL AND TRIM(completion_text) <> ''
          AND created_at >= DATETIME('now','-30 day')
    """)
    q_b = daily_counts("SELECT created_at FROM quiz_attempts WHERE created_at >= DATETIME('now','-30 day')")

    # build continuous date axis
    days = []
    today = datetime.utcnow().date()
    for i in range(29, -1, -1):
        d = (today).fromordinal(today.toordinal() - i)
        iso = d.isoformat()
        days.append(iso)

    return {
        "totals": {
            "users": total_users,
            "stories": total_stories,
            "finishes": total_finishes,
            "attempts": total_attempts,
            "avg_score_pct": avg_score
        },
        "series": {
            "labels": days,
            "stories": [s_b.get(d,0) for d in days],
            "finishes": [f_b.get(d,0) for d in days],
            "attempts": [q_b.get(d,0) for d in days],
        }
    }

@app.get("/api/admin/l1l2")
@login_required
@admin_required
def api_admin_l1l2():
    """
    Compare L1 (native EN) vs L2 across production metrics.
    uses users.is_english_native: 1=L1, 0=L2, NULL=unknown
    """
    db = get_db()

    def summarize(group_sql):
        # stories authored
        s = db.execute(f"""
          SELECT COUNT(*) AS c FROM stories
          WHERE author_name IN ({group_sql})
        """).fetchone()["c"]

        # completed finishes written
        f = db.execute(f"""
          SELECT COUNT(*) AS c FROM finish_drafts
          WHERE completion_text IS NOT NULL AND TRIM(completion_text) <> ''
            AND learner_name IN ({group_sql})
        """).fetchone()["c"]

        # quiz attempts + average
        row = db.execute(f"""
          SELECT COUNT(*) AS n, AVG(CAST(score AS FLOAT)/NULLIF(total,0)) AS p
          FROM quiz_attempts
          WHERE taker_name IN ({group_sql}) AND total > 0
        """).fetchone()
        attempts = row["n"] or 0
        avg_pct = round((row["p"] or 0)*100, 1) if attempts else 0.0
        return s, f, attempts, avg_pct

    # build subqueries (usernames by group)
    l1_users_sql = "SELECT username FROM users WHERE is_english_native=1"
    l2_users_sql = "SELECT username FROM users WHERE is_english_native=0"
    unk_users_sql = "SELECT username FROM users WHERE is_english_native IS NULL"

    l1 = summarize(l1_users_sql)
    l2 = summarize(l2_users_sql)
    uk = summarize(unk_users_sql)

    return {
        "groups": ["L1", "L2", "Unknown"],
        "stories": [l1[0], l2[0], uk[0]],
        "finishes": [l1[1], l2[1], uk[1]],
        "attempts": [l1[2], l2[2], uk[2]],
        "avg_scores": [l1[3], l2[3], uk[3]]
    }
@app.get("/api/admin/users")
@login_required
@admin_required
def api_admin_users():
    """
    Lightweight per-user rollup for a table; includes last activity.
    """
    db = get_db()
    rows = db.execute("""
      SELECT
        u.id, u.username, u.email, u.is_english_native, u.age, u.gender, u.created_at,
        (SELECT COUNT(*) FROM stories s WHERE s.author_name = u.username) AS story_count,
        (SELECT COUNT(*) FROM finish_drafts fd
          WHERE fd.learner_name = u.username
            AND fd.completion_text IS NOT NULL AND TRIM(fd.completion_text) <> ''
        ) AS finish_count,
        (SELECT COUNT(*) FROM quiz_attempts qa WHERE qa.taker_name = u.username) AS attempt_count,
        (SELECT MAX(x.dt) FROM (
            SELECT MAX(s.created_at) AS dt FROM stories s WHERE s.author_name=u.username
            UNION ALL
            SELECT MAX(fd.created_at) FROM finish_drafts fd WHERE fd.learner_name=u.username
            UNION ALL
            SELECT MAX(qa.created_at) FROM quiz_attempts qa WHERE qa.taker_name=u.username
        ) AS x) AS last_activity
      FROM users u
      ORDER BY LOWER(u.username) ASC
    """).fetchall()

    def as_bool(v):
        return None if v is None else bool(int(v))

    out = []
    for r in rows:
        out.append({
            "id": r["id"],
            "username": r["username"],
            "email": r["email"],
            "l1": as_bool(r["is_english_native"]),
            "age": r["age"],
            "gender": r["gender"],
            "created_at": r["created_at"],
            "story_count": r["story_count"],
            "finish_count": r["finish_count"],
            "attempt_count": r["attempt_count"],
            "last_activity": r["last_activity"]
        })
    return {"items": out}
@app.get("/api/admin/user/<int:user_id>")
@login_required
@admin_required
def api_admin_user_detail(user_id):
    """
    Detailed time-series for a single user (last 60 days)
    """
    db = get_db()
    u = db.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    if not u:
        return {"ok": False, "error": "user not found"}, 404

    uname = u["username"]

    def ts(query):
        rows = db.execute(query, (uname,)).fetchall()
        buckets = defaultdict(int)
        for r in rows:
            t = _parse_dt(r["created_at"])
            if not t: continue
            buckets[t.date().isoformat()] += 1
        return buckets

    s_b = ts("SELECT created_at FROM stories WHERE author_name=? AND created_at>=DATETIME('now','-60 day')")
    f_b = ts("""
        SELECT created_at FROM finish_drafts
        WHERE learner_name=? AND created_at>=DATETIME('now','-60 day')
          AND completion_text IS NOT NULL AND TRIM(completion_text) <> ''
    """)
    q_b = ts("SELECT created_at FROM quiz_attempts WHERE taker_name=? AND created_at>=DATETIME('now','-60 day')")

    labels = []
    today = datetime.utcnow().date()
    for i in range(59, -1, -1):
        d = (today).fromordinal(today.toordinal() - i)
        labels.append(d.isoformat())

    # last 10 attempts w/ percent
    last_attempts = db.execute("""
        SELECT created_at, score, total
        FROM quiz_attempts
        WHERE taker_name = ?
        ORDER BY datetime(created_at) DESC LIMIT 10
    """, (uname,)).fetchall()
    last_attempts = [
        {"when": r["created_at"], "percent": (round((r["score"]/r["total"])*100,1) if r["total"] else 0)}
        for r in last_attempts
    ]

    return {
        "ok": True,
        "user": {"id": u["id"], "username": uname, "l1": u["is_english_native"]},
        "series": {
            "labels": labels,
            "stories": [s_b.get(d,0) for d in labels],
            "finishes": [f_b.get(d,0) for d in labels],
            "attempts": [q_b.get(d,0) for d in labels],
        },
        "attempts_recent": last_attempts
    }

from datetime import datetime, timedelta
import json

@app.post("/api/admin/user_reflection/<int:user_id>")
@login_required
def api_admin_user_reflection(user_id: int):
    """
    Generate short, individualized feedback for a single learner,
    based on their story and quiz activity.
    """
    if not (g.current_user and g.current_user.get("is_admin")):
        return jsonify({"ok": False, "error": "forbidden"}), 403

    db = get_db()

    # Look up user
    user_row = db.execute(
        "SELECT id, username, created_at FROM users WHERE id = ?",
        (user_id,)
    ).fetchone()
    if not user_row:
        return jsonify({"ok": False, "error": "not_found"}), 404

    username = user_row["username"]

    # Basic story stats
    story_row = db.execute(
        "SELECT COUNT(*) AS cnt FROM stories WHERE author_name = ?",
        (username,)
    ).fetchone()
    story_count = (story_row["cnt"] if story_row else 0) or 0

    # Quiz attempts (last ~50, newest first)
    attempts = db.execute(
        """
        SELECT score, total, created_at
        FROM quiz_attempts
        WHERE taker_name = ?
        ORDER BY created_at DESC
        LIMIT 50
        """,
        (username,)
    ).fetchall()

    attempt_count = len(attempts)
    percents = []
    for row in attempts:
        total = row["total"] or 0
        if total > 0:
            pct = round(row["score"] * 100.0 / total)
            percents.append(pct)

    avg_percent = round(sum(percents) / len(percents), 1) if percents else None
    best_percent = max(percents) if percents else None
    worst_percent = min(percents) if percents else None
    recent_percents = percents[:5]

    # Simple "activity last 30d" measure from attempts
    cutoff = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S")
    active_days_30 = 0
    if attempts:
        days = set()
        for row in attempts:
            ts = row["created_at"]  # TEXT "YYYY-MM-DD HH:MM:SS"
            if ts >= cutoff:
                days.add(ts[:10])
        active_days_30 = len(days)

    payload = {
        "username": username,
        "story_count": story_count,
        "attempt_count": attempt_count,
        "avg_percent": avg_percent,
        "best_percent": best_percent,
        "worst_percent": worst_percent,
        "recent_percents": recent_percents,
        "active_days_30": active_days_30,
    }

    # Build a short, structured prompt for the model
    prompt = (
        "You are an encouraging English teacher writing a brief feedback note "
        "for one learner, based on their story writing and quiz performance.\n\n"
        "Use 3–5 sentences. Be specific but kind. Mention:\n"
        "- Overall effort and participation (stories written, quiz attempts).\n"
        "- Typical quiz performance level.\n"
        "- 1–2 concrete strengths.\n"
        "- 1–2 gentle suggestions for next steps.\n"
        "Avoid technical terms and percentages overload; keep it student-friendly.\n\n"
        f"Data (JSON):\n{json.dumps(payload, ensure_ascii=False)}"
    )

    try:
        resp = client.responses.create(
            model=DEFAULT_MODEL,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are a supportive EFL (English as a Foreign Language) teacher. "
                        "Write short, friendly feedback notes that a middle or high school "
                        "student can easily understand."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )

        # Helper that matches your other uses of responses API
        def extract_text(r):
            try:
                return r.output_text
            except Exception:
                try:
                    # fallback for older-style responses
                    return "".join(
                        block.text.value
                        for block in r.output[0].content
                        if getattr(block, "type", None) == "output_text"
                        or getattr(getattr(block, "text", None), "value", None)
                    )
                except Exception:
                    return ""

        text = (extract_text(resp) or "").strip()
        if not text:
            return jsonify({"ok": False, "text": ""})

        # Front-end supports either `feedback` or plain `text`
        return jsonify({"ok": True, "text": text})

    except Exception as e:
        app.logger.exception("user_reflection failed for user %s: %s", user_id, e)
        return jsonify({"ok": False, "error": "model_error"}), 500

@app.post("/api/admin/research/reflection")
@login_required
@admin_required
def api_admin_research_reflection():
    """
    Summarize L1 vs L2 characteristics + possible teaching actions from current aggregates.
    Returns plain text for display/clipboard.
    """
    try:
        l1l2 = api_admin_l1l2()[0] if isinstance(api_admin_l1l2(), tuple) else api_admin_l1l2()
    except Exception:
        l1l2 = {}

    try:
        overview = api_admin_overview()[0] if isinstance(api_admin_overview(), tuple) else api_admin_overview()
    except Exception:
        overview = {}

    prompt = (
        "You are an educational researcher. Based on the following JSON aggregates, write a concise analysis "
        "comparing L1 (native English) vs L2 learners in terms of production (stories/finishes), quiz participation, and mean scores. "
        "Suggest 3 actionable teaching adjustments tailored for L2 without disadvantaging L1, and 2 fair assessment ideas.\n\n"
        f"OVERVIEW:\n{json.dumps(overview, ensure_ascii=False)}\n\n"
        f"L1L2:\n{json.dumps(l1l2, ensure_ascii=False)}\n\n"
        "Output 3 sections with short bullets:\n"
        "1) Observations\n2) Teaching adjustments\n3) Assessment ideas\n"
        "Avoid hedging language; keep under 220 words."
    )

    txt = "No model configured."
    try:
        resp = client.responses.create(
            model=DEFAULT_MODEL,
            input=[{"role":"user","content":prompt}],
            temperature=0.3,
            max_output_tokens=700
        )
        txt = (getattr(resp, "output_text", "") or "").strip()
    except Exception as e:
        txt = f"Could not run analysis: {e}"

    return {"text": txt}

@app.route("/finish", methods=["GET", "POST"])
@login_required
def finish_new():
    db = get_db()
    if request.method == "POST":
        seed = (request.form.get("seed_prompt") or "").strip()
        language = request.form.get("language", "en")

        learner = g.current_user["username"]

        if not seed:
            flash("Please add a short seed (phonics/vocab).", "warning")
            return redirect(url_for("finish_new"))

        try:
            full = llm_story_from_prompt(seed, language, "phonics", learner, learner_profile=get_learner_profile())
        except Exception:
            full = naive_story_from_prompt(seed, language)

        partial = make_partial_from_story(full)

        db.execute(
            """INSERT INTO finish_drafts (seed_prompt, partial_text, learner_name, language)
               VALUES (?, ?, ?, ?)""",
            (seed, partial, learner, language),
        )
        draft_id = db.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]
        db.commit()

        log_input("finish_seed", {"seed": seed, "language": language, "learner": learner})
        resp = redirect(url_for("finish_view", draft_id=draft_id))
        resp.headers["Cache-Control"] = "no-store"
        return resp

    # GET: list unfinished drafts for the current user
    learner = g.current_user["username"]
    drafts = db.execute(
        """
        SELECT
          fd.id,
          fd.seed_prompt,
          fd.created_at,
          fd.language,
          (
            SELECT s.title
            FROM stories s
            WHERE s.prompt = fd.seed_prompt
              AND s.author_name = fd.learner_name
              AND s.language = fd.language
            ORDER BY datetime(s.created_at) DESC
            LIMIT 1
          ) AS story_title
        FROM finish_drafts fd
        WHERE fd.learner_name = ?
          AND (fd.completion_text IS NULL OR TRIM(fd.completion_text) = '')
        ORDER BY datetime(fd.created_at) DESC
        """,
        (learner,)
    ).fetchall()
    return render_template("finish_new.html", drafts=drafts)
@app.post("/finish/<int:draft_id>/comment")
@login_required
def finish_comment(draft_id):
    db = get_db()
    d = db.execute("SELECT id FROM finish_drafts WHERE id = ?", (draft_id,)).fetchone()
    if not d:
        flash("Draft not found.", "danger")
        return redirect(url_for("finish_new"))

    if not (g.current_user and g.current_user.get("is_admin")):
        flash("Only admins can post comments.", "danger")
        return redirect(url_for("finish_view", draft_id=draft_id))

    body = (request.form.get("body") or "").strip()
    if not body:
        flash("Comment cannot be empty.", "warning")
        return redirect(url_for("finish_view", draft_id=draft_id))

    author = g.current_user.get("username") or "admin"
    db.execute(
        "INSERT INTO finish_comments (draft_id, author_name, body) VALUES (?, ?, ?)",
        (draft_id, author, body)
    )
    db.commit()
    flash("Comment posted.", "success")
    return redirect(url_for("finish_view", draft_id=draft_id))
@app.route("/finish/<int:draft_id>", methods=["GET", "POST"])
@login_required
def finish_view(draft_id):
    db = get_db()

    # Load draft
    d = db.execute("SELECT * FROM finish_drafts WHERE id = ?", (draft_id,)).fetchone()
    if not d:
        return ("Draft not found", 404)

    current_username = g.current_user["username"]
    is_owner = (d["learner_name"] == current_username)

    # Save completion
    if request.method == "POST":
        if not is_owner:
            flash("Only the author can edit this ending.", "danger")
            return redirect(url_for("finish_view", draft_id=draft_id))
        completion = (request.form.get("completion_text") or "").strip()
        db.execute("UPDATE finish_drafts SET completion_text = ? WHERE id = ?", (completion, draft_id))
        db.commit()
        log_input("finish_submit", {"draft_id": draft_id, "has_completion": bool(completion)})
        flash("Your ending has been saved.", "success")
        return redirect(url_for("finish_view", draft_id=draft_id, saved=1))

    # Find linked story (NOW ALSO GET visuals)
    story_row = db.execute(
        """
        SELECT id, title, slug, content, language, level, visuals
        FROM stories
        WHERE prompt = ?
          AND author_name = ?
          AND language = ?
        ORDER BY datetime(created_at) DESC
        LIMIT 1
        """,
        (d["seed_prompt"], d["learner_name"], d["language"])
    ).fetchone()

    d["story_title"]   = story_row["title"]   if story_row else None
    d["story_slug"]    = story_row["slug"]    if story_row else None
    d["story_id"]      = story_row["id"]      if story_row else None
    d["story_visuals"] = story_row["visuals"] if story_row else None   # <-- NEW

    has_quiz = False
    if story_row:
        qcount = db.execute(
            "SELECT COUNT(*) AS cnt FROM quiz_questions WHERE story_id = ?",
            (story_row["id"],)
        ).fetchone()["cnt"]
        has_quiz = (qcount > 0)

    # Non-owner cannot view unfinished
    if (not is_owner) and (not d["completion_text"] or not d["completion_text"].strip()):
        flash("This story isn’t finished yet. Check back later!", "info")
        return redirect(url_for("library"))

    # Load precomputed vocab (bilingual)
    vocab_words = []
    if story_row:
        vrows = db.execute(
            "SELECT word, definition, example, definition_ko, example_ko "
            "FROM vocab_items WHERE story_id = ? ORDER BY id ASC",
            (story_row["id"],)
        ).fetchall()
        for r in vrows:
            vocab_words.append({
                "word": (r.get("word") or "").lower(),
                "definition_en": r.get("definition") or "",
                "example_en":    r.get("example") or "",
                "definition_ko": r.get("definition_ko") or "",
                "example_ko":    r.get("example_ko") or "",
            })

    # Load comments
    comments = db.execute(
        """
        SELECT author_name, body, created_at
        FROM finish_comments
        WHERE draft_id = ?
        ORDER BY datetime(created_at) ASC
        """,
        (draft_id,)
    ).fetchall()

    is_admin = bool(g.current_user and g.current_user.get("is_admin"))

    return render_template(
        "finish_view.html",
        draft=d,
        can_edit=is_owner,
        has_quiz=has_quiz,
        vocab_words=vocab_words,
        comments=comments,
        is_admin=is_admin,
    )

@app.post("/library/delete/<int:draft_id>")
@login_required
def library_delete_draft(draft_id: int):
    if not current_user_is_admin():
        return ("Forbidden", 403)

    db = get_db()
    # Make sure it exists and is completed (matches what shows up in Library)
    d = db.execute(
        """
        SELECT id, seed_prompt, learner_name, language, completion_text
        FROM finish_drafts
        WHERE id = ?
        """,
        (draft_id,)
    ).fetchone()

    if not d:
        flash("Draft not found.", "warning")
        return redirect(url_for("library"))

    if not d.get("completion_text") or not str(d.get("completion_text")).strip():
        # Not in Library anyway, but still guard
        flash("This item is not a completed story.", "warning")
        return redirect(url_for("library"))

    # Delete just the completed finish_draft “article”
    db.execute("DELETE FROM finish_drafts WHERE id = ?", (draft_id,))
    db.commit()

    log_input("library_delete_draft", {"draft_id": draft_id, "by": g.current_user["username"]})
    flash("The story was deleted.", "success")
    return redirect(url_for("library"))



@app.get("/library")
@login_required
def library():
    db = get_db()
    completes = db.execute(
        """
        SELECT
          fd.id,
          fd.seed_prompt,
          fd.learner_name,
          fd.language,
          fd.created_at,
          (
            SELECT s.title
            FROM stories s
            WHERE s.prompt = fd.seed_prompt
              AND s.author_name = fd.learner_name
              AND s.language = fd.language
            ORDER BY datetime(s.created_at) DESC
            LIMIT 1
          ) AS story_title
        FROM finish_drafts fd
        WHERE fd.completion_text IS NOT NULL
          AND TRIM(fd.completion_text) <> ''
        ORDER BY datetime(fd.created_at) DESC
        """
    ).fetchall()
    return render_template("library.html", completes=completes)

@app.get("/dashboard")
@login_required
def dashboard():
    db = get_db()
    learner = g.current_user["username"]

    story_count = db.execute(
        "SELECT COUNT(*) AS c FROM stories WHERE author_name = ?",
        (learner,)
    ).fetchone()["c"]

    completed_count = db.execute(
        """
        SELECT COUNT(*) AS c
        FROM finish_drafts
        WHERE learner_name = ?
          AND completion_text IS NOT NULL
          AND TRIM(completion_text) <> ''
        """,
        (learner,)
    ).fetchone()["c"]

    unfinished_drafts = db.execute(
        """
        SELECT
          fd.id,
          fd.seed_prompt,
          fd.created_at,
          fd.language,
          (
            SELECT s.title
            FROM stories s
            WHERE s.prompt = fd.seed_prompt
              AND s.author_name = fd.learner_name
              AND s.language = fd.language
            ORDER BY datetime(s.created_at) DESC
            LIMIT 1
          ) AS story_title
        FROM finish_drafts fd
        WHERE fd.learner_name = ?
          AND (fd.completion_text IS NULL OR TRIM(fd.completion_text) = '')
        ORDER BY datetime(fd.created_at) DESC
        LIMIT 25
        """,
        (learner,)
    ).fetchall()

    completed_drafts = db.execute(
        """
        SELECT
          fd.id,
          fd.seed_prompt,
          fd.created_at,
          fd.language,
          (
            SELECT s.title
            FROM stories s
            WHERE s.prompt = fd.seed_prompt
              AND s.author_name = fd.learner_name
              AND s.language = fd.language
            ORDER BY datetime(s.created_at) DESC
            LIMIT 1
          ) AS story_title
        FROM finish_drafts fd
        WHERE fd.learner_name = ?
          AND fd.completion_text IS NOT NULL
          AND TRIM(completion_text) <> ''
        ORDER BY datetime(fd.created_at) DESC
        LIMIT 25
        """,
        (learner,)
    ).fetchall()

    attempts = db.execute(
        """
        SELECT * FROM quiz_attempts
        WHERE taker_name = ?
        ORDER BY datetime(created_at) DESC
        LIMIT 25
        """,
        (learner,)
    ).fetchall()

    quizzes_ready = db.execute(
        """
        SELECT
          s.id,
          s.title,
          s.slug,
          s.created_at,
          (SELECT COUNT(*) FROM quiz_questions qq WHERE qq.story_id = s.id) AS qcount,
          (SELECT qa.score
             FROM quiz_attempts qa
            WHERE qa.story_id = s.id AND qa.taker_name = ?
            ORDER BY datetime(qa.created_at) DESC
            LIMIT 1) AS last_score,
          (SELECT qa.total
             FROM quiz_attempts qa
            WHERE qa.story_id = s.id AND qa.taker_name = ?
            ORDER BY datetime(qa.created_at) DESC
            LIMIT 1) AS last_total,
          (SELECT qa.created_at
             FROM quiz_attempts qa
            WHERE qa.story_id = s.id AND qa.taker_name = ?
            ORDER BY datetime(qa.created_at) DESC
            LIMIT 1) AS last_when
        FROM stories s
        WHERE s.author_name = ?
          AND EXISTS (SELECT 1 FROM quiz_questions qq WHERE qq.story_id = s.id)
        ORDER BY datetime(s.created_at) DESC
        LIMIT 25
        """,
        (learner, learner, learner, learner)
    ).fetchall()

    return render_template(
        "dashboard.html",
        story_count=story_count,
        finish_count=completed_count,
        attempts=attempts,
        unfinished_drafts=unfinished_drafts,
        completed_drafts=completed_drafts,
        quizzes_ready=quizzes_ready,
    )

@app.get("/dashboard/export.csv")
@login_required
def dashboard_export():
    db = get_db()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["type", "id", "created_at", "language_or_na", "prompt_or_seed"])

    for s in db.execute("SELECT id, created_at, language, prompt FROM stories").fetchall():
        writer.writerow(["story", s["id"], s["created_at"], s["language"], s["prompt"]])

    for d in db.execute("SELECT id, created_at, language, seed_prompt FROM finish_drafts").fetchall():
        writer.writerow(["finish", d["id"], d["created_at"], d["language"], d["seed_prompt"]])

    mem = io.BytesIO(output.getvalue().encode("utf-8"))
    mem.seek(0)
    return send_file(mem, mimetype="text/csv", as_attachment=True, download_name="lingoplay_export.csv")
@app.post("/dashboard/delete-finish/<int:draft_id>")
@login_required
def dashboard_delete_finish(draft_id: int):
    db = get_db()
    d = db.execute(
        "SELECT id, learner_name, completion_text FROM finish_drafts WHERE id = ?",
        (draft_id,)
    ).fetchone()

    if not d:
        flash("Draft not found.", "warning")
        return redirect(url_for("dashboard"))

    # Only the owner can delete on Dashboard
    if d["learner_name"] != g.current_user["username"]:
        flash("You can only delete your own drafts.", "danger")
        return redirect(url_for("dashboard"))

    # Delete the draft (unfinished or finished—Dashboard shows both lists)
    db.execute("DELETE FROM finish_drafts WHERE id = ?", (draft_id,))
    db.commit()
    log_input("dashboard_delete_finish", {
        "draft_id": draft_id,
        "by": g.current_user["username"],
        "finished": bool(d.get("completion_text") and str(d.get("completion_text")).strip())
    })
    flash("The draft was deleted.", "success")
    return redirect(url_for("dashboard"))


@app.post("/dashboard/delete-story/<int:story_id>")
@login_required
def dashboard_delete_story(story_id: int):
    db = get_db()
    s = db.execute(
        "SELECT id, title, author_name FROM stories WHERE id = ?",
        (story_id,)
    ).fetchone()

    if not s:
        flash("Story not found.", "warning")
        return redirect(url_for("dashboard"))

    # Only the owner can delete their story
    if s["author_name"] != g.current_user["username"]:
        flash("You can only delete your own stories.", "danger")
        return redirect(url_for("dashboard"))

    # Delete story; quiz_questions/vocab_items cascade by FK
    db.execute("DELETE FROM stories WHERE id = ?", (story_id,))
    db.commit()
    log_input("dashboard_delete_story", {"story_id": story_id, "by": g.current_user["username"]})
    flash("The story was deleted.", "success")
    return redirect(url_for("dashboard"))

# keep the choices-json template filter used in quiz rendering
@app.template_filter("load_json")
def load_json_filter_choices(s):
    try:
        val = json.loads(s) if s else []
        return val if isinstance(val, list) else []
    except Exception:
        return []
@app.post("/finish/<int:draft_id>/build-worksheet-ai")
@login_required
def finish_build_worksheet_ai(draft_id):
    import time
    db = get_db()
    d = db.execute("SELECT * FROM finish_drafts WHERE id = ?", (draft_id,)).fetchone()
    if not d:
        flash("Draft not found.", "danger")
        return redirect(url_for("finish_new"))

    # Find or create story bound to this draft
    story = db.execute(
        """
        SELECT id, title, slug, content, language, level, prompt, author_name
        FROM stories
        WHERE prompt = ? AND author_name = ? AND language = ?
        ORDER BY datetime(created_at) DESC
        LIMIT 1
        """,
        (d["seed_prompt"], d["learner_name"], d["language"])
    ).fetchone()

    if not story:
        partial = (d.get("partial_text") or "").replace("", "").strip()
        completion = (d.get("completion_text") or "").strip()
        content_text = (partial + (" " + completion if completion else "")).strip() or "A short learner story."

        title = (d["seed_prompt"] or "My Story").strip()
        if len(title) > 48:
            title = title[:48].rstrip() + "…"
        slug = f"{slugify(title)}-{int(datetime.utcnow().timestamp())}"

        db.execute(
            """INSERT INTO stories (title, slug, prompt, language, level, content, author_name, visuals)
               VALUES (?, ?, ?, ?, ?, ?, ?, NULL)""",
            (title, slug, d["seed_prompt"], d["language"] or "en", "phonics", content_text, d["learner_name"])
        )
        db.commit()
        story = db.execute(
            "SELECT id, title, slug, content, language, level FROM stories WHERE slug = ?",
            (slug,)
        ).fetchone()

    try:
        questions = generate_mixed_questions_via_gpt(
            story_text=story["content"],
            language=story["language"],
            level=story["level"],
            target_total=10,
            breakdown={"mcq":6,"short":3,"long":1}
        )
    except Exception as e:
        log_input("finish_build_worksheet_ai_gpt_only_error", {"draft_id": draft_id, "error": str(e)})
        flash("AI had an issue generating the quiz. Please try again.", "danger")
        return redirect(url_for("finish_view", draft_id=draft_id))

    db.execute("DELETE FROM quiz_questions WHERE story_id = ?", (story["id"],))
    for item in questions:
        _insert_quiz_question(db, story["id"], item)
    db.commit()

    log_input("finish_build_worksheet_ai_gpt_only", {
        "draft_id": draft_id, "story_id": story["id"], "slug": story.get("slug"), "count": len(questions)
    })
    flash("Worksheet & quiz generated (GPT-only).", "success")
    return redirect(url_for("quiz_take", slug=story["slug"], _=int(time.time())))


from flask import jsonify

@app.get("/api/define")
@login_required
def api_define():
    """
    Resolve a word’s bilingual definition/example for a given story.
    Order: vocab_items (precomputed) -> fallbacks.
    Expect ?word=...&story_id=...  (language param not needed now)
    """
    word = (request.args.get("word") or "").strip().lower()
    try:
        story_id = int(request.args.get("story_id") or 0)
    except Exception:
        story_id = 0

    if not word:
        return jsonify({"ok": False, "error": "missing word"}), 400
    if not story_id:
        return jsonify({"ok": False, "error": "missing story_id"}), 400

    db = get_db()

    row = db.execute(
        """SELECT definition, example, definition_ko, example_ko
           FROM vocab_items
           WHERE story_id = ? AND lower(word) = ?""",
        (story_id, word)
    ).fetchone()

    if row:
        return jsonify({
            "ok": True,
            "word": word,
            "definition_en": row.get("definition") or "",
            "example_en":    row.get("example") or "",
            "definition_ko": row.get("definition_ko") or "",
            "example_ko":    row.get("example_ko") or "",
        })

    # Fallbacks if somehow not saved (should be rare)
    de, ee = kid_def_fallback(word, "en")
    dk, ek = kid_def_fallback(word, "ko")
    return jsonify({
        "ok": True,
        "word": word,
        "definition_en": de, "example_en": ee,
        "definition_ko": dk, "example_ko": ek
    })

import secrets
import string

def generate_class_code(length: int = 6) -> str:
    """Generate a short class code like AB3FQ9."""
    alphabet = string.ascii_uppercase + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))
def current_user_id() -> int | None:
    u = g.get("current_user")
    return u["id"] if u and u.get("id") else None

def is_teacher_for_class(db, classroom_id: int) -> bool:
    uid = current_user_id()
    if not uid:
        return False
    # class owner is teacher
    row = db.execute("SELECT owner_id FROM classrooms WHERE id = ?", (classroom_id,)).fetchone()
    if row and row["owner_id"] == uid:
        return True
    # or explicitly marked as teacher member
    row = db.execute(
        """SELECT 1 FROM classroom_members 
           WHERE classroom_id = ? AND user_id = ? AND role = 'teacher'""",
        (classroom_id, uid),
    ).fetchone()
    return bool(row)
@app.route("/classes", methods=["GET", "POST"])
@login_required
def classes_home():
    """
    Teacher/admin: create a class (POST create_form).
    Student: join a class via code (POST join_form).
    Everyone: see classes they belong to.
    """
    db = get_db()
    current_username = g.current_user["username"]
    is_admin = bool(g.current_user and g.current_user.get("is_admin"))

    # Handle create & join in one place (two separate forms)
    action = request.form.get("action") if request.method == "POST" else None

    # --- Create class (admin only) ---
    if request.method == "POST" and action == "create":
        if not is_admin:
            flash("Only admins can create classes.", "danger")
            return redirect(url_for("classes_home"))

        name = (request.form.get("name") or "").strip()
        if not name:
            flash("Please enter a class name.", "warning")
            return redirect(url_for("classes_home"))

        # Generate unique code
        code = generate_class_code()
        for _ in range(10):
            row = db.execute("SELECT id FROM classes WHERE code = ?", (code,)).fetchone()
            if not row:
                break
            code = generate_class_code()

        db.execute(
            "INSERT INTO classes (name, code, owner_name) VALUES (?, ?, ?)",
            (name, code, current_username),
        )
        class_id = db.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]

        # Also insert creator as teacher
        db.execute(
            "INSERT INTO class_members (class_id, username, role) VALUES (?, ?, ?)",
            (class_id, current_username, "teacher"),
        )
        db.commit()

        flash("Class created.", "success")
        return redirect(url_for("class_detail", class_id=class_id))

    # --- Join class (student or teacher) ---
    if request.method == "POST" and action == "join":
        code = (request.form.get("code") or "").strip().upper()
        if not code:
            flash("Please enter a class code.", "warning")
            return redirect(url_for("classes_home"))

        c = db.execute("SELECT * FROM classes WHERE code = ?", (code,)).fetchone()
        if not c:
            flash("Class not found. Check the code again.", "danger")
            return redirect(url_for("classes_home"))

        # Insert membership if not exists
        existing = db.execute(
            "SELECT id FROM class_members WHERE class_id = ? AND username = ?",
            (c["id"], current_username),
        ).fetchone()
        if existing:
            flash("You are already in this class.", "info")
            return redirect(url_for("class_detail", class_id=c["id"]))

        db.execute(
            "INSERT INTO class_members (class_id, username, role) VALUES (?, ?, ?)",
            (c["id"], current_username, "student"),
        )
        db.commit()
        flash(f"Joined class {c['name']}.", "success")
        return redirect(url_for("class_detail", class_id=c["id"]))

    # --- GET: list classes user belongs to ---
    # As teacher (owner or role=teacher)
    teaching = db.execute(
        """
        SELECT DISTINCT c.*
        FROM classes c
        LEFT JOIN class_members m ON m.class_id = c.id
        WHERE (c.owner_name = ?)
           OR (m.username = ? AND m.role = 'teacher')
        ORDER BY datetime(c.created_at) DESC
        """,
        (current_username, current_username),
    ).fetchall()

    # As student
    learning = db.execute(
        """
        SELECT c.*
        FROM classes c
        JOIN class_members m ON m.class_id = c.id
        WHERE m.username = ?
          AND m.role = 'student'
        ORDER BY datetime(c.created_at) DESC
        """,
        (current_username,),
    ).fetchall()

    return render_template(
        "classes.html",
        teaching=teaching,
        learning=learning,
        is_admin=is_admin,
    )
@app.route("/classrooms/<int:classroom_id>")
@login_required
def classroom_detail(classroom_id):
    db = get_db()

    classroom = db.execute(
        "SELECT * FROM classrooms WHERE id = ?", (classroom_id,)
    ).fetchone()
    if not classroom:
        flash("Classroom not found.", "warning")
        return redirect(url_for("classrooms"))

    uid = current_user_id()

    member = db.execute(
        "SELECT * FROM classroom_members WHERE classroom_id = ? AND user_id = ?",
        (classroom_id, uid),
    ).fetchone()
    if not member:
        flash("You are not a member of this classroom.", "warning")
        return redirect(url_for("classrooms"))

    is_admin = bool(g.current_user and g.current_user.get("is_admin"))
    is_teacher = is_admin or (member["role"] == "teacher")

    # ---------- Assignments ----------
    if is_teacher:
        # Teacher: same as before, no review mark needed
        assignments = db.execute(
            """
            SELECT a.*,
                   s.title AS story_title
            FROM classroom_assignments a
            LEFT JOIN stories s ON s.id = a.story_id
            WHERE a.classroom_id = ?
            ORDER BY datetime(a.created_at) DESC
            """,
            (classroom_id,),
        ).fetchall()
    else:
        # Student: include my submission status + whether teacher has commented
        assignments = db.execute(
            """
            SELECT a.*,
                   s.title AS story_title,
                   sub.status AS my_status,
                   CASE
                     WHEN EXISTS (
                       SELECT 1
                       FROM finish_comments fc
                       WHERE fc.draft_id = sub.draft_id
                     ) THEN 1
                     ELSE 0
                   END AS has_review
            FROM classroom_assignments a
            LEFT JOIN stories s ON s.id = a.story_id
            LEFT JOIN assignment_submissions sub
              ON sub.assignment_id = a.id
             AND sub.user_id = ?
            WHERE a.classroom_id = ?
            ORDER BY datetime(a.created_at) DESC
            """,
            (uid, classroom_id),
        ).fetchall()

    # ---------- Roster ----------
    members = db.execute(
        """
        SELECT u.username, cm.role
        FROM classroom_members cm
        JOIN users u ON u.id = cm.user_id
        WHERE cm.classroom_id = ?
        ORDER BY cm.role DESC, u.username ASC
        """,
        (classroom_id,),
    ).fetchall()

    return render_template(
        "classroom_detail.html",
        classroom=classroom,
        assignments=assignments,
        members=members,
        is_teacher=is_teacher,
    )


@app.route("/classrooms/<int:classroom_id>/assignments/new", methods=["GET", "POST"])
@login_required
def assignment_new(classroom_id):
    db = get_db()
    if not is_teacher_for_class(db, classroom_id):
        flash("Only teachers can create assignments.", "warning")
        return redirect(url_for("classroom_detail", classroom_id=classroom_id))

    room = db.execute("SELECT * FROM classrooms WHERE id = ?", (classroom_id,)).fetchone()
    if not room:
        flash("Classroom not found.", "warning")
        return redirect(url_for("classrooms"))

    if request.method == "POST":
        title = (request.form.get("title") or "").strip()
        description = (request.form.get("description") or "").strip()
        story_id = request.form.get("story_id")
        story_id = int(story_id) if story_id and story_id.isdigit() else None
        due_at = (request.form.get("due_at") or "").strip() or None

        if not title:
            flash("Please enter an assignment title.", "warning")
            return redirect(url_for("assignment_new", classroom_id=classroom_id))

        db.execute(
            """INSERT INTO classroom_assignments (classroom_id, title, description, story_id, due_at)
               VALUES (?, ?, ?, ?, ?)""",
            (classroom_id, title, description, story_id, due_at),
        )
        assignment_id = db.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]
        db.commit()

        flash("Assignment created.", "success")
        return redirect(url_for("assignment_detail", assignment_id=assignment_id))

    # story options for seed (only show recent)
    stories = db.execute(
        "SELECT id, title FROM stories ORDER BY datetime(created_at) DESC LIMIT 50"
    ).fetchall()

    return render_template(
        "assignment_new.html",
        classroom=room,
        stories=stories,
    )
@app.route("/assignments/<int:assignment_id>", methods=["GET", "POST"])
@login_required
def assignment_detail(assignment_id: int):
    db = get_db()
    uid = current_user_id()

    # Load assignment + classroom + optional seed story
    assignment = db.execute(
        """
        SELECT a.*,
               c.id   AS classroom_id,
               c.name AS classroom_name,
               s.content AS story_content
        FROM classroom_assignments a
        JOIN classrooms c ON c.id = a.classroom_id
        LEFT JOIN stories s ON s.id = a.story_id
        WHERE a.id = ?
        """,
        (assignment_id,),
    ).fetchone()
    if not assignment:
        flash("Assignment not found.", "warning")
        return redirect(url_for("classrooms"))

    # Determine role
    is_admin = bool(g.current_user and g.current_user.get("is_admin"))
    is_teacher = is_admin or is_teacher_for_class(db, assignment["classroom_id"])

    # ==================== TEACHER VIEW ====================
    if is_teacher:
        # Roster + submissions + story text
        roster = db.execute(
            """
            SELECT cm.role,
                   u.username,
                   sub.id  AS submission_id,
                   sub.status,
                   sub.draft_id,
                   fd.partial_text,
                   fd.completion_text
            FROM classroom_members cm
            JOIN users u ON u.id = cm.user_id
            LEFT JOIN assignment_submissions sub
              ON sub.assignment_id = ? AND sub.user_id = u.id
            LEFT JOIN finish_drafts fd
              ON fd.id = sub.draft_id
            WHERE cm.classroom_id = ?
            ORDER BY cm.role DESC, u.username ASC
            """,
            (assignment_id, assignment["classroom_id"]),
        ).fetchall()

        # Load all comments for all drafts in this assignment
        draft_ids = [r["draft_id"] for r in roster if r["draft_id"]]
        comments_by_draft: dict[int, list[sqlite3.Row]] = {}
        if draft_ids:
            placeholders = ",".join(["?"] * len(draft_ids))
            rows = db.execute(
                f"""
                SELECT draft_id, author_name, body, created_at
                FROM finish_comments
                WHERE draft_id IN ({placeholders})
                ORDER BY datetime(created_at) ASC
                """,
                draft_ids,
            ).fetchall()
            for row in rows:
                comments_by_draft.setdefault(row["draft_id"], []).append(row)

        return render_template(
            "assignment_detail.html",
            assignment=assignment,
            is_teacher=True,
            roster=roster,
            comments_by_draft=comments_by_draft,
        )

    # ==================== STUDENT VIEW ====================
    # Load this student's submission + draft
    submission = db.execute(
        """
        SELECT sub.*,
               fd.partial_text,
               fd.completion_text,
               fd.id AS draft_id
        FROM assignment_submissions sub
        JOIN finish_drafts fd ON fd.id = sub.draft_id
        WHERE sub.assignment_id = ? AND sub.user_id = ?
        """,
        (assignment_id, uid),
    ).fetchone()

    # Load teacher comments (if any) for this student's draft
    comments = []
    if submission and submission["draft_id"]:
        comments = db.execute(
            """
            SELECT author_name, body, created_at
            FROM finish_comments
            WHERE draft_id = ?
            ORDER BY datetime(created_at) ASC
            """,
            (submission["draft_id"],),
        ).fetchall()

    if request.method == "POST":
        # Only allow saving if a draft exists
        if not submission:
            flash("Please click 'Start writing' first to create your draft.", "warning")
            return redirect(url_for("assignment_detail", assignment_id=assignment_id))

        completion_text = (request.form.get("completion_text") or "").strip()

        # Update finish_drafts
        db.execute(
            "UPDATE finish_drafts SET completion_text = ? WHERE id = ?",
            (completion_text, submission["draft_id"]),
        )

        # Update submission status
        new_status = "submitted" if completion_text else "in_progress"
        db.execute(
            "UPDATE assignment_submissions SET status = ?, updated_at = DATETIME('now') WHERE id = ?",
            (new_status, submission["id"]),
        )

        db.commit()
        flash("Your ending has been saved.", "success")

        # Reload updated submission + comments
        submission = db.execute(
            """
            SELECT sub.*,
                   fd.partial_text,
                   fd.completion_text,
                   fd.id AS draft_id
            FROM assignment_submissions sub
            JOIN finish_drafts fd ON fd.id = sub.draft_id
            WHERE sub.assignment_id = ? AND sub.user_id = ?
            """,
            (assignment_id, uid),
        ).fetchone()
        comments = db.execute(
            """
            SELECT author_name, body, created_at
            FROM finish_comments
            WHERE draft_id = ?
            ORDER BY datetime(created_at) ASC
            """,
            (submission["draft_id"],),
        ).fetchall()

    return render_template(
        "assignment_detail.html",
        assignment=assignment,
        is_teacher=False,
        submission=submission,
        comments=comments,
    )
@app.post("/assignments/<int:assignment_id>/comment/<int:draft_id>")
@login_required
def assignment_comment(assignment_id: int, draft_id: int):
    db = get_db()

    # Check assignment exists
    assignment = db.execute(
        "SELECT id, classroom_id FROM classroom_assignments WHERE id = ?",
        (assignment_id,),
    ).fetchone()
    if not assignment:
        flash("Assignment not found.", "danger")
        return redirect(url_for("classrooms"))

    # Permission: teacher in this class or admin
    is_admin = bool(g.current_user and g.current_user.get("is_admin"))
    if not (is_admin or is_teacher_for_class(db, assignment["classroom_id"])):
        flash("Only teachers can leave comments.", "danger")
        return redirect(url_for("assignment_detail", assignment_id=assignment_id))

    # Verify draft really belongs to this assignment
    belongs = db.execute(
        """
        SELECT 1
        FROM assignment_submissions sub
        WHERE sub.assignment_id = ? AND sub.draft_id = ?
        """,
        (assignment_id, draft_id),
    ).fetchone()
    if not belongs:
        flash("This story is not part of this assignment.", "warning")
        return redirect(url_for("assignment_detail", assignment_id=assignment_id))

    body = (request.form.get("body") or "").strip()
    if not body:
        flash("Comment cannot be empty.", "warning")
        return redirect(url_for("assignment_detail", assignment_id=assignment_id) + f"#story-{draft_id}")

    author = g.current_user.get("username") or "teacher"
    db.execute(
        "INSERT INTO finish_comments (draft_id, author_name, body) VALUES (?, ?, ?)",
        (draft_id, author, body),
    )
    db.commit()
    flash("Comment added.", "success")
    return redirect(url_for("assignment_detail", assignment_id=assignment_id) + f"#story-{draft_id}")

@app.route("/assignments/<int:assignment_id>/write")
@login_required
def assignment_write(assignment_id: int):
    db = get_db()
    uid = current_user_id()

    a = db.execute(
        """
        SELECT a.*,
               s.content      AS story_content,
               s.prompt       AS story_prompt,
               s.language     AS language
        FROM classroom_assignments a
        LEFT JOIN stories s ON s.id = a.story_id
        WHERE a.id = ?
        """,
        (assignment_id,),
    ).fetchone()
    if not a:
        return ("Assignment not found", 404)

    # If the student already has a submission, just go back to assignment page
    sub = db.execute(
        """
        SELECT sub.*, fd.partial_text, fd.completion_text
        FROM assignment_submissions sub
        JOIN finish_drafts fd ON fd.id = sub.draft_id
        WHERE sub.assignment_id = ? AND sub.user_id = ?
        """,
        (assignment_id, uid),
    ).fetchone()
    if sub:
        return redirect(url_for("assignment_detail", assignment_id=assignment_id))

    # otherwise create a new finish_drafts row as seed
    seed_prompt = a.get("description") or a.get("story_prompt") or "Classroom assignment"
    base_text = a.get("story_content") or ""
    partial = make_partial_from_story(base_text) if base_text else ""

    learner_name = g.current_user["username"] if g.current_user and g.current_user.get("username") else "guest"
    language = (a.get("language") or "en")

    db.execute(
        """
        INSERT INTO finish_drafts (seed_prompt, partial_text, learner_name, language)
        VALUES (?, ?, ?, ?)
        """,
        (seed_prompt, partial, learner_name, language),
    )
    draft_id = db.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]

    db.execute(
        """
        INSERT INTO assignment_submissions (assignment_id, user_id, draft_id, status)
        VALUES (?, ?, ?, 'in_progress')
        """,
        (assignment_id, uid, draft_id),
    )
    db.commit()

    # Now return to assignment detail where the inline editor lives
    return redirect(url_for("assignment_detail", assignment_id=assignment_id))

@app.route("/classrooms", methods=["GET", "POST"])
@login_required
def classrooms():
    db = get_db()
    user = g.current_user

    # Only admins can create classrooms (teachers)
    if request.method == "POST":
        if not current_user_is_admin():
            flash("Only teacher/admin accounts can create classrooms.", "warning")
            return redirect(url_for("classrooms"))

        name = (request.form.get("name") or "").strip()
        if not name:
            flash("Please enter a class name.", "warning")
            return redirect(url_for("classrooms"))

        # generate unique code
        while True:
            code = generate_class_code()
            exists = db.execute("SELECT 1 FROM classrooms WHERE code = ?", (code,)).fetchone()
            if not exists:
                break

        db.execute(
            "INSERT INTO classrooms (name, code, owner_id) VALUES (?, ?, ?)",
            (name, code, user["id"]),
        )
        classroom_id = db.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]
        # add teacher as member
        db.execute(
            "INSERT INTO classroom_members (classroom_id, user_id, role) VALUES (?, ?, 'teacher')",
            (classroom_id, user["id"]),
        )
        db.commit()
        flash("Classroom created.", "success")
        return redirect(url_for("classroom_detail", classroom_id=classroom_id))

    uid = current_user_id()
    # classrooms user belongs to
    classes = db.execute(
        """
        SELECT c.*, 
               CASE WHEN c.owner_id = ? THEN 1 ELSE 0 END AS is_owner,
               cm.role AS member_role
        FROM classrooms c
        JOIN classroom_members cm ON cm.classroom_id = c.id
        WHERE cm.user_id = ?
        ORDER BY datetime(c.created_at) DESC
        """,
        (uid, uid),
    ).fetchall()

    return render_template("classrooms.html", classes=classes)

@app.route("/classrooms/join", methods=["GET", "POST"])
@login_required
def classroom_join():
    db = get_db()
    if request.method == "POST":
        code = (request.form.get("code") or "").strip().upper()
        if not code:
            flash("Please enter a class code.", "warning")
            return redirect(url_for("classroom_join"))

        room = db.execute("SELECT * FROM classrooms WHERE upper(code) = ?", (code,)).fetchone()
        if not room:
            flash("No classroom found with that code.", "danger")
            return redirect(url_for("classroom_join"))

        uid = current_user_id()
        try:
            db.execute(
                "INSERT OR IGNORE INTO classroom_members (classroom_id, user_id, role) VALUES (?, ?, 'student')",
                (room["id"], uid),
            )
            db.commit()
        except Exception:
            pass

        flash(f"Joined classroom: {room['name']}", "success")
        return redirect(url_for("classroom_detail", classroom_id=room["id"]))

    return render_template("classroom_join.html")

# ------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
