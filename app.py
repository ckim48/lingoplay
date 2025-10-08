import os
import io
import re
import csv
import json
import sqlite3
from datetime import datetime
from slugify import slugify
from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, send_file, g, session
)
from werkzeug.security import generate_password_hash, check_password_hash

# --- OpenAI SDK (env-based; do NOT hardcode secrets) ---
from openai import OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL = os.getenv("LINGOPLAY_MODEL", "gpt-4.1-mini")

# ------------------------------------------------------------------------------------
# App setup
# ------------------------------------------------------------------------------------
app = Flask(__name__, instance_relative_config=True)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "dev-secret")

# Ensure instance folder exists (DB goes here)
os.makedirs(app.instance_path, exist_ok=True)
DB_PATH = os.path.join(app.instance_path, "lingoplay.db")

# ------------------------------------------------------------------------------------
# SQLite helpers
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
def load_json_filter(s):
    try:
        return json.loads(s) if s else {}
    except Exception:
        return {}

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
            choices_json TEXT NOT NULL,
            correct_index INTEGER NOT NULL,
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
            FOREIGN KEY (story_id) REFERENCES stories(id) ON DELETE CASCADE
        );
        """
    )
    db.commit()

# Create DB on first run
with app.app_context():
    init_db()

# ------------------------------------------------------------------------------------
# Auth helpers
# ------------------------------------------------------------------------------------
def get_user_by_identifier(identifier: str):
    """Allow login with either email or username (case-insensitive)."""
    db = get_db()
    ident = (identifier or "").strip().lower()
    # try email first
    row = db.execute(
        "SELECT * FROM users WHERE lower(email)=?",
        (ident,)
    ).fetchone()
    if row:
        return row
    # then username
    row = db.execute(
        "SELECT * FROM users WHERE lower(username)=?",
        (ident,)
    ).fetchone()
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
# Generators
# ------------------------------------------------------------------------------------
def naive_story_from_prompt(prompt: str, language: str = "en") -> str:
    if language == "ko":
        return (f"{prompt} 단어를 활용한 이야기를 길게 써 볼게요. 주인공은 쉬운 말과 짧한 문장으로 생각을 나누고, "
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

def llm_story_from_prompt(prompt: str, language: str, level: str, author: str) -> str:
    if client is None:
        return naive_story_from_prompt(prompt, language)

    level_note = {
        "phonics": "Use very short sentences where possible; repeat key phonics and vocabulary naturally.",
        "early-reader": "Simple sentences (8–12 words); concrete, decodable vocabulary.",
        "custom": "Neutral elementary reading level unless the prompt implies otherwise."
    }.get(level, "Use simple sentences.")

    lang_note = (
        "Write entirely in Korean." if language == "ko"
        else "Write each sentence in English, then its Korean translation on the next line." if language == "en-ko"
        else "Write entirely in English."
    )

    system = (
        "You are a children's story generator for phonics & early readers. "
        "IMPORTANT: Do not use labels like [Beginning], [Middle], or [End]. "
        "Write a single continuous story without section headings or extra commentary."
    )

    user = (
        f"Author/Learner: {author or 'guest'}\n"
        f"Target words/phonics: {prompt}\n"
        f"Level: {level}\n"
        f"{level_note}\n"
        f"{lang_note}\n"
        "Length: 180–260 words (or equivalent in Korean/en-ko). "
        "Keep sentences short and decodable; use gentle repetition; keep a warm tone; end with a hopeful note."
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

    text = getattr(resp, "output_text", None)
    if not text:
        try:
            parts = []
            for item in resp.output:
                if hasattr(item, "content"):
                    for c in item.content:
                        if c["type"] == "output_text":
                            parts.append(c["text"])
            text = "".join(parts).strip()
        except Exception:
            text = None
    return text or naive_story_from_prompt(prompt, language)
def generate_quiz_via_gpt(story_text: str, language: str = "en", level: str = "phonics") -> list[dict]:
    TARGET_QS = 10

    if client is None:
        # Simple fallback: 10 generic Qs
        base = [
            {"question": "What did the hero practice?", "choices": ["Math", "New sounds", "Running", "Cooking"], "correct_index": 1},
            {"question": "Who supported the hero?", "choices": ["Friends", "A doctor", "A pilot", "A chef"], "correct_index": 0},
            {"question": "Where did reading happen?", "choices": ["At a park", "In a cave", "On a boat", "In a plane"], "correct_index": 0},
            {"question": "What is the main idea?", "choices": ["Cooking", "Reading together", "Racing", "Sleeping"], "correct_index": 1},
            {"question": "What happens near the start?", "choices": ["They sleep", "They share words", "They cook", "They race"], "correct_index": 1},
        ]
        # pad to 10
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
        "- 5 vocabulary-in-context (pick common or decodable words from the story);\n"
        "- 3 detail questions (who/what/where/when);\n"
        "- 1 sequence/order question;\n"
        "- 1 main idea question.\n"
        "Rules:\n"
        "- 4 answer choices per question; only one correct.\n"
        "- Keep wording short and unambiguous; no trick answers; child-friendly.\n"
        "- Reference only information in the story.\n"
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

    # Enforce exactly 10
    while len(out) < TARGET_QS:
        out.append({
            "question": "Which choice best matches the story?",
            "choices": ["A", "B", "C", "D"],
            "correct_index": 0
        })
    return out[:TARGET_QS]


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

def generate_kid_definitions(words: list[str], language: str="en") -> list[dict]:
    out = []
    if client is None or not words:
        for w in words:
            d, e = kid_def_fallback(w, language)
            out.append({"word": w, "definition": d, "example": e})
        return out

    lang_note = "Write in English." if language != "ko" else "Write in Korean."
    prompt = (
        f"{lang_note} For each WORD, return a one-sentence kid-friendly DEFINITION "
        f"(≤16 words) and a short EXAMPLE sentence (≤12 words). "
        "Return STRICT JSON only: {\"items\":[{\"word\":\"\",\"definition\":\"\",\"example\":\"\"},...]}\n"
        "WORDS:\n" + "\n".join(f"- {w}" for w in words)
    )
    try:
        resp = client.responses.create(
            model=DEFAULT_MODEL,
            input=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_output_tokens=600,
        )
        raw = (getattr(resp, "output_text", "") or "").strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I|re.M)
        data = json.loads(raw)
        for item in data.get("items", []):
            w = str(item.get("word", "")).lower().strip()
            if not w:
                continue
            d = str(item.get("definition", "")).strip()
            e = str(item.get("example", "")).strip()
            if not d or not e:
                d, e = kid_def_fallback(w, language)
            out.append({"word": w, "definition": d, "example": e})
    except Exception:
        for w in words:
            d, e = kid_def_fallback(w, language)
            out.append({"word": w, "definition": d, "example": e})
    return out

def make_partial_from_story(full_text: str) -> str:
    sentences = re.split(r'(?<=[.!?。！？])\s+', full_text.strip())
    if len(sentences) < 4:
        keep = sentences[:max(1, int(len(sentences)*0.75))]
    else:
        keep = sentences[:max(3, int(len(sentences)*0.8))]
    partial = " ".join(keep).strip()
    if partial and not partial.endswith(('.', '!', '?', '。', '！', '？')):
        partial += "."
    partial += "\n\nYour Turn: Write the ending."
    return partial

def log_input(action: str, payload: dict):
    db = get_db()
    db.execute(
        "INSERT INTO input_logs (action, payload_json) VALUES (?, ?)",
        (action, json.dumps(payload)),
    )
    db.commit()

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

        # Basic validation
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
                "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                (username, email, pwd_hash)
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
    if request.method == "POST":
        username = (request.form.get("username") or "").strip().lower()
        password = request.form.get("password") or ""

        if not username or not password:
            flash("Please enter your username and password.", "warning")
            return redirect(url_for("login"))

        db = get_db()
        user = db.execute(
            "SELECT * FROM users WHERE lower(username)=?",
            (username,)
        ).fetchone()

        if not user or not check_password_hash(user["password_hash"], password):
            flash("Invalid username or password.", "danger")
            return redirect(url_for("login"))

        session["user_id"] = user["id"]
        flash(f"Welcome back, {user['username']}!", "success")
        return redirect(url_for("index"))

    return render_template("login.html")


@app.post("/logout")
def logout():
    session.pop("user_id", None)
    flash("You have been signed out.", "info")
    return redirect(url_for("index"))
@app.route("/story/new", methods=["GET", "POST"])
def story_new():
    if request.method == "POST":
        db = get_db()

        # Core fields
        title = (request.form.get("title") or "").strip() or "My Story"
        prompt = (request.form.get("prompt") or "").strip()
        language = request.form.get("language", "en")
        level = request.form.get("level", "phonics")

        # Prefer the signed-in username as the author if present
        author = (g.current_user["username"] if (g.current_user and g.current_user.get("username")) else None) \
                 or (request.form.get("author_name") or "guest")

        want_image = bool(request.form.get("gen_image"))

        # New optional controls
        theme = (request.form.get("theme") or "").strip()
        characters = (request.form.get("characters") or "").strip()
        tone = (request.form.get("tone") or "").strip()
        bme = bool(request.form.get("bme"))  # Beginning–Middle–End toggle

        if not prompt:
            flash("Please provide phonics letters or vocabulary.", "warning")
            return redirect(url_for("story_new"))

        # Build a style/meta directive for the generator (keeps vocab parsing clean)
        meta_bits = []
        if theme:
            meta_bits.append(f"Theme: {theme}")
        if characters:
            meta_bits.append(f"Main characters: {characters} (keep names consistent).")
        if tone:
            meta_bits.append(f"Tone: {tone}")
        if bme:
            meta_bits.append("Follow a clear Beginning–Middle–End arc.")
        # Enrich the generator prompt without changing the user-entered phonics/vocab
        gen_prompt = prompt if not meta_bits else (prompt + "\n\n" + " ".join(meta_bits))

        # ----- Generate story text -----
        try:
            content = llm_story_from_prompt(gen_prompt, language, level, author)
        except Exception as e:
            content = naive_story_from_prompt(prompt, language)
            flash("AI generator had an issue; used a fallback story.", "warning")
            log_input("generate_story_error", {"error": str(e)})

        # ----- Optional cover image -----
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

        # ----- Save story -----
        slug = f"{slugify(title)}-{int(datetime.utcnow().timestamp())}"
        db.execute(
            """INSERT INTO stories (title, slug, prompt, language, level, content, author_name, visuals)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (title, slug, prompt, language, level, content, author, visuals_data_url),
        )
        db.commit()
        story_row = db.execute("SELECT id FROM stories WHERE slug = ?", (slug,)).fetchone()
        story_id = story_row["id"]

        # ----- Vocab + kid definitions (based on original prompt only) -----
        vocab = parse_vocab_from_prompt(prompt)
        defs = generate_kid_definitions(vocab, language=language)
        for item in defs:
            db.execute(
                """INSERT INTO vocab_items (story_id, word, definition, example, picture_url)
                   VALUES (?, ?, ?, ?, NULL)""",
                (story_id, item["word"], item["definition"], item["example"])
            )
        db.commit()

        log_input("generate_story", {
            "prompt": prompt,
            "language": language,
            "level": level,
            "author": author,
            "model": DEFAULT_MODEL,
            "vocab_count": len(defs),
            "with_image": want_image,
            "theme": theme,
            "characters": characters,
            "tone": tone,
            "bme": bme,
        })

        # ----- Create a Finish Draft tied to this story -----
        partial = make_partial_from_story(content)
        db.execute(
            """INSERT INTO finish_drafts (seed_prompt, partial_text, learner_name, language)
               VALUES (?, ?, ?, ?)""",
            (prompt, partial, author, language),
        )
        draft_id = db.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]
        db.commit()

        log_input("finish_seed_auto_from_story_new", {
            "story_id": story_id, "slug": slug, "draft_id": draft_id
        })

        # ----- Redirect back here so the success modal pops, with a finish_url to that draft -----
        # The template you updated will auto-show the modal when ?generated=1.
        # The "Finish the Story" button will use this finish_url.
        return redirect(url_for("story_new", generated=1,
                                finish_url=url_for('finish_view', draft_id=draft_id)))

    # GET
    return render_template("story_new.html")




@app.get("/story/<slug>")
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

@app.post("/story/<slug>/build-worksheet")
def build_worksheet(slug):
    db = get_db()
    s = db.execute("SELECT * FROM stories WHERE slug = ?", (slug,)).fetchone()
    if not s:
        return ("Story not found", 404)

    db.execute("DELETE FROM quiz_questions WHERE story_id = ?", (s["id"],))
    for q, choices, correct in simple_questions_from_story(s["content"]):
        db.execute(
            """INSERT INTO quiz_questions (story_id, question, choices_json, correct_index)
               VALUES (?, ?, ?, ?)""",
            (s["id"], q, json.dumps(choices), correct),
        )
    db.commit()
    log_input("build_worksheet", {"story_id": s["id"], "slug": s["slug"]})
    flash("Worksheet & quiz generated (simple).", "success")
    return redirect(url_for("quiz_take", slug=slug))

@app.post("/story/<slug>/build-worksheet-ai")
def build_worksheet_ai(slug):
    db = get_db()
    s = db.execute("SELECT * FROM stories WHERE slug = ?", (slug,)).fetchone()
    if not s:
        return ("Story not found", 404)

    questions = generate_quiz_via_gpt(s["content"], language=s["language"], level=s["level"])
    db.execute("DELETE FROM quiz_questions WHERE story_id = ?", (s["id"],))
    for item in questions:
        db.execute(
            """INSERT INTO quiz_questions (story_id, question, choices_json, correct_index)
               VALUES (?, ?, ?, ?)""",
            (s["id"], item["question"], json.dumps(item["choices"]), int(item["correct_index"]))
        )
    db.commit()

    log_input("build_worksheet_ai", {
        "story_id": s["id"], "slug": s["slug"], "qcount": len(questions), "model": DEFAULT_MODEL
    })
    flash("Worksheet & quiz generated by AI.", "success")
    return redirect(url_for("quiz_take", slug=slug))
@app.route("/quiz/<slug>", methods=["GET", "POST"])
def quiz_take(slug):
    db = get_db()

    # Load the story
    s = db.execute(
        "SELECT * FROM stories WHERE slug = ?",
        (slug,)
    ).fetchone()
    if not s:
        return ("Story not found", 404)

    # 🔗 NEW: find a matching finish_draft so we can link to /finish/<id>
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

    # Pull questions
    qs = db.execute(
        "SELECT * FROM quiz_questions WHERE story_id = ? ORDER BY id ASC",
        (s["id"],)
    ).fetchall()

    if not qs:
        flash("No quiz yet. Build the worksheet first.", "warning")
        return redirect(url_for("story_view", slug=slug))

    # POST: grade/save attempt
    if request.method == "POST":
        taker = request.form.get(
            "taker_name",
            g.current_user["username"] if (g.current_user and g.current_user.get("username")) else "guest"
        )
        total = len(qs)
        score = 0
        details = []
        chosen_map = {}

        for q in qs:
            ans = request.form.get(f"q{q['id']}")
            try:
                chosen = int(ans)
            except Exception:
                chosen = -1
            correct = (chosen == q["correct_index"])
            if correct:
                score += 1
            details.append({
                "qid": q["id"],
                "chosen": chosen,
                "correct": q["correct_index"]
            })
            chosen_map[q["id"]] = chosen

        # Save attempt
        db.execute(
            """INSERT INTO quiz_attempts (story_id, taker_name, score, total, detail_json)
               VALUES (?, ?, ?, ?, ?)""",
            (s["id"], taker, score, total, json.dumps(details)),
        )
        db.commit()

        pct = round((score / total) * 100) if total else 0
        log_input("take_quiz", {
            "story_id": s["id"], "taker": taker, "score": score, "total": total, "percent": pct
        })

        # Build normalized question list
        q_for_view = []
        for q in qs:
            choices = q["choices_json"]
            if isinstance(choices, str):
                try:
                    choices = json.loads(choices) if choices else []
                except Exception:
                    choices = []
            q_for_view.append({
                "id": q["id"],
                "question": q["question"],
                "choices": choices,
                "correct_index": q["correct_index"],
                "chosen_index": chosen_map.get(q["id"], -1)
            })

        # Explanations
        explanations = explain_answers_via_gpt(
            story_text=s["content"],
            questions=[{"question": x["question"], "choices": x["choices"], "correct_index": x["correct_index"]} for x in q_for_view]
        )

        # 👉 Pass draft_id through so results page can also link back to Finish
        return render_template(
            "quiz_result.html",
            story=s,
            taker=taker,
            score=score,
            total=total,
            percent=pct,
            questions=q_for_view,
            explanations=explanations,
            draft_id=draft_id
        )

    # GET: render the quiz
    return render_template("quiz_take.html", story=s, questions=qs, draft_id=draft_id)


def explain_answers_via_gpt(story_text: str, questions: list[dict]) -> list[str]:
    """
    Returns a list of short explanations for each question index.
    Falls back to simple generic lines if the model fails.
    """
    # Fallback first (in case client is None)
    fallback = [
        "This answer matches a clear detail stated in the story.",
        "The story describes this event directly in that order.",
        "Vocabulary meaning fits how the word is used in context.",
        "This is the main idea repeated across the story.",
        "Sequence is supported by the order of events."
    ]

    if client is None or not questions:
        return (fallback * ((len(questions) // len(fallback)) + 1))[:len(questions)]

    # Pack questions into JSON for the model
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
    user = (
        "STORY:\n"
        + story_text
        + "\n\nQUESTIONS:\n"
        + json.dumps(qpack, ensure_ascii=False)
        + "\n\nGive one simple reason per question in order."
    )

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
def story_to_finish(slug):
    db = get_db()
    s = db.execute("SELECT * FROM stories WHERE slug = ?", (slug,)).fetchone()
    if not s:
        return ("Story not found", 404)

    # Require login
    if not (g.current_user and g.current_user.get("username")):
        flash("Please sign in to use Finish the Story.", "warning")
        return redirect(url_for("login"))

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
@app.route("/finish", methods=["GET", "POST"])
def finish_new():
    db = get_db()

    if request.method == "POST":
        seed = (request.form.get("seed_prompt") or "").strip()
        language = request.form.get("language", "en")

        # Prefer the logged-in username if available; otherwise use the form value or "guest"
        if g.current_user and g.current_user.get("username"):
            learner = g.current_user["username"]
        else:
            learner = (request.form.get("learner_name") or "guest").strip() or "guest"

        if not seed:
            flash("Please add a short seed (phonics/vocab).", "warning")
            return redirect(url_for("finish_new"))

        try:
            full = llm_story_from_prompt(seed, language, "phonics", learner)
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

    # -------- GET: list unfinished drafts for the current user ----------
    if g.current_user and g.current_user.get("username"):
        learner = g.current_user["username"]
    else:
        learner = "guest"

    # Pull unfinished drafts + derive a matching story title (if any) via subquery
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
@app.route("/finish/<int:draft_id>", methods=["GET", "POST"])
def finish_view(draft_id):
    db = get_db()
    d = db.execute("SELECT * FROM finish_drafts WHERE id = ?", (draft_id,)).fetchone()
    if not d:
        return ("Draft not found", 404)

    # Who's logged in?
    current_username = g.current_user["username"] if (g.current_user and g.current_user.get("username")) else None
    is_owner = (current_username is not None and d["learner_name"] == current_username)

    # Handle save (only author can modify)
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

    # Link back to the original story (so we can build/take a quiz)
    story_row = db.execute(
        """
        SELECT id, title, slug
        FROM stories
        WHERE prompt = ?
          AND author_name = ?
          AND language = ?
        ORDER BY datetime(created_at) DESC
        LIMIT 1
        """,
        (d["seed_prompt"], d["learner_name"], d["language"])
    ).fetchone()

    d["story_title"] = story_row["title"] if story_row else None
    d["story_slug"]  = story_row["slug"] if story_row else None
    d["story_id"]    = story_row["id"] if story_row else None

    # Does a quiz already exist?
    has_quiz = False
    if story_row:
        qcount = db.execute(
            "SELECT COUNT(*) AS cnt FROM quiz_questions WHERE story_id = ?",
            (story_row["id"],)
        ).fetchone()["cnt"]
        has_quiz = (qcount > 0)

    # Viewing rules
    if (not is_owner) and (not d["completion_text"] or not d["completion_text"].strip()):
        flash("This story isn’t finished yet. Check back later!", "info")
        return redirect(url_for("library"))

    return render_template(
        "finish_view.html",
        draft=d,
        can_edit=is_owner,
        has_quiz=has_quiz
    )

@app.get("/library")
def library():
    db = get_db()
    # Completed drafts + derived story title (if we can find a matching story)
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
def dashboard():
    db = get_db()
    learner = (g.current_user.get("username") if g.current_user and g.current_user.get("username") else "guest")

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

    # NEW: stories by this author that already have quizzes
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
@app.template_filter("load_json")
def load_json_filter(s):
    try:
        val = json.loads(s) if s else []
        # always return a list for choices_json use
        return val if isinstance(val, list) else []
    except Exception:
        return []

@app.post("/finish/<int:draft_id>/build-worksheet-ai")
def finish_build_worksheet_ai(draft_id):
    db = get_db()

    # 1) Load the draft
    d = db.execute(
        "SELECT * FROM finish_drafts WHERE id = ?",
        (draft_id,)
    ).fetchone()
    if not d:
        flash("Draft not found.", "danger")
        return redirect(url_for("finish_new"))

    # 2) Find the matching story (same prompt/author/lang)
    story = db.execute(
        """
        SELECT id, title, slug, content, language, level
        FROM stories
        WHERE prompt = ?
          AND author_name = ?
          AND language = ?
        ORDER BY datetime(created_at) DESC
        LIMIT 1
        """,
        (d["seed_prompt"], d["learner_name"], d["language"])
    ).fetchone()

    if not story:
        flash(
            "We couldn’t find the original story to build a worksheet from. "
            "If you created this via “New Story,” the worksheet/quiz is available on that story’s page.",
            "warning"
        )
        return redirect(url_for("finish_view", draft_id=draft_id))

    # 3) Build quiz with GPT and save questions
    try:
        questions = generate_quiz_via_gpt(
            story_text=story["content"],
            language=story["language"],
            level=story["level"]
        )
    except Exception as e:
        log_input("build_worksheet_ai_error", {"draft_id": draft_id, "error": str(e)})
        flash("AI had an issue generating the quiz. Please try again.", "danger")
        return redirect(url_for("finish_view", draft_id=draft_id))

    db.execute("DELETE FROM quiz_questions WHERE story_id = ?", (story["id"],))
    for item in questions:
        db.execute(
            """INSERT INTO quiz_questions (story_id, question, choices_json, correct_index)
               VALUES (?, ?, ?, ?)""",
            (story["id"], item["question"], json.dumps(item["choices"]), int(item["correct_index"]))
        )
    db.commit()

    log_input("build_worksheet_ai_from_finish", {
        "draft_id": draft_id, "story_id": story["id"], "slug": story["slug"], "qcount": len(questions)
    })
    flash("Worksheet & quiz generated by AI.", "success")
    return redirect(url_for("quiz_take", slug=story["slug"]))

# ------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
