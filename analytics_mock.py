import sqlite3
import os
from pathlib import Path

# Define the path where the database file will be created
# This matches the logic used in your Flask application.
INSTANCE_DIR = Path("instance")
DB_NAME = "lingoplay.db"
DB_PATH = INSTANCE_DIR / DB_NAME

# SQL schema definitions for all tables derived from your app.py logic and schema snippets.
SQL_SCHEMA = """
PRAGMA foreign_keys = ON;

-- -----------------------------------------------------------
-- CORE TABLES (Users, Stories, Drafts)
-- -----------------------------------------------------------

CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    l1_language TEXT,
    l2_language TEXT,
    age INTEGER,
    gender TEXT,
    is_admin INTEGER DEFAULT 0,
    is_english_native INTEGER, -- 0 or 1
    created_at TEXT DEFAULT (DATETIME('now'))
);

CREATE TABLE IF NOT EXISTS stories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    slug TEXT NOT NULL UNIQUE,
    prompt TEXT,
    language TEXT NOT NULL,
    level TEXT,
    content TEXT,
    visuals TEXT, -- Base64 data URL for cover image
    author_name TEXT,
    is_shared_library INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (DATETIME('now')),
    -- Migrated column
    mcq_questions_json TEXT
);

CREATE TABLE IF NOT EXISTS finish_drafts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    story_id INTEGER NOT NULL,
    seed_prompt TEXT NOT NULL,
    partial_text TEXT NOT NULL,
    learner_name TEXT,
    language TEXT,
    completion_text TEXT,
    created_at TEXT DEFAULT (DATETIME('now')),
    FOREIGN KEY (story_id) REFERENCES stories(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS vocab_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    story_id INTEGER NOT NULL,
    word TEXT NOT NULL,
    definition TEXT,
    example TEXT,
    definition_ko TEXT,
    example_ko TEXT,
    picture_url TEXT,
    created_at TEXT DEFAULT (DATETIME('now')),
    FOREIGN KEY (story_id) REFERENCES stories(id) ON DELETE CASCADE
);

-- -----------------------------------------------------------
-- ASSIGNMENT / REVIEW TABLES (Standardized)
-- -----------------------------------------------------------

CREATE TABLE IF NOT EXISTS assignments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    story_id INTEGER NOT NULL,
    draft_id INTEGER,
    assignee_id INTEGER NOT NULL,
    assignment_type TEXT NOT NULL, -- 'finish', 'mcq', 'writing', 'reading'
    status TEXT DEFAULT 'assigned', -- assigned / submitted / graded
    score REAL,                     -- Latest score (used by student list/profile)
    attempt_count INTEGER DEFAULT 0,
    assignment_title TEXT,          -- Added via migration logic
    questions_json TEXT,            -- Added via migration logic (for generic worksheets)
    created_at TEXT DEFAULT (DATETIME('now')),
    assigned_by INTEGER,
    FOREIGN KEY (story_id) REFERENCES stories(id) ON DELETE CASCADE,
    FOREIGN KEY (draft_id) REFERENCES finish_drafts(id) ON DELETE SET NULL,
    FOREIGN KEY (assignee_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (assigned_by) REFERENCES users(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS assignment_submissions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    assignment_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    story_id INTEGER NOT NULL,
    draft_id INTEGER,
    completion_text TEXT,   -- For 'finish' type
    answers_json TEXT,      -- For 'mcq' or 'reading' answers
    score REAL,             -- Score specific to this submission attempt
    comment TEXT,           -- Admin/Reviewer comment (migrated)
    reviewed_at TEXT,       -- Timestamp when reviewed (migrated)
    created_at TEXT DEFAULT (DATETIME('now')),
    updated_at TEXT DEFAULT (DATETIME('now')),
    FOREIGN KEY (assignment_id) REFERENCES assignments(id) ON DELETE CASCADE,
    FOREIGN KEY (story_id) REFERENCES stories(id) ON DELETE CASCADE,
    FOREIGN KEY (draft_id) REFERENCES finish_drafts(id) ON DELETE SET NULL,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);


-- -----------------------------------------------------------
-- LEARNING DATA TABLES
-- -----------------------------------------------------------

CREATE TABLE IF NOT EXISTS level_test_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    score INTEGER NOT NULL,
    total INTEGER NOT NULL,
    level TEXT NOT NULL,
    created_at TEXT DEFAULT (DATETIME('now')),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS scramble_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    word TEXT NOT NULL,
    is_correct INTEGER NOT NULL, -- 0 or 1
    created_at TEXT DEFAULT (DATETIME('now')),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS scramble_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    correct_count INTEGER NOT NULL,
    wrong_count INTEGER NOT NULL,
    total INTEGER NOT NULL,
    created_at TEXT DEFAULT (DATETIME('now')),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- -----------------------------------------------------------
-- DICTIONARY TABLES
-- -----------------------------------------------------------

CREATE TABLE IF NOT EXISTS dict_lookups (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    source_lang TEXT NOT NULL,
    target_lang TEXT NOT NULL,
    query TEXT NOT NULL,
    headword TEXT,
    translation TEXT,
    example_en TEXT,
    example_ko TEXT,
    direction TEXT DEFAULT 'en_ko',
    result_json TEXT, -- Keeping this from snippet for completeness
    created_at TEXT DEFAULT (DATETIME('now')),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS dict_bookmarks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    lookup_id INTEGER, -- Added via migration logic in app.py
    source_lang TEXT,
    target_lang TEXT,
    word TEXT,
    translation TEXT,
    note TEXT,
    created_at TEXT DEFAULT (DATETIME('now')),
    UNIQUE(user_id, source_lang, target_lang, word),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (lookup_id) REFERENCES dict_lookups(id) ON DELETE CASCADE
);

-- -----------------------------------------------------------
-- LOGGING TABLES
-- -----------------------------------------------------------

CREATE TABLE IF NOT EXISTS input_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    action TEXT NOT NULL,
    payload_json TEXT,
    created_at TEXT DEFAULT (DATETIME('now'))
);
"""


def create_database():
    """
    Creates the necessary directory and runs the consolidated SQL schema
    to initialize the SQLite database file.
    """
    print(f"Ensuring instance directory exists: {INSTANCE_DIR}")
    INSTANCE_DIR.mkdir(exist_ok=True)

    # Check if DB already exists
    if DB_PATH.exists():
        print(f"Database file already exists at: {DB_PATH}. Deleting and recreating.")
        os.remove(DB_PATH)

    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        print("Running schema creation script...")
        cursor.executescript(SQL_SCHEMA)
        conn.commit()

        print("\n✅ Database and all tables created successfully!")
        print(f"File path: {DB_PATH.resolve()}")

    except sqlite3.Error as e:
        print(f"\n❌ An error occurred during database creation: {e}")

    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    create_database()