#!/usr/bin/env python3
"""
One-time migration script for LexiTale DB.

- Ensures `users` table has:
    l1_language TEXT
    l2_language TEXT
    age INTEGER
    gender TEXT

- Ensures `level_test_results` table exists, and has:
    level_name  TEXT
    level_score INTEGER

Run:
    python migrate_lexitale_db.py
"""

import sqlite3
import os
import sys

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# CHANGE THIS to the actual path of your DB file
# e.g. "lingoplay.db" or "lexitale.db"
DB_PATH = "instance/lingoplay.db"
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


def add_column_if_missing(conn, table, column, col_type):
    """Add a column to a table if it doesn't already exist."""
    cur = conn.execute(f"PRAGMA table_info({table})")
    cols = [row[1] for row in cur.fetchall()]  # row[1] = column name
    if column in cols:
        print(f"[info] {table}.{column} already exists, skipping.")
        return

    sql = f"ALTER TABLE {table} ADD COLUMN {column} {col_type}"
    print(f"[migrate] {sql}")
    conn.execute(sql)


def ensure_level_test_table(conn):
    """
    Create level_test_results table if it does not exist,
    then add missing columns.
    """
    # Create table if it doesn't exist at all
    sql_create = """
    CREATE TABLE IF NOT EXISTS level_test_results (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id    INTEGER NOT NULL,
        created_at TEXT DEFAULT (datetime('now')),
        FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
    );
    """
    print("[migrate] CREATE TABLE IF NOT EXISTS level_test_results")
    conn.execute(sql_create)

    # Make sure columns exist
    add_column_if_missing(conn, "level_test_results", "level_name", "TEXT")
    add_column_if_missing(conn, "level_test_results", "level_score", "INTEGER")


def main():
    if not os.path.exists(DB_PATH):
        print(f"[error] DB file not found at: {DB_PATH}")
        print("Please update DB_PATH in migrate_lexitale_db.py.")
        sys.exit(1)

    print(f"[info] Using DB: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")

    try:
        # --- add new profile columns on users ---
        add_column_if_missing(conn, "users", "l1_language", "TEXT")
        add_column_if_missing(conn, "users", "l2_language", "TEXT")
        add_column_if_missing(conn, "users", "age", "INTEGER")
        add_column_if_missing(conn, "users", "gender", "TEXT")

        # --- ensure level_test_results has needed columns ---
        ensure_level_test_table(conn)

        conn.commit()
        print("[success] Migration completed.")
    except Exception as e:
        conn.rollback()
        print("[error] Migration failed:", e)
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
