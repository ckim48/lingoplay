#!/usr/bin/env python3
import argparse
import json
import random
import sqlite3
import string
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

def now_text() -> str:
    # match your DB style: TEXT timestamps are used widely
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table,),
    ).fetchone()
    return row is not None

def cols(conn: sqlite3.Connection, table: str) -> List[str]:
    if not table_exists(conn, table):
        return []
    return [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]

def insert_row(conn: sqlite3.Connection, table: str, data: Dict[str, Any]) -> int:
    """
    Inserts only keys that exist as columns in the table.
    Returns lastrowid.
    """
    table_cols = set(cols(conn, table))
    payload = {k: v for k, v in data.items() if k in table_cols}
    if not payload:
        raise RuntimeError(f"No insertable columns for table '{table}'. Check schema.")
    keys = list(payload.keys())
    q = ",".join(["?"] * len(keys))
    sql = f"INSERT INTO {table} ({','.join(keys)}) VALUES ({q})"
    cur = conn.execute(sql, [payload[k] for k in keys])
    return int(cur.lastrowid)

def pick_existing_id(conn: sqlite3.Connection, table: str) -> Optional[int]:
    if not table_exists(conn, table):
        return None
    row = conn.execute(f"SELECT id FROM {table} ORDER BY id ASC LIMIT 1").fetchone()
    return int(row[0]) if row else None

def rand_slug(n=10) -> str:
    return "mock-" + "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(n))

def ensure_story_and_draft(conn: sqlite3.Connection) -> Tuple[int, int]:
    """
    Make sure we have a story and a finish_draft row.
    Returns (story_id, draft_id).
    """
    story_id = pick_existing_id(conn, "stories")
    draft_id = pick_existing_id(conn, "finish_drafts")

    # If story exists but no draft, create a draft for that story
    if story_id is not None and draft_id is None and table_exists(conn, "finish_drafts"):
        draft_id = insert_row(conn, "finish_drafts", {
            "story_id": story_id,
            "seed_prompt": "Mock seed prompt",
            "partial_text": "Once upon a time, a student began a story...",
            "completion_text": "They finished it happily.",
            "learner_name": "Mock",
            "language": "en",
            "created_at": now_text(),
            "updated_at": now_text(),
        })

    # If no story exists, create story + draft
    if story_id is None and table_exists(conn, "stories"):
        story_id = insert_row(conn, "stories", {
            "slug": rand_slug(),
            "title": "Mock Story for Analytics",
            "language": "en",
            "level": "beginner",
            "content": "This is a short mock story used to seed analytics data.",
            "mcq_questions_json": json.dumps({"questions": []}),
            "author_name": "EduWeaver AI",
            "created_at": now_text(),
            "updated_at": now_text(),
        })

    if draft_id is None and table_exists(conn, "finish_drafts"):
        draft_id = insert_row(conn, "finish_drafts", {
            "story_id": story_id,
            "seed_prompt": "Mock seed prompt",
            "partial_text": "Once upon a time, a student began a story...",
            "completion_text": "They finished it happily.",
            "learner_name": "Mock",
            "language": "en",
            "created_at": now_text(),
            "updated_at": now_text(),
        })

    if story_id is None:
        raise RuntimeError("No 'stories' table found or cannot create story. Please ensure your DB has stories.")
    if draft_id is None:
        # Some DBs might not have finish_drafts; still OK if assignments allow draft_id NULL.
        draft_id = 0

    return story_id, draft_id

def ensure_class_membership(conn: sqlite3.Connection, class_id: int, user_id: int):
    if not table_exists(conn, "class_members"):
        return
    cm_cols = set(cols(conn, "class_members"))
    data = {}
    if "class_id" in cm_cols: data["class_id"] = class_id
    if "user_id" in cm_cols: data["user_id"] = user_id
    if "role" in cm_cols: data["role"] = "student"
    if "joined_at" in cm_cols: data["joined_at"] = now_text()

    # insert-or-ignore if possible
    if "class_id" in cm_cols and "user_id" in cm_cols:
        conn.execute(
            "INSERT OR IGNORE INTO class_members (class_id, user_id, role) VALUES (?, ?, ?)",
            (class_id, user_id, "student"),
        )
    else:
        insert_row(conn, "class_members", data)

def make_mock_users(conn: sqlite3.Connection, n: int) -> List[int]:
    if not table_exists(conn, "users"):
        raise RuntimeError("No 'users' table found.")
    user_cols = set(cols(conn, "users"))

    created_ids = []
    for i in range(n):
        uname = f"mock_student_{int(time.time())}_{i}_{random.randint(1000,9999)}"
        email = f"{uname}@example.com"

        # L1 ~25%, L2 ~75%
        is_native = 1 if (i % 4 == 0) else 0

        exposure_years = round((8 + (i % 5)) if is_native else (0.5 + (i % 10) * 0.8), 1)
        learned_where = "abroad" if i % 6 == 0 else ("school" if i % 6 in (2,3,4,5) else "academy")
        use_freq = "daily" if exposure_years >= 5 else "sometimes"
        self_level = "advanced" if exposure_years >= 7 else ("intermediate" if exposure_years >= 3 else "beginner")

        # You do NOT need real login for analytics. But password_hash is usually NOT NULL.
        # Put any placeholder string.
        udata = {
            "username": uname,
            "email": email,
            "password_hash": "mock_hash",
            "role": "student",
            "l1_language": "english" if is_native else "korean",
            "l2_language": "none" if is_native else "english",
            "age": 12 + (i % 5),
            "gender": "female" if i % 2 == 0 else "male",
            # screening fields (these exist in your register insert)
            "english_exposure_years": exposure_years,
            "english_learned_where": learned_where,
            "english_use_frequency": use_freq,
            "english_self_level": self_level,
            # some DBs store native flag
            "is_english_native": is_native if "is_english_native" in user_cols else None,
            "created_at": now_text(),
            "updated_at": now_text(),
        }

        # Remove None keys (insert_row already filters by columns, but keep clean)
        udata = {k: v for k, v in udata.items() if v is not None}

        uid = insert_row(conn, "users", udata)
        created_ids.append(uid)

        # Optional: level_test_results row helps your analytics page in user view
        if table_exists(conn, "level_test_results"):
            level = "Beginner" if exposure_years < 3 else ("Intermediate" if exposure_years < 7 else "Advanced")
            score = 3 if level == "Beginner" else (6 if level == "Intermediate" else 9)
            insert_row(conn, "level_test_results", {
                "user_id": uid,
                "score": score,
                "total": 10,
                "level": level,
                "created_at": now_text(),
            })

    return created_ids

def make_reading_answers(exposure_years: float, is_native: int) -> str:
    """
    Create answers_json with 6 Q each type: factual/inference/debatable.
    Stored as {"questions":[{"type":"factual","correct":true}, ...]}
    """
    base_acc = 0.78 if is_native else min(0.35 + exposure_years * 0.07, 0.80)
    inf_acc  = min(base_acc - 0.08 + exposure_years * 0.02, 0.85)   # inference harder
    deb_acc  = max(base_acc - 0.15, 0.25)

    def gen(qtype: str, p: float) -> Dict[str, Any]:
        return {"type": qtype, "correct": (random.random() < p)}

    payload = {
        "questions": (
            [gen("factual", base_acc) for _ in range(6)] +
            [gen("inference", inf_acc) for _ in range(6)] +
            [gen("debatable", deb_acc) for _ in range(6)]
        )
    }
    return json.dumps(payload)

def make_main_scores(exposure_years: float) -> Tuple[str, float]:
    """
    MAIN story grammar scores: each dimension 0..2
    Returns (story_grammar_json, total)
    """
    boost = 0 if exposure_years < 2 else (1 if exposure_years < 5 else 2)

    def dim():
        return min(2, random.randint(0, 1) + (1 if boost == 2 else 0 if boost == 0 else random.randint(0,1)))

    scores = {
        "character": dim(),
        "setting": dim(),
        "problem": dim(),
        "actions": min(2, dim() + (1 if boost >= 1 else 0)),
        "resolution": dim(),
    }
    total = float(sum(scores.values()))
    payload = {"scores": scores}
    return json.dumps(payload), total

def seed_assignments_and_submissions(conn: sqlite3.Connection, user_ids: List[int], story_id: int, draft_id: int, class_id: Optional[int]):
    if not table_exists(conn, "assignments"):
        raise RuntimeError("No 'assignments' table found.")
    if not table_exists(conn, "assignment_submissions"):
        raise RuntimeError("No 'assignment_submissions' table found.")

    ucols = set(cols(conn, "users"))

    for uid in user_ids:
        u = conn.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
        if not u:
            continue

        exposure = float(u["english_exposure_years"]) if "english_exposure_years" in u.keys() and u["english_exposure_years"] is not None else 0.0
        is_native = int(u["is_english_native"]) if "is_english_native" in u.keys() and u["is_english_native"] is not None else 0

        # --- Reading assignment ---
        reading_aid = insert_row(conn, "assignments", {
            "story_id": story_id,
            "draft_id": (draft_id if draft_id != 0 else None),
            "assignee_id": uid,
            "assignment_type": "reading",
            "questions_json": json.dumps({"questions": []}),
            "assigned_by": None,
            "assignment_title": "Mock Reading Worksheet",
            "class_id": class_id,
            "status": "assigned",
            "created_at": now_text(),
            "updated_at": now_text(),
        })

        answers_json = make_reading_answers(exposure, is_native)
        insert_row(conn, "assignment_submissions", {
            "assignment_id": reading_aid,
            "user_id": uid,
            "story_id": story_id,
            "draft_id": (draft_id if draft_id != 0 else None),
            "answers_json": answers_json,
            "created_at": now_text(),
            "updated_at": now_text(),
        })

        # --- Writing assignment ---
        writing_aid = insert_row(conn, "assignments", {
            "story_id": story_id,
            "draft_id": (draft_id if draft_id != 0 else None),
            "assignee_id": uid,
            "assignment_type": "writing",
            "questions_json": json.dumps({"type": "writing_main", "version": 1}),
            "assigned_by": None,
            "assignment_title": "Mock Writing MAIN",
            "class_id": class_id,
            "status": "assigned",
            "created_at": now_text(),
            "updated_at": now_text(),
        })

        grammar_json, grammar_total = make_main_scores(exposure)
        insert_row(conn, "assignment_submissions", {
            "assignment_id": writing_aid,
            "user_id": uid,
            "story_id": story_id,
            "draft_id": (draft_id if draft_id != 0 else None),
            "completion_text": "This is a mock writing submission.",
            "story_grammar_json": grammar_json,
            "story_grammar_total": grammar_total,
            "created_at": now_text(),
            "updated_at": now_text(),
        })

def main():
    ap = argparse.ArgumentParser(description="Seed mock analytics data into LexiTale SQLite DB (no Flask).")
    ap.add_argument("--db", required=True, help="Path to your SQLite DB file (e.g., ./lexitale.db)")
    ap.add_argument("--n", type=int, default=18, help="Number of mock students to create (default 18)")
    ap.add_argument("--class-id", type=int, default=None, help="Optional class_id to enroll mock students into")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    # basic sanity checks
    for t in ["users", "assignments", "assignment_submissions"]:
        if not table_exists(conn, t):
            raise SystemExit(f"Missing required table: {t}")

    story_id, draft_id = ensure_story_and_draft(conn)

    user_ids = make_mock_users(conn, args.n)

    if args.class_id is not None:
        for uid in user_ids:
            ensure_class_membership(conn, args.class_id, uid)

    seed_assignments_and_submissions(conn, user_ids, story_id, draft_id, args.class_id)

    conn.commit()
    conn.close()

    print("✅ Done.")
    print(f"  Inserted mock students: {len(user_ids)}")
    print(f"  Story used: story_id={story_id}, draft_id={draft_id if draft_id != 0 else 'NULL/0'}")
    if args.class_id is not None:
        print(f"  Enrolled into class_id={args.class_id}")

if __name__ == "__main__":
    main()
