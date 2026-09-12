"""
Persistent research memory for DataBroker.

This is the "Knowledge Layer" + "Personal Research Memory" (spec sections 13-14).
SQLite is used for the MVP; the access pattern is isolated in this module so it
can be swapped for Postgres later without touching agent logic.
"""

import sqlite3
import json
import datetime
from contextlib import contextmanager
from pathlib import Path

DEFAULT_DB_PATH = Path.home() / ".databroker" / "databroker.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS companies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    notes TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS watchlist (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id INTEGER NOT NULL REFERENCES companies(id),
    reason TEXT,
    added_at TEXT NOT NULL,
    last_swept_at TEXT
);

CREATE TABLE IF NOT EXISTS thesis (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id INTEGER NOT NULL REFERENCES companies(id),
    summary TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS thesis_points (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thesis_id INTEGER NOT NULL REFERENCES thesis(id),
    point_text TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'unassessed',  -- supported/weakened/contradicted/new_risk/new_opportunity/unassessed
    last_note TEXT,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT,
    source_type TEXT,        -- filing, regulatory, investor_relations, primary_statement, news, industry_pub, community, social
    publication_date TEXT,
    retrieved_at TEXT NOT NULL,
    reliability_tier INTEGER -- 1 (highest, e.g. filings) .. 8 (lowest, e.g. social posts), per spec section 11
);

CREATE TABLE IF NOT EXISTS claims (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id INTEGER NOT NULL REFERENCES companies(id),
    text TEXT NOT NULL,
    source_id INTEGER REFERENCES sources(id),
    confidence TEXT,          -- low/medium/high
    verification_status TEXT DEFAULT 'unverified',  -- unverified/single_source/cross_checked/primary_confirmed/conflicting
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS conflicts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_a_id INTEGER NOT NULL REFERENCES claims(id),
    claim_b_id INTEGER NOT NULL REFERENCES claims(id),
    description TEXT,
    resolved INTEGER DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id INTEGER NOT NULL REFERENCES companies(id),
    title TEXT NOT NULL,
    description TEXT,
    importance_level TEXT NOT NULL,  -- low/medium/high/critical
    importance_rationale TEXT,
    thesis_impact TEXT,              -- supports/weakens/contradicts/new_risk/new_opportunity/neutral
    confidence TEXT,
    status TEXT DEFAULT 'new',       -- new/updated/duplicate
    parent_event_id INTEGER REFERENCES events(id),  -- for change detection: links updates to the original
    source_ids TEXT,                 -- JSON list of source ids
    notified INTEGER DEFAULT 0,
    first_seen_at TEXT NOT NULL,
    last_updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS research_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id INTEGER REFERENCES companies(id),
    objective TEXT NOT NULL,
    plan_json TEXT,
    result_summary TEXT,
    created_at TEXT NOT NULL
);
"""


def now() -> str:
    return datetime.datetime.utcnow().isoformat()


class DB:
    def __init__(self, path: Path = DEFAULT_DB_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.conn.commit()

    def _migrate(self):
        """Idempotent, additive-only migrations for DBs created before a schema change."""
        cols = {row["name"] for row in self.conn.execute("PRAGMA table_info(watchlist)")}
        if "last_swept_at" not in cols:
            self.conn.execute("ALTER TABLE watchlist ADD COLUMN last_swept_at TEXT")

    @contextmanager
    def cursor(self):
        cur = self.conn.cursor()
        try:
            yield cur
            self.conn.commit()
        finally:
            cur.close()

    # ---- companies / watchlist ----
    def add_company(self, ticker: str, name: str, notes: str = "") -> int:
        with self.cursor() as cur:
            cur.execute(
                "INSERT INTO companies (ticker, name, notes, created_at) VALUES (?,?,?,?) "
                "ON CONFLICT(ticker) DO UPDATE SET name=excluded.name",
                (ticker.upper(), name, notes, now()),
            )
            cur.execute("SELECT id FROM companies WHERE ticker=?", (ticker.upper(),))
            return cur.fetchone()["id"]

    def get_company(self, ticker: str):
        with self.cursor() as cur:
            cur.execute("SELECT * FROM companies WHERE ticker=?", (ticker.upper(),))
            return cur.fetchone()

    def watch(self, company_id: int, reason: str = ""):
        with self.cursor() as cur:
            cur.execute(
                "INSERT INTO watchlist (company_id, reason, added_at) VALUES (?,?,?)",
                (company_id, reason, now()),
            )

    def list_watchlist(self):
        with self.cursor() as cur:
            cur.execute(
                "SELECT c.*, w.id as watchlist_id, w.reason, w.added_at, w.last_swept_at "
                "FROM watchlist w JOIN companies c ON c.id = w.company_id ORDER BY w.added_at"
            )
            return cur.fetchall()

    def mark_swept(self, watchlist_id: int):
        with self.cursor() as cur:
            cur.execute("UPDATE watchlist SET last_swept_at=? WHERE id=?", (now(), watchlist_id))

    # ---- thesis ----
    def set_thesis(self, company_id: int, summary: str, points: list[str]) -> int:
        with self.cursor() as cur:
            cur.execute(
                "INSERT INTO thesis (company_id, summary, created_at, updated_at) VALUES (?,?,?,?)",
                (company_id, summary, now(), now()),
            )
            thesis_id = cur.lastrowid
            for p in points:
                cur.execute(
                    "INSERT INTO thesis_points (thesis_id, point_text, updated_at) VALUES (?,?,?)",
                    (thesis_id, p, now()),
                )
            return thesis_id

    def get_latest_thesis(self, company_id: int):
        with self.cursor() as cur:
            cur.execute(
                "SELECT * FROM thesis WHERE company_id=? ORDER BY created_at DESC LIMIT 1",
                (company_id,),
            )
            thesis = cur.fetchone()
            if not thesis:
                return None, []
            cur.execute(
                "SELECT * FROM thesis_points WHERE thesis_id=? ORDER BY id", (thesis["id"],)
            )
            return thesis, cur.fetchall()

    def update_thesis_point(self, point_id: int, status: str, note: str):
        with self.cursor() as cur:
            cur.execute(
                "UPDATE thesis_points SET status=?, last_note=?, updated_at=? WHERE id=?",
                (status, note, now(), point_id),
            )

    # ---- sources / claims ----
    def add_source(self, url, source_type, publication_date, reliability_tier) -> int:
        with self.cursor() as cur:
            cur.execute(
                "INSERT INTO sources (url, source_type, publication_date, retrieved_at, reliability_tier) "
                "VALUES (?,?,?,?,?)",
                (url, source_type, publication_date, now(), reliability_tier),
            )
            return cur.lastrowid

    def add_claim(self, company_id, text, source_id, confidence, verification_status="unverified") -> int:
        with self.cursor() as cur:
            cur.execute(
                "INSERT INTO claims (company_id, text, source_id, confidence, verification_status, created_at) "
                "VALUES (?,?,?,?,?,?)",
                (company_id, text, source_id, confidence, verification_status, now()),
            )
            return cur.lastrowid

    def update_claim_verification(self, claim_id: int, status: str):
        with self.cursor() as cur:
            cur.execute("UPDATE claims SET verification_status=? WHERE id=?", (status, claim_id))

    def add_conflict(self, claim_a_id, claim_b_id, description) -> int:
        with self.cursor() as cur:
            cur.execute(
                "INSERT INTO conflicts (claim_a_id, claim_b_id, description, created_at) VALUES (?,?,?,?)",
                (claim_a_id, claim_b_id, description, now()),
            )
            return cur.lastrowid

    # ---- events (change detection + importance) ----
    def recent_events(self, company_id: int, days: int = 60):
        cutoff = (datetime.datetime.utcnow() - datetime.timedelta(days=days)).isoformat()
        with self.cursor() as cur:
            cur.execute(
                "SELECT * FROM events WHERE company_id=? AND last_updated_at >= ? ORDER BY last_updated_at DESC",
                (company_id, cutoff),
            )
            return cur.fetchall()

    def add_event(
        self, company_id, title, description, importance_level, importance_rationale,
        thesis_impact, confidence, source_ids, status="new", parent_event_id=None,
    ) -> int:
        with self.cursor() as cur:
            cur.execute(
                """INSERT INTO events
                (company_id, title, description, importance_level, importance_rationale,
                 thesis_impact, confidence, status, parent_event_id, source_ids,
                 first_seen_at, last_updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (company_id, title, description, importance_level, importance_rationale,
                 thesis_impact, confidence, status, parent_event_id, json.dumps(source_ids),
                 now(), now()),
            )
            return cur.lastrowid

    def mark_event_updated(self, event_id: int, new_description: str):
        with self.cursor() as cur:
            cur.execute(
                "UPDATE events SET description=?, status='updated', last_updated_at=? WHERE id=?",
                (new_description, now(), event_id),
            )

    def unnotified_events(self, company_id: int, min_level="high"):
        levels = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        threshold = levels[min_level]
        with self.cursor() as cur:
            cur.execute("SELECT * FROM events WHERE company_id=? AND notified=0", (company_id,))
            return [r for r in cur.fetchall() if levels.get(r["importance_level"], 0) >= threshold]

    def mark_notified(self, event_id: int):
        with self.cursor() as cur:
            cur.execute("UPDATE events SET notified=1 WHERE id=?", (event_id,))

    # ---- research sessions ----
    def log_session(self, company_id, objective, plan, result_summary) -> int:
        with self.cursor() as cur:
            cur.execute(
                "INSERT INTO research_sessions (company_id, objective, plan_json, result_summary, created_at) "
                "VALUES (?,?,?,?,?)",
                (company_id, objective, json.dumps(plan), result_summary, now()),
            )
            return cur.lastrowid

    def past_sessions(self, company_id: int, limit=10):
        with self.cursor() as cur:
            cur.execute(
                "SELECT * FROM research_sessions WHERE company_id=? ORDER BY created_at DESC LIMIT ?",
                (company_id, limit),
            )
            return cur.fetchall()
