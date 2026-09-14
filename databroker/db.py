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
from .graph import normalize_entity_name, normalize_predicate

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

-- Phase 5: knowledge graph. A lightweight property graph on top of SQLite —
-- entities (companies, people, products, regulators, etc.) and directed
-- relationships between them, each traceable back to the claim it came from.
CREATE TABLE IF NOT EXISTS entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,                    -- canonical display name (first form seen)
    normalized_name TEXT NOT NULL UNIQUE,  -- dedup key, see graph.py
    entity_type TEXT,                      -- company | person | product | regulator | other
    company_id INTEGER REFERENCES companies(id),  -- set if this entity IS a tracked company
    first_seen_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS relationships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_entity_id INTEGER NOT NULL REFERENCES entities(id),
    predicate TEXT NOT NULL,               -- e.g. acquired, partnered_with, invested_in, competitor_of
    object_entity_id INTEGER NOT NULL REFERENCES entities(id),
    claim_id INTEGER REFERENCES claims(id),  -- provenance
    confidence TEXT,
    created_at TEXT NOT NULL
);
"""


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


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
        cutoff = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days)).isoformat()
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

    # ---- knowledge graph (Phase 5) ----
    def upsert_entity(self, name: str, entity_type: str | None = None) -> int:
        """Returns the entity id, creating it if a matching normalized_name
        doesn't already exist. If this name matches a tracked company's name,
        links the entity to that company_id — that's what makes "NVIDIA" as
        mentioned inside some other company's claim resolve back to your
        actual watchlist entry for NVDA, enabling cross-company queries."""
        norm = normalize_entity_name(name)
        with self.cursor() as cur:
            cur.execute("SELECT id, company_id FROM entities WHERE normalized_name=?", (norm,))
            existing = cur.fetchone()
            if existing:
                if existing["company_id"] is None:
                    # A company may have been added to `companies` after this
                    # entity was first seen — check again on every upsert.
                    company_id = self._match_company_id(cur, norm)
                    if company_id is not None:
                        cur.execute("UPDATE entities SET company_id=? WHERE id=?", (company_id, existing["id"]))
                return existing["id"]

            company_id = self._match_company_id(cur, norm)
            cur.execute(
                "INSERT INTO entities (name, normalized_name, entity_type, company_id, first_seen_at) "
                "VALUES (?,?,?,?,?)",
                (name, norm, entity_type, company_id, now()),
            )
            return cur.lastrowid

    @staticmethod
    def _match_company_id(cur, normalized_entity_name: str):
        cur.execute("SELECT id, name FROM companies")
        for row in cur.fetchall():
            if normalize_entity_name(row["name"]) == normalized_entity_name:
                return row["id"]
        return None

    def add_relationship(self, subject_entity_id: int, predicate: str, object_entity_id: int,
                          claim_id: int | None, confidence: str | None = None) -> int:
        with self.cursor() as cur:
            cur.execute(
                "INSERT INTO relationships (subject_entity_id, predicate, object_entity_id, "
                "claim_id, confidence, created_at) VALUES (?,?,?,?,?,?)",
                (subject_entity_id, normalize_predicate(predicate), object_entity_id, claim_id, confidence, now()),
            )
            return cur.lastrowid

    def get_relationships_for_entity(self, entity_id: int):
        """All relationships where this entity is either the subject or the object."""
        with self.cursor() as cur:
            cur.execute(
                """SELECT r.*, s.name AS subject_name, o.name AS object_name
                   FROM relationships r
                   JOIN entities s ON s.id = r.subject_entity_id
                   JOIN entities o ON o.id = r.object_entity_id
                   WHERE r.subject_entity_id=? OR r.object_entity_id=?
                   ORDER BY r.created_at DESC""",
                (entity_id, entity_id),
            )
            return cur.fetchall()

    def get_entity_by_company(self, company_id: int):
        with self.cursor() as cur:
            cur.execute("SELECT * FROM entities WHERE company_id=?", (company_id,))
            return cur.fetchone()

    def find_cross_watchlist_connections(self):
        """Relationships where BOTH sides are entities linked to a company on
        the watchlist — e.g. 'NVIDIA acquired Hugging Face' surfacing here if
        both NVIDIA and Hugging Face happen to be tracked companies. This is
        the concrete payoff of the graph: a connection between two portfolio
        holdings that would otherwise be buried in two separate reports."""
        with self.cursor() as cur:
            cur.execute(
                """SELECT r.*, s.name AS subject_name, o.name AS object_name,
                          sc.ticker AS subject_ticker, oc.ticker AS object_ticker
                   FROM relationships r
                   JOIN entities s ON s.id = r.subject_entity_id
                   JOIN entities o ON o.id = r.object_entity_id
                   JOIN companies sc ON sc.id = s.company_id
                   JOIN companies oc ON oc.id = o.company_id
                   JOIN watchlist ws ON ws.company_id = sc.id
                   JOIN watchlist wo ON wo.company_id = oc.id
                   WHERE s.company_id IS NOT NULL AND o.company_id IS NOT NULL
                     AND s.company_id != o.company_id
                   ORDER BY r.created_at DESC"""
            )
            return cur.fetchall()
