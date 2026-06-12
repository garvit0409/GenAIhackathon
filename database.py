import sqlite3
import pandas as pd
from config import INTEL_DB


# ══════════════════════════════════════════
# SCHEMA INIT & MIGRATION
# ══════════════════════════════════════════
def init_db():
    conn = sqlite3.connect(INTEL_DB)
    c = conn.cursor()

    c.execute("""CREATE TABLE IF NOT EXISTS meetings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        ingested_at TEXT,
        raw_content TEXT,
        source_type TEXT
    )""")

    c.execute("""CREATE TABLE IF NOT EXISTS projects (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        meeting_id INTEGER,
        name TEXT,
        status TEXT,
        description TEXT,
        FOREIGN KEY(meeting_id) REFERENCES meetings(id)
    )""")

    c.execute("""CREATE TABLE IF NOT EXISTS action_items (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        meeting_id INTEGER,
        task TEXT,
        owner TEXT,
        deadline TEXT,
        priority TEXT,
        status TEXT DEFAULT 'Pending',
        FOREIGN KEY(meeting_id) REFERENCES meetings(id)
    )""")

    # ── ENHANCED escalations schema ──────────────────────────────────────────
    # duplicate_of      → id of the canonical (first-seen) escalation, NULL if original
    # occurrence_count  → how many times this exact issue has recurred across meetings
    # similarity_score  → float 0-1, how similar it was to the matched canonical
    # detection_method  → 'exact'|'fuzzy'|'semantic' — which algo caught it
    # ─────────────────────────────────────────────────────────────────────────
    c.execute("""CREATE TABLE IF NOT EXISTS escalations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        meeting_id INTEGER,
        issue TEXT,
        raised_by TEXT,
        assigned_to TEXT,
        severity TEXT,
        status TEXT DEFAULT 'Open',
        duplicate_of INTEGER DEFAULT NULL,
        occurrence_count INTEGER DEFAULT 1,
        similarity_score REAL DEFAULT NULL,
        detection_method TEXT DEFAULT NULL,
        FOREIGN KEY(meeting_id) REFERENCES meetings(id),
        FOREIGN KEY(duplicate_of) REFERENCES escalations(id)
    )""")

    # Migration: add new columns to existing DBs created without them
    for col, definition in [
        ("duplicate_of",     "INTEGER DEFAULT NULL"),
        ("occurrence_count", "INTEGER DEFAULT 1"),
        ("similarity_score", "REAL DEFAULT NULL"),
        ("detection_method", "TEXT DEFAULT NULL"),
    ]:
        try:
            c.execute(f"ALTER TABLE escalations ADD COLUMN {col} {definition}")
        except sqlite3.OperationalError:
            pass  # column already exists

    c.execute("""CREATE TABLE IF NOT EXISTS risks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        meeting_id INTEGER,
        description TEXT,
        impact TEXT,
        teams_involved TEXT,
        severity TEXT,
        FOREIGN KEY(meeting_id) REFERENCES meetings(id)
    )""")

    c.execute("""CREATE TABLE IF NOT EXISTS decisions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        meeting_id INTEGER,
        decision TEXT,
        rationale TEXT,
        decision_maker TEXT,
        FOREIGN KEY(meeting_id) REFERENCES meetings(id)
    )""")

    c.execute("""CREATE TABLE IF NOT EXISTS stakeholders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        meeting_id INTEGER,
        name TEXT,
        role TEXT,
        responsibility TEXT,
        FOREIGN KEY(meeting_id) REFERENCES meetings(id)
    )""")

    conn.commit()
    conn.close()


# ══════════════════════════════════════════
# QUERY HELPERS
# ══════════════════════════════════════════
def get_dashboard_stats() -> dict:
    conn  = sqlite3.connect(INTEL_DB)
    stats = {}
    for table in ["meetings", "action_items", "escalations", "risks", "decisions", "projects"]:
        stats[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    stats["open_escalations"]      = conn.execute("SELECT COUNT(*) FROM escalations WHERE status='Open'").fetchone()[0]
    stats["pending_tasks"]         = conn.execute("SELECT COUNT(*) FROM action_items WHERE status='Pending'").fetchone()[0]
    stats["critical_risks"]        = conn.execute("SELECT COUNT(*) FROM risks WHERE severity='Critical'").fetchone()[0]
    stats["recurring_escalations"] = conn.execute(
        "SELECT COUNT(*) FROM escalations WHERE duplicate_of IS NULL AND occurrence_count > 1"
    ).fetchone()[0]
    conn.close()
    return stats


def get_table_df(table: str, limit: int = 100) -> pd.DataFrame:
    conn = sqlite3.connect(INTEL_DB)
    try:
        query_map = {
            "action_items": """SELECT a.id, m.title as Meeting, a.task as Task, a.owner as Owner,
                               a.deadline as Deadline, a.priority as Priority, a.status as Status
                               FROM action_items a JOIN meetings m ON a.meeting_id=m.id
                               ORDER BY a.id DESC LIMIT ?""",
            "escalations":  """SELECT e.id, m.title as Meeting, e.issue as Issue, e.raised_by as 'Raised By',
                               e.assigned_to as 'Assigned To', e.severity as Severity, e.status as Status,
                               e.occurrence_count as 'Times Raised', e.duplicate_of as 'Duplicate Of',
                               e.similarity_score as 'Similarity', e.detection_method as 'Detected Via'
                               FROM escalations e JOIN meetings m ON e.meeting_id=m.id
                               ORDER BY e.id DESC LIMIT ?""",
            "risks":        """SELECT r.id, m.title as Meeting, r.description as Risk,
                               r.impact as Impact, r.teams_involved as Teams, r.severity as Severity
                               FROM risks r JOIN meetings m ON r.meeting_id=m.id
                               ORDER BY r.id DESC LIMIT ?""",
            "decisions":    """SELECT d.id, m.title as Meeting, d.decision as Decision,
                               d.rationale as Rationale, d.decision_maker as 'Decision Maker'
                               FROM decisions d JOIN meetings m ON d.meeting_id=m.id
                               ORDER BY d.id DESC LIMIT ?""",
            "projects":     """SELECT p.id, m.title as Meeting, p.name as Project,
                               p.status as Status, p.description as Description
                               FROM projects p JOIN meetings m ON p.meeting_id=m.id
                               ORDER BY p.id DESC LIMIT ?""",
            "stakeholders": """SELECT s.id, m.title as Meeting, s.name as Name,
                               s.role as Role, s.responsibility as Responsibility
                               FROM stakeholders s JOIN meetings m ON s.meeting_id=m.id
                               ORDER BY s.id DESC LIMIT ?""",
        }
        q = query_map.get(table)
        if q:
            df = pd.read_sql(q, conn, params=(limit,))
        else:
            df = pd.read_sql(f"SELECT * FROM {table} ORDER BY id DESC LIMIT ?", conn, params=(limit,))
        conn.close()
        return df
    except Exception:
        conn.close()
        return pd.DataFrame()


def update_status(table: str, row_id: int, new_status: str):
    conn = sqlite3.connect(INTEL_DB)
    conn.execute(f"UPDATE {table} SET status=? WHERE id=?", (new_status, row_id))
    conn.commit()
    conn.close()