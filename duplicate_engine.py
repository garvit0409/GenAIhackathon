import re
import sqlite3

import difflib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import INTEL_DB, FUZZY_THRESHOLD, TFIDF_THRESHOLD


# ══════════════════════════════════════════
# INTERNAL HELPERS
# ══════════════════════════════════════════

def _normalize(text: str) -> str:
    """Lowercase, strip old append-notes, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r'\[re-raised in meeting:.*?\]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def _tfidf_similarity(text_a: str, text_b: str) -> float:
    """Return cosine similarity between two texts using TF-IDF."""
    try:
        vec   = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        tfidf = vec.fit_transform([text_a, text_b])
        return float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
    except Exception:
        return 0.0


# ══════════════════════════════════════════
# THREE-PASS DUPLICATE FINDER
# ══════════════════════════════════════════

def find_duplicate_escalation(
    issue_text: str,
    conn: sqlite3.Connection,
    exclude_id: int = None,
) -> dict | None:
    """
    Three-pass duplicate detection:
      Pass 1 — Exact / substring containment  → method='exact'
      Pass 2 — Fuzzy string ratio (difflib)   → method='fuzzy'
      Pass 3 — TF-IDF cosine similarity       → method='semantic'

    Returns a dict {id, issue, similarity_score, detection_method, occurrence_count}
    for the best canonical match, or None if no duplicate found.
    Only compares against *canonical* originals (duplicate_of IS NULL) that are Open.
    """
    c = conn.cursor()
    query = "SELECT id, issue, occurrence_count FROM escalations WHERE status='Open' AND duplicate_of IS NULL"
    if exclude_id:
        query += f" AND id != {int(exclude_id)}"
    c.execute(query)
    open_originals = c.fetchall()

    if not open_originals:
        return None

    norm_new   = _normalize(issue_text)
    best_match = None
    best_score = 0.0

    for esc_id, existing_issue, occ_count in open_originals:
        norm_existing = _normalize(existing_issue)

        # ── Pass 1: exact / substring ────────────────────────────────────
        if norm_new == norm_existing:
            return {
                "id": esc_id, "issue": existing_issue,
                "similarity_score": 1.0, "detection_method": "exact",
                "occurrence_count": occ_count,
            }

        if norm_new in norm_existing or norm_existing in norm_new:
            score = len(min(norm_new, norm_existing, key=len)) / len(max(norm_new, norm_existing, key=len))
            if score > best_score:
                best_score = score
                best_match = {
                    "id": esc_id, "issue": existing_issue,
                    "similarity_score": round(score, 3), "detection_method": "exact",
                    "occurrence_count": occ_count,
                }

        # ── Pass 2: fuzzy string ratio ───────────────────────────────────
        fuzzy_score = difflib.SequenceMatcher(None, norm_new, norm_existing).ratio()
        if fuzzy_score >= FUZZY_THRESHOLD and fuzzy_score > best_score:
            best_score = fuzzy_score
            best_match = {
                "id": esc_id, "issue": existing_issue,
                "similarity_score": round(fuzzy_score, 3), "detection_method": "fuzzy",
                "occurrence_count": occ_count,
            }

        # ── Pass 3: semantic TF-IDF ──────────────────────────────────────
        tfidf_score = _tfidf_similarity(norm_new, norm_existing)
        if tfidf_score >= TFIDF_THRESHOLD and tfidf_score > best_score:
            best_score = tfidf_score
            best_match = {
                "id": esc_id, "issue": existing_issue,
                "similarity_score": round(tfidf_score, 3), "detection_method": "semantic",
                "occurrence_count": occ_count,
            }

    return best_match


# ══════════════════════════════════════════
# CLUSTER QUERIES
# ══════════════════════════════════════════

def get_escalation_recurrence_timeline(canonical_id: int) -> pd.DataFrame:
    """Return all occurrences (original + duplicates) of an escalation cluster."""
    conn = sqlite3.connect(INTEL_DB)
    df = pd.read_sql("""
        SELECT e.id, m.title AS Meeting, m.ingested_at AS Date,
               e.issue AS Issue, e.raised_by AS 'Raised By',
               e.severity AS Severity, e.status AS Status,
               e.duplicate_of AS 'Duplicate Of',
               e.similarity_score AS Similarity,
               e.detection_method AS 'Detected Via',
               e.occurrence_count AS 'Occurrence #'
        FROM escalations e
        JOIN meetings m ON e.meeting_id = m.id
        WHERE e.id = ? OR e.duplicate_of = ?
        ORDER BY m.ingested_at ASC
    """, conn, params=(canonical_id, canonical_id))
    conn.close()
    return df


def get_duplicate_clusters() -> list[dict]:
    """
    Return all escalation clusters that have at least one duplicate.
    Each element: {canonical: Series, duplicates: DataFrame}
    """
    conn = sqlite3.connect(INTEL_DB)

    df_canonical = pd.read_sql("""
        SELECT e.id, e.issue, e.raised_by, e.severity, e.status,
               e.occurrence_count, m.title AS meeting, m.ingested_at AS first_seen
        FROM escalations e
        JOIN meetings m ON e.meeting_id = m.id
        WHERE e.duplicate_of IS NULL
          AND e.occurrence_count > 1
        ORDER BY e.occurrence_count DESC, e.id DESC
    """, conn)

    clusters = []
    for _, canon in df_canonical.iterrows():
        df_dupes = pd.read_sql("""
            SELECT e.id, e.issue, e.raised_by, e.severity, e.status,
                   e.similarity_score, e.detection_method,
                   m.title AS meeting, m.ingested_at AS seen_at
            FROM escalations e
            JOIN meetings m ON e.meeting_id = m.id
            WHERE e.duplicate_of = ?
            ORDER BY m.ingested_at ASC
        """, conn, params=(int(canon["id"]),))
        clusters.append({"canonical": canon, "duplicates": df_dupes})

    conn.close()
    return clusters