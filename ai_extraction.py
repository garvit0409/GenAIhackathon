import re
import json
import sqlite3
from datetime import datetime

import pandas as pd
import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.documents import Document

from config import INTEL_DB
from resources import init_resources, init_vector_store
from duplicate_engine import find_duplicate_escalation


# ══════════════════════════════════════════
# AI EXTRACTION ENGINE
# ══════════════════════════════════════════

EXTRACTION_PROMPT = """You are an expert organizational intelligence analyst. Extract ALL structured information from the meeting content below.

Return a single valid JSON object with EXACTLY these keys (use empty lists/strings if nothing found):

{{
  "meeting_title": "string - infer a descriptive title",
  "projects": [
    {{"name": "string", "status": "string", "description": "string"}}
  ],
  "action_items": [
    {{"task": "string", "owner": "string", "deadline": "string or empty", "priority": "High|Medium|Low"}}
  ],
  "escalations": [
    {{"issue": "string", "raised_by": "string", "assigned_to": "string or empty", "severity": "Critical|High|Medium|Low"}}
  ],
  "risks": [
    {{"description": "string", "impact": "string", "teams_involved": "string", "severity": "Critical|High|Medium|Low"}}
  ],
  "decisions": [
    {{"decision": "string", "rationale": "string", "decision_maker": "string or empty"}}
  ],
  "stakeholders": [
    {{"name": "string", "role": "string", "responsibility": "string"}}
  ]
}}

MEETING CONTENT:
{content}

Return ONLY the JSON. No markdown, no explanation, no extra text."""


def extract_intelligence(content: str) -> dict:
    _, llm, _ = init_resources()
    prompt = EXTRACTION_PROMPT.format(content=content)
    try:
        response = llm.invoke([SystemMessage(content=prompt)]).content.strip()
        response = re.sub(r"```json|```", "", response).strip()
        return json.loads(response)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
        return {
            "meeting_title": "Untitled Meeting",
            "projects": [], "action_items": [],
            "escalations": [], "risks": [], "decisions": [], "stakeholders": [],
        }
    except Exception as e:
        st.error(f"Extraction error: {e}")
        return {}


# ══════════════════════════════════════════
# INTELLIGENCE STORAGE (with dup detection)
# ══════════════════════════════════════════

def store_intelligence(intel: dict, raw_content: str, source_type: str) -> tuple[int, list]:
    """
    Persist extracted intelligence to SQLite + ChromaDB.
    Returns (meeting_id, dup_report).
    dup_report is a list of dicts describing any duplicate escalations found.
    """
    vector_store = init_vector_store()
    conn = sqlite3.connect(INTEL_DB)
    c    = conn.cursor()

    title = intel.get("meeting_title", "Untitled Meeting")
    now   = datetime.now().strftime("%Y-%m-%d %H:%M")

    c.execute(
        "INSERT INTO meetings (title, ingested_at, raw_content, source_type) VALUES (?,?,?,?)",
        (title, now, raw_content, source_type),
    )
    meeting_id = c.lastrowid

    for p in intel.get("projects", []):
        c.execute(
            "INSERT INTO projects (meeting_id, name, status, description) VALUES (?,?,?,?)",
            (meeting_id, p.get("name",""), p.get("status",""), p.get("description","")),
        )

    for a in intel.get("action_items", []):
        c.execute(
            "INSERT INTO action_items (meeting_id, task, owner, deadline, priority) VALUES (?,?,?,?,?)",
            (meeting_id, a.get("task",""), a.get("owner",""), a.get("deadline",""), a.get("priority","Medium")),
        )

    # ── Duplicate Escalation Detection (Three-pass engine) ─────────────────
    dup_report = []

    for e in intel.get("escalations", []):
        issue_desc   = e.get("issue", "")
        new_severity = e.get("severity", "Medium")

        match = find_duplicate_escalation(issue_desc, conn)

        if match:
            canonical_id = match["id"]
            score        = match["similarity_score"]
            method       = match["detection_method"]

            c.execute("""
                INSERT INTO escalations
                  (meeting_id, issue, raised_by, assigned_to, severity,
                   duplicate_of, similarity_score, detection_method)
                VALUES (?,?,?,?,?,?,?,?)
            """, (meeting_id, issue_desc, e.get("raised_by",""), e.get("assigned_to",""),
                  new_severity, canonical_id, score, method))

            c.execute("UPDATE escalations SET occurrence_count = occurrence_count + 1 WHERE id = ?",
                      (canonical_id,))

            severity_rank = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
            current_sev = c.execute("SELECT severity FROM escalations WHERE id=?", (canonical_id,)).fetchone()[0]
            if severity_rank.get(new_severity, 0) > severity_rank.get(current_sev, 0):
                c.execute("UPDATE escalations SET severity=? WHERE id=?", (new_severity, canonical_id))

            dup_report.append({
                "issue":           issue_desc,
                "canonical_id":    canonical_id,
                "canonical_issue": match["issue"],
                "similarity":      score,
                "method":          method,
                "occurrences":     match["occurrence_count"] + 1,
            })

        else:
            c.execute("""
                INSERT INTO escalations
                  (meeting_id, issue, raised_by, assigned_to, severity,
                   occurrence_count, duplicate_of)
                VALUES (?,?,?,?,?,1,NULL)
            """, (meeting_id, issue_desc, e.get("raised_by",""), e.get("assigned_to",""), new_severity))

    for r in intel.get("risks", []):
        c.execute(
            "INSERT INTO risks (meeting_id, description, impact, teams_involved, severity) VALUES (?,?,?,?,?)",
            (meeting_id, r.get("description",""), r.get("impact",""),
             r.get("teams_involved",""), r.get("severity","Medium")),
        )

    for d in intel.get("decisions", []):
        c.execute(
            "INSERT INTO decisions (meeting_id, decision, rationale, decision_maker) VALUES (?,?,?,?)",
            (meeting_id, d.get("decision",""), d.get("rationale",""), d.get("decision_maker","")),
        )

    for s in intel.get("stakeholders", []):
        c.execute(
            "INSERT INTO stakeholders (meeting_id, name, role, responsibility) VALUES (?,?,?,?)",
            (meeting_id, s.get("name",""), s.get("role",""), s.get("responsibility","")),
        )

    conn.commit()
    conn.close()

    doc_text = f"Meeting: {title}\n\n{raw_content}"
    vector_store.add_documents([Document(
        page_content=doc_text,
        metadata={"meeting_id": str(meeting_id), "title": title, "date": now},
    )])

    return meeting_id, dup_report


# ══════════════════════════════════════════
# NL QUERY ENGINE
# ══════════════════════════════════════════

NL_QUERY_PROMPT = """You are an expert meeting intelligence analyst with access to a structured database. Base your entire answer ONLY on the provided DATABASE CONTENTS and SEMANTIC SEARCH RESULTS.
   - Do not use outside knowledge or extrapolate facts.
   - If the requested information cannot be found, say so clearly. Do not hallucinate.

DATABASE CONTENTS (current snapshot):
{db_context}

SEMANTIC SEARCH RESULTS (related meeting transcripts):
{vector_context}

Answer the user's question accurately and concisely using the data above.
- Be specific: name people, projects, deadlines
- Structure multi-item answers as clean lists
- If nothing relevant found, say so clearly
- Never hallucinate data

USER QUESTION: {question}"""


def build_db_context() -> str:
    conn = sqlite3.connect(INTEL_DB)
    ctx  = []

    df_meetings = pd.read_sql(
        "SELECT id, title, ingested_at FROM meetings ORDER BY id DESC LIMIT 20", conn)
    if not df_meetings.empty:
        ctx.append("=== MEETINGS ===\n" + df_meetings.to_string(index=False))

    df_ai = pd.read_sql("""
        SELECT a.task, a.owner, a.deadline, a.priority, a.status, m.title as meeting
        FROM action_items a JOIN meetings m ON a.meeting_id=m.id
        ORDER BY a.id DESC LIMIT 50""", conn)
    if not df_ai.empty:
        ctx.append("\n=== ACTION ITEMS ===\n" + df_ai.to_string(index=False))

    df_esc = pd.read_sql("""
        SELECT e.issue, e.raised_by, e.assigned_to, e.severity, e.status,
               e.occurrence_count, e.duplicate_of, m.title as meeting
        FROM escalations e JOIN meetings m ON e.meeting_id=m.id
        ORDER BY e.id DESC LIMIT 30""", conn)
    if not df_esc.empty:
        ctx.append("\n=== ESCALATIONS ===\n" + df_esc.to_string(index=False))

    df_risks = pd.read_sql("""
        SELECT r.description, r.impact, r.teams_involved, r.severity, m.title as meeting
        FROM risks r JOIN meetings m ON r.meeting_id=m.id
        ORDER BY r.id DESC LIMIT 30""", conn)
    if not df_risks.empty:
        ctx.append("\n=== RISKS ===\n" + df_risks.to_string(index=False))

    df_dec = pd.read_sql("""
        SELECT d.decision, d.rationale, d.decision_maker, m.title as meeting
        FROM decisions d JOIN meetings m ON d.meeting_id=m.id
        ORDER BY d.id DESC LIMIT 20""", conn)
    if not df_dec.empty:
        ctx.append("\n=== DECISIONS ===\n" + df_dec.to_string(index=False))

    df_proj = pd.read_sql("""
        SELECT p.name, p.status, p.description, m.title as meeting
        FROM projects p JOIN meetings m ON p.meeting_id=m.id
        ORDER BY p.id DESC LIMIT 20""", conn)
    if not df_proj.empty:
        ctx.append("\n=== PROJECTS ===\n" + df_proj.to_string(index=False))

    conn.close()
    return "\n".join(ctx) if ctx else "No data ingested yet."


def answer_nl_query(question: str, chat_history: list) -> str:
    _, llm, _ = init_resources()
    vector_store = init_vector_store()

    db_ctx = build_db_context()
    try:
        docs    = vector_store.similarity_search(question, k=3)
        vec_ctx = "\n\n".join([
            f"[{d.metadata.get('title','')}]\n{d.page_content[:600]}" for d in docs
        ])
    except Exception:
        vec_ctx = "No semantic results."

    prompt   = NL_QUERY_PROMPT.format(db_context=db_ctx, vector_context=vec_ctx, question=question)
    messages = [SystemMessage(content=prompt)]
    for msg in chat_history[-6:]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=question))

    response = llm.invoke(messages).content.strip()
    return re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()