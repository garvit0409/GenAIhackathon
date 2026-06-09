import os
import re
import sqlite3
import json
import io
from datetime import datetime
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# Email Modules
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

# File Processing
from docx import Document as DocxReader

# LangChain & Vector DB
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq

# Native Groq Client for Audio/Whisper
from groq import Groq

# ══════════════════════════════════════════
# CONFIG & ENV
# ══════════════════════════════════════════
load_dotenv()
if not os.getenv("GROQ_API_KEY"):
    st.error("Missing GROQ_API_KEY in .env file.")
    st.stop()

# Email Credentials
SENDER_EMAIL = os.getenv("GMAIL_USER")
APP_PASSWORD = os.getenv("GMAIL_PASS")

INTEL_DB   = "meeting_intel.db"
CHROMA_PATH = "./chroma_meeting_db"
COLLECTION  = "meeting_transcripts"


# ══════════════════════════════════════════
# DATABASE SCHEMA
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

    c.execute("""CREATE TABLE IF NOT EXISTS escalations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        meeting_id INTEGER,
        issue TEXT,
        raised_by TEXT,
        assigned_to TEXT,
        severity TEXT,
        status TEXT DEFAULT 'Open',
        FOREIGN KEY(meeting_id) REFERENCES meetings(id)
    )""")

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

init_db()

# ══════════════════════════════════════════
# LLM, EMBEDDINGS & AUDIO CLIENT
# ══════════════════════════════════════════
@st.cache_resource
def init_resources():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.1)
    
    try:
        groq_audio_client = Groq()
    except Exception as e:
        st.error(f"Failed to initialize Groq Audio Client. Error: {e}")
        groq_audio_client = None
        
    return embeddings, llm, groq_audio_client

embeddings, llm, groq_audio_client = init_resources()

@st.cache_resource
def init_vector_store():
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION
    )

vector_store = init_vector_store()


# ══════════════════════════════════════════
# AUTOMATED EMAIL ENGINE
# ══════════════════════════════════════════
def send_email_alert(subject, body, to_email, attachment=None, attachment_name=""):
    if not SENDER_EMAIL or not APP_PASSWORD:
        st.error("Email configuration missing. Please verify GMAIL_USER and GMAIL_PASS in your environment.")
        return False

    if not to_email or to_email == "manager@example.com":
        st.warning("Skipping email alert: Please provide a valid recipient email address.")
        return False

    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    if attachment is not None:
        payload = MIMEBase('application', 'octet-stream')
        payload.set_payload(attachment.read())
        encoders.encode_base64(payload)
        payload.add_header('Content-Disposition', f'attachment; filename={attachment_name}')
        msg.attach(payload)
        attachment.seek(0)

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(SENDER_EMAIL, APP_PASSWORD)
        server.sendmail(SENDER_EMAIL, to_email, msg.as_string())
        server.quit()
        st.toast(f"✉️ Email Alert Dispatched to {to_email}!")
        return True
    except Exception as e:
        st.error(f"Failed to transmit email notification: {e}")
        return False

def generate_meeting_summary_email_body(intel: dict) -> str:
    """Formats the JSON structure into an organized text email body."""
    title = intel.get("meeting_title", "Untitled Meeting")
    body = f"Hello,\n\nA new meeting context has been ingested into MeetingIQ.\n\n"
    body += f"📌 MEETING TITLE: {title}\n"
    body += "═" * 40 + "\n\n"

    sections = [
        ("🗂 PROJECTS & INITIATIVES", "projects", ["name", "status", "description"]),
        ("✅ EXTRACTED ACTION ITEMS", "action_items", ["task", "owner", "deadline", "priority"]),
        ("🚨 ESCALATIONS LOGGED", "escalations", ["issue", "raised_by", "assigned_to", "severity"]),
        ("⚠️ RISK REGISTER ENTRIES", "risks", ["description", "impact", "teams_involved", "severity"]),
        ("🎯 CRITICAL DECISIONS LOGGED", "decisions", ["decision", "rationale", "decision_maker"])
    ]

    for label, key, fields in sections:
        items = intel.get(key, [])
        body += f"{label}:\n"
        if not items:
            body += "  • None detected\n"
        else:
            for i, item in enumerate(items, 1):
                body += f"  {i}. "
                details = [f"{f.replace('_', ' ').title()}: {item.get(f, 'N/A')}" for f in fields if item.get(f)]
                body += " | ".join(details) + "\n"
        body += "\n"

    body += "This is an automated operational notification generated via MeetingIQ."
    return body


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
            except:
                pass
        return {"meeting_title": "Untitled Meeting", "projects": [], "action_items": [],
                "escalations": [], "risks": [], "decisions": [], "stakeholders": []}
    except Exception as e:
        st.error(f"Extraction error: {e}")
        return {}
import difflib

def find_duplicate_escalation(issue_text: str, conn: sqlite3.Connection) -> int:
    """
    Checks if an issue matches an existing open escalation in the DB.
    Returns the duplicate's ID if found, otherwise None.
    """
    c = conn.cursor()
    # Pull all unresolved open escalations to check against
    c.execute("SELECT id, issue FROM escalations WHERE status = 'Open'")
    open_escalations = c.fetchall()
    
    issue_text_lower = issue_text.lower().strip()
    
    for esc_id, existing_issue in open_escalations:
        existing_lower = existing_issue.lower().strip()
        
        # 1. Direct or partial text containment match
        if issue_text_lower in existing_lower or existing_lower in issue_text_lower:
            return esc_id
            
        # 2. Fuzzy match breakdown for slight variations
        similarity = difflib.SequenceMatcher(None, issue_text_lower, existing_lower).ratio()
        if similarity > 0.75:  # Adjust this threshold (0.0 to 1.0) based on your strictness needs
            return esc_id
            
    return None        

import difflib

def store_intelligence(intel: dict, raw_content: str, source_type: str) -> int:
    conn = sqlite3.connect(INTEL_DB)
    c = conn.cursor()

    title = intel.get("meeting_title", "Untitled Meeting")
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    c.execute("INSERT INTO meetings (title, ingested_at, raw_content, source_type) VALUES (?,?,?,?)",
              (title, now, raw_content, source_type))
    meeting_id = c.lastrowid

    for p in intel.get("projects", []):
        c.execute("INSERT INTO projects (meeting_id, name, status, description) VALUES (?,?,?,?)",
                  (meeting_id, p.get("name",""), p.get("status",""), p.get("description","")))

    for a in intel.get("action_items", []):
        c.execute("INSERT INTO action_items (meeting_id, task, owner, deadline, priority) VALUES (?,?,?,?,?)",
                  (meeting_id, a.get("task",""), a.get("owner",""), a.get("deadline",""), a.get("priority","Medium")))

    # ══════════════════════════════════════════
    # DUPLICATE ESCALATION DETECTION LOOP
    # ══════════════════════════════════════════
    for e in intel.get("escalations", []):
        issue_desc = e.get("issue", "")
        issue_desc_lower = issue_desc.lower().strip()
        
        # Pull current open escalations to analyze against incoming ones
        c.execute("SELECT id, issue FROM escalations WHERE status = 'Open'")
        open_escalations = c.fetchall()
        
        duplicate_id = None
        for esc_id, existing_issue in open_escalations:
            existing_lower = existing_issue.lower().strip()
            
            # Check 1: Strict substring containment match
            if issue_desc_lower in existing_lower or existing_lower in issue_desc_lower:
                duplicate_id = esc_id
                break
                
            # Check 2: Fuzzy text match ratio (0.75 ratio covers phrasing nuances)
            similarity = difflib.SequenceMatcher(None, issue_desc_lower, existing_lower).ratio()
            if similarity > 0.75:
                duplicate_id = esc_id
                break

        if duplicate_id:
            # Duplicate found -> Append tracking history notice to the original open record
            update_note = f" [Re-raised in Meeting: '{title}']"
            c.execute("""
                UPDATE escalations 
                SET issue = issue || ? 
                WHERE id = ?
            """, (update_note, duplicate_id))
            
            # Dynamically bump up severity tier if a re-raised instance demands higher priority
            new_severity = e.get("severity", "Medium")
            if new_severity in ["Critical", "High"]:
                c.execute("UPDATE escalations SET severity = ? WHERE id = ?", (new_severity, duplicate_id))
        else:
            # Unique item -> Proceed with regular row insertion
            c.execute("""
                INSERT INTO escalations (meeting_id, issue, raised_by, assigned_to, severity) 
                VALUES (?,?,?,?,?)
            """, (meeting_id, issue_desc, e.get("raised_by",""), e.get("assigned_to",""), e.get("severity","Medium")))

    # ══════════════════════════════════════════
    # END DUPLICATE DETECTION LOGIC
    # ══════════════════════════════════════════

    for r in intel.get("risks", []):
        c.execute("INSERT INTO risks (meeting_id, description, impact, teams_involved, severity) VALUES (?,?,?,?,?)",
                  (meeting_id, r.get("description",""), r.get("impact",""), r.get("teams_involved",""), r.get("severity","Medium")))

    for d in intel.get("decisions", []):
        c.execute("INSERT INTO decisions (meeting_id, decision, rationale, decision_maker) VALUES (?,?,?,?)",
                  (meeting_id, d.get("decision",""), d.get("rationale",""), d.get("decision_maker","")))

    for s in intel.get("stakeholders", []):
        c.execute("INSERT INTO stakeholders (meeting_id, name, role, responsibility) VALUES (?,?,?,?)",
                  (meeting_id, s.get("name",""), s.get("role",""), s.get("responsibility","")))

    conn.commit()
    conn.close()

    doc_text = f"Meeting: {title}\n\n{raw_content}"
    vector_store.add_documents([Document(
        page_content=doc_text,
        metadata={"meeting_id": str(meeting_id), "title": title, "date": now}
    )])

    return meeting_id

# ══════════════════════════════════════════
# FILE PARSING (DOCUMENTS & AUDIO)
# ══════════════════════════════════════════
def parse_uploaded_file(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    
    if any(name.endswith(ext) for ext in [".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm"]):
        if groq_audio_client is None:
            st.error("Audio processing is unavailable because the Groq client is not initialized.")
            return ""
        try:
            file_name = uploaded_file.name
            file_bytes = uploaded_file.read()
            
            transcription = groq_audio_client.audio.transcriptions.create(
                file=(file_name, file_bytes),
                model="whisper-large-v3",
                response_format="text"
            )
            return transcription
        except Exception as e:
            st.error(f"Error transcribing audio path file via Whisper: {e}")
            return ""
            
    elif name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="ignore")
    elif name.endswith(".pdf"):
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(uploaded_file.read())
            tmp_path = f.name
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        os.unlink(tmp_path)
        return "\n".join([p.page_content for p in pages])
    elif name.endswith(".docx"):
        doc = DocxReader(io.BytesIO(uploaded_file.read()))
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    return ""

# ══════════════════════════════════════════
# NL QUERY ENGINE
# ══════════════════════════════════════════
NL_QUERY_PROMPT = """You are an expert meeting intelligence analyst with access to a structured database.

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
    ctx = []

    df_meetings = pd.read_sql("SELECT id, title, ingested_at FROM meetings ORDER BY id DESC LIMIT 20", conn)
    if not df_meetings.empty:
        ctx.append("=== MEETINGS ===\n" + df_meetings.to_string(index=False))

    df_ai = pd.read_sql("""SELECT a.task, a.owner, a.deadline, a.priority, a.status, m.title as meeting
                           FROM action_items a JOIN meetings m ON a.meeting_id=m.id
                           ORDER BY a.id DESC LIMIT 50""", conn)
    if not df_ai.empty:
        ctx.append("\n=== ACTION ITEMS ===\n" + df_ai.to_string(index=False))

    df_esc = pd.read_sql("""SELECT e.issue, e.raised_by, e.assigned_to, e.severity, e.status, m.title as meeting
                            FROM escalations e JOIN meetings m ON e.meeting_id=m.id
                            ORDER BY e.id DESC LIMIT 30""", conn)
    if not df_esc.empty:
        ctx.append("\n=== ESCALATIONS ===\n" + df_esc.to_string(index=False))

    df_risks = pd.read_sql("""SELECT r.description, r.impact, r.teams_involved, r.severity, m.title as meeting
                              FROM risks r JOIN meetings m ON r.meeting_id=m.id
                              ORDER BY r.id DESC LIMIT 30""", conn)
    if not df_risks.empty:
        ctx.append("\n=== RISKS ===\n" + df_risks.to_string(index=False))

    df_dec = pd.read_sql("""SELECT d.decision, d.rationale, d.decision_maker, m.title as meeting
                            FROM decisions d JOIN meetings m ON d.meeting_id=m.id
                            ORDER BY d.id DESC LIMIT 20""", conn)
    if not df_dec.empty:
        ctx.append("\n=== DECISIONS ===\n" + df_dec.to_string(index=False))

    df_proj = pd.read_sql("""SELECT p.name, p.status, p.description, m.title as meeting
                             FROM projects p JOIN meetings m ON p.meeting_id=m.id
                             ORDER BY p.id DESC LIMIT 20""", conn)
    if not df_proj.empty:
        ctx.append("\n=== PROJECTS ===\n" + df_proj.to_string(index=False))

    conn.close()
    return "\n".join(ctx) if ctx else "No data ingested yet."

def answer_nl_query(question: str, chat_history: list) -> str:
    db_ctx = build_db_context()

    try:
        docs = vector_store.similarity_search(question, k=3)
        vec_ctx = "\n\n".join([f"[{d.metadata.get('title','')}]\n{d.page_content[:600]}" for d in docs])
    except:
        vec_ctx = "No semantic results."

    prompt = NL_QUERY_PROMPT.format(db_context=db_ctx, vector_context=vec_ctx, question=question)

    messages = [SystemMessage(content=prompt)]
    for msg in chat_history[-6:]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=question))

    response = llm.invoke(messages).content.strip()
    return re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

# ══════════════════════════════════════════
# DASHBOARD DATA HELPERS
# ══════════════════════════════════════════
def get_dashboard_stats():
    conn = sqlite3.connect(INTEL_DB)
    stats = {}
    for table in ["meetings", "action_items", "escalations", "risks", "decisions", "projects"]:
        stats[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    stats["open_escalations"] = conn.execute("SELECT COUNT(*) FROM escalations WHERE status='Open'").fetchone()[0]
    stats["pending_tasks"] = conn.execute("SELECT COUNT(*) FROM action_items WHERE status='Pending'").fetchone()[0]
    stats["critical_risks"] = conn.execute("SELECT COUNT(*) FROM risks WHERE severity='Critical'").fetchone()[0]
    conn.close()
    return stats

def get_table_df(table: str, limit: int = 100):
    conn = sqlite3.connect(INTEL_DB)
    try:
        query_map = {
            "action_items": """SELECT a.id, m.title as Meeting, a.task as Task, a.owner as Owner,
                               a.deadline as Deadline, a.priority as Priority, a.status as Status
                               FROM action_items a JOIN meetings m ON a.meeting_id=m.id
                               ORDER BY a.id DESC LIMIT ?""",
            "escalations":  """SELECT e.id, m.title as Meeting, e.issue as Issue, e.raised_by as 'Raised By',
                               e.assigned_to as 'Assigned To', e.severity as Severity, e.status as Status
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
    except Exception as e:
        conn.close()
        return pd.DataFrame()

def update_status(table: str, row_id: int, new_status: str):
    conn = sqlite3.connect(INTEL_DB)
    conn.execute(f"UPDATE {table} SET status=? WHERE id=?", (new_status, row_id))
    conn.commit()
    conn.close()

# ══════════════════════════════════════════
# UI — PAGE CONFIG
# ══════════════════════════════════════════
st.set_page_config(
    page_title="MeetingIQ — Intelligence Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════
# GLOBAL STYLES
# ══════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght=300;400;500;600;700&family=JetBrains+Mono:wght=400;500&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; }
[data-testid="stSidebar"] { background: #0f1117; border-right: 1px solid #1e2130; }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
.metric-card { background: #1a1d2e; border: 1px solid #2a2d3e; border-radius: 12px; padding: 1.2rem 1.4rem; text-align: center; }
.metric-card .metric-value { font-size: 2.2rem; font-weight: 700; line-height: 1; margin-bottom: 4px; }
.metric-card .metric-label { font-size: 0.78rem; color: #8892a4; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 500; }
.metric-card.alert .metric-value { color: #f87171; }
.metric-card.warn  .metric-value { color: #fbbf24; }
.metric-card.ok    .metric-value { color: #34d399; }
.metric-card.info  .metric-value { color: #60a5fa; }
.intel-card { background: #1a1d2e; border: 1px solid #2a2d3e; border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 0.75rem; border-left: 4px solid #6366f1; }
.intel-card.high   { border-left-color: #f87171; }
.intel-card.medium { border-left-color: #fbbf24; }
.intel-card.low    { border-left-color: #34d399; }
.intel-card.critical { border-left-color: #dc2626; background: #1f1a2e; }
.section-header { font-size: 1.1rem; font-weight: 600; color: #e2e8f0; margin-bottom: 1rem; padding-bottom: 0.5rem; border-bottom: 1px solid #2a2d3e; display: flex; align-items: center; gap: 8px; }
.badge { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.72rem; font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase; }
.badge-red    { background: #3f1a1a; color: #f87171; }
.badge-yellow { background: #3f2f0a; color: #fbbf24; }
.badge-green  { background: #0a3f2a; color: #34d399; }
.badge-blue   { background: #0a1f3f; color: #60a5fa; }
.badge-purple { background: #1f0a3f; color: #a78bfa; }
.chat-user { background: #1e293b; border-radius: 12px 12px 2px 12px; padding: 0.8rem 1rem; margin: 0.5rem 0; max-width: 75%; margin-left: auto; border: 1px solid #2a3f5f; color: #e2e8f0; }
.chat-assistant { background: #1a1d2e; border-radius: 12px 12px 12px 2px; padding: 0.8rem 1rem; margin: 0.5rem 0; max-width: 85%; border: 1px solid #2a2d3e; color: #e2e8f0; }
.page-title { font-size: 1.5rem; font-weight: 700; color: #e2e8f0; margin-bottom: 1.5rem; }
.stTextArea textarea, .stTextInput input { background: #1a1d2e !important; border: 1px solid #2a2d3e !important; color: #e2e8f0 !important; border-radius: 8px !important; }
.stButton button { background: #6366f1 !important; color: white !important; border-radius: 8px !important; font-weight: 600 !important; }
.logo-text { font-size: 1.3rem; font-weight: 700; color: #e2e8f0; }
.logo-accent { color: #6366f1; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════
# SIDEBAR NAVIGATION & CONFIG
# ══════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="logo-text">Meeting<span class="logo-accent">IQ</span></div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#8892a4;font-size:0.78rem;margin-bottom:1.5rem;">Organizational Intelligence Platform</div>', unsafe_allow_html=True)

    # Global Recipient Config
    st.markdown("---")
    st.markdown('<div style="font-size:0.72rem;color:#8892a4;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px;">Alert Routing</div>', unsafe_allow_html=True)
    manager_email = st.text_input("Manager Email Address", value="manager@example.com", label_visibility="collapsed")
    st.markdown("---")

    stats = get_dashboard_stats()
    st.markdown(f"""
    <div style="background:#0f1117;border:1px solid #1e2130;border-radius:8px;padding:0.8rem 1rem;margin-bottom:1.2rem;">
        <div style="font-size:0.72rem;color:#8892a4;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:8px;">Database Status</div>
        <div style="display:flex;justify-content:space-between;margin-bottom:4px;"><span style="color:#8892a4;font-size:0.82rem;">Meetings</span><span style="color:#60a5fa;font-weight:600;font-size:0.82rem;">{stats['meetings']}</span></div>
        <div style="display:flex;justify-content:space-between;margin-bottom:4px;"><span style="color:#8892a4;font-size:0.82rem;">Open Escalations</span><span style="color:#f87171;font-weight:600;font-size:0.82rem;">{stats['open_escalations']}</span></div>
        <div style="display:flex;justify-content:space-between;margin-bottom:4px;"><span style="color:#8892a4;font-size:0.82rem;">Pending Tasks</span><span style="color:#fbbf24;font-weight:600;font-size:0.82rem;">{stats['pending_tasks']}</span></div>
        <div style="display:flex;justify-content:space-between;"><span style="color:#8892a4;font-size:0.82rem;">Critical Risks</span><span style="color:#f87171;font-weight:600;font-size:0.82rem;">{stats['critical_risks']}</span></div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["🏠  Dashboard", "➕  Ingest Meeting", "✅  Action Items", "🚨  Escalations",
         "⚠️  Risks", "🎯  Decisions", "📋  Projects", "👥  Stakeholders", "💬  Query Intelligence"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown('<div style="color:#8892a4;font-size:0.72rem;">Powered by Groq LLaMA 3.3 · Whisper-Large-V3 · ChromaDB</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════
if page == "🏠  Dashboard":
    st.markdown('<div class="page-title">🧠 Intelligence Dashboard</div>', unsafe_allow_html=True)

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    kpis = [
        (col1, stats['meetings'],          "Meetings Ingested",   "info"),
        (col2, stats['projects'],          "Projects Tracked",    "info"),
        (col3, stats['pending_tasks'],     "Pending Tasks",       "warn"),
        (col4, stats['open_escalations'],  "Open Escalations",    "alert"),
        (col5, stats['critical_risks'],    "Critical Risks",      "alert"),
        (col6, stats['decisions'],         "Decisions Logged",    "ok"),
    ]
    for col, val, label, cls in kpis:
        with col:
            st.markdown(f'<div class="metric-card {cls}"><div class="metric-value">{val}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns([1.1, 0.9])

    with col_left:
        st.markdown('<div class="section-header">🚨 Recent Escalations</div>', unsafe_allow_html=True)
        df_esc = get_table_df("escalations", 5)
        if df_esc.empty:
            st.info("No escalations logged yet.")
        else:
            for _, row in df_esc.iterrows():
                issue_text = str(row.get("Issue", ""))
                sev = str(row.get("Severity","")).lower()
                cls = "critical" if sev == "critical" else ("high" if sev == "high" else ("medium" if sev == "medium" else "low"))
                badge_cls = "badge-red" if sev in ["critical","high"] else ("badge-yellow" if sev == "medium" else "badge-green")
                
                # ══════════════════════════════════════════
                # RECURRING DETECTION SYSTEM
                # ══════════════════════════════════════════
                # Count how many times the duplicate string sequence appears in the issue text
                duplicate_count = issue_text.count("[Re-raised in Meeting:")
                is_recurring = duplicate_count > 0
                
                # Optional: Build a small numeric counter indicator tag if it's hit multiple times
                counter_tag = f" ({duplicate_count}x)" if duplicate_count > 1 else ""
                recurring_badge_html = f'<span class="badge badge-purple" style="font-size:0.68rem; margin-right: 4px;">🔄 Recurring{counter_tag}</span>' if is_recurring else ''
                # ══════════════════════════════════════════

                st.markdown(f"""
                <div class="intel-card {cls}">
                    <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:8px;">
                        <div style="color:#e2e8f0;font-weight:500;font-size:0.88rem;">
                            {recurring_badge_html}{issue_text}
                        </div>
                        <span class="badge {badge_cls}">{row.get('Severity','')}</span>
                    </div>
                    <div style="color:#8892a4;font-size:0.78rem;margin-top:6px;">Raised by <strong style="color:#a5b4fc">{row.get('Raised By','—')}</strong> · {row.get('Meeting','')[:40]}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-header" style="margin-top:1.5rem;">⚠️ Active Risks</div>', unsafe_allow_html=True)
        df_risks = get_table_df("risks", 4)
        if df_risks.empty:
            st.info("No risks logged yet.")
        else:
            for _, row in df_risks.iterrows():
                sev = str(row.get("Severity","")).lower()
                cls = "critical" if sev == "critical" else ("high" if sev == "high" else "medium")
                st.markdown(f"""
                <div class="intel-card {cls}">
                    <div style="color:#e2e8f0;font-weight:500;font-size:0.88rem;">{row.get('Risk','')[:80]}</div>
                    <div style="color:#8892a4;font-size:0.78rem;margin-top:4px;">Impact: {row.get('Impact','')[:60]} · Teams: <strong style="color:#a5b4fc">{row.get('Teams','')[:40]}</strong></div>
                </div>""", unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-header">✅ Pending Action Items</div>', unsafe_allow_html=True)
        df_ai = get_table_df("action_items", 8)
        df_pending = df_ai[df_ai["Status"] == "Pending"] if not df_ai.empty else pd.DataFrame()
        if df_pending.empty:
            st.info("All tasks cleared! 🎉")
        else:
            for _, row in df_pending.iterrows():
                pri = str(row.get("Priority","")).lower()
                badge_cls = "badge-red" if pri == "high" else ("badge-yellow" if pri == "medium" else "badge-green")
                dl = row.get("Deadline","")
                st.markdown(f"""
                <div class="intel-card {'high' if pri=='high' else ('medium' if pri=='medium' else 'low')}">
                    <div style="color:#e2e8f0;font-size:0.86rem;font-weight:500;">{row.get('Task','')[:70]}</div>
                    <div style="display:flex;gap:8px;align-items:center;margin-top:6px;flex-wrap:wrap;">
                        <span style="color:#a5b4fc;font-size:0.78rem;">👤 {row.get('Owner','Unassigned')}</span>
                        {'<span style="color:#8892a4;font-size:0.78rem;">📅 '+str(dl)+'</span>' if dl else ''}
                        <span class="badge {badge_cls}">{row.get('Priority','')}</span>
                    </div>
                </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-header" style="margin-top:1.5rem;">🎯 Recent Decisions</div>', unsafe_allow_html=True)
        df_dec = get_table_df("decisions", 4)
        if df_dec.empty:
            st.info("No decisions logged yet.")
        else:
            for _, row in df_dec.iterrows():
                st.markdown(f"""
                <div class="intel-card">
                    <div style="color:#e2e8f0;font-weight:500;font-size:0.86rem;">{row.get('Decision','')[:70]}</div>
                    <div style="color:#8892a4;font-size:0.78rem;margin-top:4px;">{row.get('Rationale','')[:60]}{' · <strong style="color:#a5b4fc">'+str(row.get('Decision Maker',''))+'</strong>' if row.get('Decision Maker') else ''}</div>
                </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════
# PAGE: INGEST MEETING
# ══════════════════════════════════════════
elif page == "➕  Ingest Meeting":
    st.markdown('<div class="page-title">➕ Ingest Meeting</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["✍️  Paste Text", "📎  Upload File / Audio", "🎬  Example Demo"])

    with tab1:
        st.markdown("Paste a meeting summary, transcript, or discussion notes below.")
        meeting_text = st.text_area("Meeting Content", height=280, placeholder="e.g. Rahul will coordinate with backend team...", label_visibility="collapsed")
        if st.button("🧠 Extract Intelligence", use_container_width=True):
            if meeting_text.strip():
                with st.spinner("Analyzing meeting content with AI..."):
                    intel = extract_intelligence(meeting_text)
                    if intel:
                        mid = store_intelligence(intel, meeting_text, "text_paste")
                        st.success(f"✅ Meeting ingested successfully! (ID: {mid})")
                        
                        # --- TRIGGER EMAIL: MEETING INGUSTED ---
                        email_body = generate_meeting_summary_email_body(intel)
                        sent = send_email_alert(
                            subject=f"[MeetingIQ Summary] {intel.get('meeting_title', 'New Meeting Ingested')}",
                            body=email_body,
                            to_email=manager_email
                        )
                        if sent:
                            st.success("📩 Dashboard summary email successfully sent to Manager!")
                        st.balloons()

                        # Show preview
                        st.markdown("### 📊 Extracted Intelligence Preview")
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.metric("Projects", len(intel.get("projects", [])))
                            st.metric("Action Items", len(intel.get("action_items", [])))
                        with c2:
                            st.metric("Escalations", len(intel.get("escalations", [])))
                            st.metric("Risks", len(intel.get("risks", [])))
                        with c3:
                            st.metric("Decisions", len(intel.get("decisions", [])))
                            st.metric("Stakeholders", len(intel.get("stakeholders", [])))
            else:
                st.warning("Please enter some meeting content first.")

    with tab2:
        st.markdown("Upload documents or meeting audio recordings here.")
        uploaded = st.file_uploader("Upload meeting document or audio file", type=["txt", "pdf", "docx", "mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"])
        if uploaded:
            name_lower = uploaded.name.lower()
            if any(name_lower.endswith(ext) for ext in [".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm"]):
                st.audio(uploaded, format='audio/wav')
                
            with st.spinner("Parsing file (transcribing audio files may take a moment)..."):
                content = parse_uploaded_file(uploaded)
                
            if content:
                st.success(f"Successfully processed `{uploaded.name}`!")
                with st.expander("Preview parsed context/transcript text"):
                    st.text(content[:1200] + ("..." if len(content) > 1200 else ""))

                if st.button("🧠 Extract Intelligence from File", use_container_width=True):
                    with st.spinner("Analyzing with AI..."):
                        intel = extract_intelligence(content)
                        if intel:
                            mid = store_intelligence(intel, content, f"file:{uploaded.name}")
                            st.success(f"✅ Ingested! Meeting ID: {mid}")
                            
                            # --- TRIGGER EMAIL: MEETING INGESTED VIA FILE ---
                            email_body = generate_meeting_summary_email_body(intel)
                            sent = send_email_alert(
                                subject=f"[MeetingIQ File Summary] {intel.get('meeting_title', 'New File Upload')}",
                                body=email_body,
                                to_email=manager_email
                            )
                            if sent:
                                st.success("📩 Dashboard summary email successfully sent to Manager!")
            else:
                st.error("Could not parse file content. Please verify your config layout.")

    with tab3:
        st.markdown("**Try with this illustrative scenario from the problem statement:**")
        demo_text = """Team: Engineering + Product sync — Q3 Planning\nDate: Today\n\nThe payment integration project is significantly delayed because the Vendor API has been highly unstable over the past week, causing repeated failures in our staging environment. \n\nRahul (Backend Lead) will coordinate directly with the backend team and the vendor's technical contact before this Friday to resolve authentication issues. If this issue continues beyond Friday, it may critically impact the Phase-2 release scheduled for next month.\n\nPriya (Product Manager) escalated the concern formally to leadership, flagging this as a high-priority blocker.\n\nAdditionally, Amit confirmed that the new dashboard feature is on track for deployment by end of month. The design team (led by Sara) needs to finalize UI mockups by Wednesday so the frontend team can begin integration.\n\nDecision made: The team agreed to delay Phase-2 launch by two weeks to buffer for the API instability. This was decided by CTO Rajesh after reviewing risk scenarios.\n\nSarah raised a risk that if the backend delay extends, it will also impact the mobile team's release cycle since they depend on the same payment APIs.\n\nKey stakeholders: Rahul (Backend), Priya (Product), Amit (Frontend), Sara (Design), Rajesh (CTO)"""
        st.text_area("Demo meeting content:", value=demo_text, height=200, disabled=True)

        if st.button("🧠 Run Demo Extraction", use_container_width=True):
            with st.spinner("Extracting intelligence..."):
                intel = extract_intelligence(demo_text)
                if intel:
                    mid = store_intelligence(intel, demo_text, "demo")
                    st.success(f"✅ Demo meeting ingested! (ID: {mid})")
                    
                    # --- TRIGGER EMAIL: DEMO MEETING INGESTED ---
                    email_body = generate_meeting_summary_email_body(intel)
                    sent = send_email_alert(
                        subject=f"[MeetingIQ Demo Summary] {intel.get('meeting_title')}",
                        body=email_body,
                        to_email=manager_email
                    )
                    if sent:
                        st.success("📩 Dashboard summary email successfully sent to Manager!")

# ══════════════════════════════════════════
# PAGE: ACTION ITEMS
# ══════════════════════════════════════════
elif page == "✅  Action Items":
    st.markdown('<div class="page-title">✅ Action Items</div>', unsafe_allow_html=True)
    df = get_table_df("action_items", 200)

    if df.empty:
        st.info("No action items yet. Ingest a meeting to get started.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1: filter_status = st.selectbox("Status", ["All", "Pending", "Done"])
        with col2: filter_priority = st.selectbox("Priority", ["All", "High", "Medium", "Low"])
        with col3: filter_owner = st.text_input("Filter by owner", placeholder="e.g. Rahul")

        filtered = df.copy()
        if filter_status != "All": filtered = filtered[filtered["Status"] == filter_status]
        if filter_priority != "All": filtered = filtered[filtered["Priority"] == filter_priority]
        if filter_owner: filtered = filtered[filtered["Owner"].str.contains(filter_owner, case=False, na=False)]

        for _, row in filtered.iterrows():
            pri = str(row.get("Priority","")).lower()
            badge_cls = "badge-red" if pri == "high" else ("badge-yellow" if pri == "medium" else "badge-green")
            card_cls  = pri if pri in ["high","medium","low"] else ""
            status_badge = "badge-green" if row.get("Status") == "Done" else "badge-yellow"

            with st.container():
                st.markdown(f"""
                <div class="intel-card {card_cls}">
                    <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:8px;">
                        <div style="color:#e2e8f0;font-weight:500;font-size:0.9rem;">{row.get('Task','')}</div>
                        <div style="display:flex;gap:6px;flex-shrink:0;">
                            <span class="badge {badge_cls}">{row.get('Priority','')}</span>
                            <span class="badge {status_badge}">{row.get('Status','')}</span>
                        </div>
                    </div>
                    <div style="color:#8892a4;font-size:0.78rem;margin-top:6px;display:flex;gap:12px;flex-wrap:wrap;">
                        <span>👤 <strong style="color:#a5b4fc">{row.get('Owner','Unassigned')}</strong></span>
                        {'<span>📅 '+str(row.get('Deadline',''))+'</span>' if row.get('Deadline') else ''}
                        <span>📁 {str(row.get('Meeting',''))[:40]}</span>
                    </div>
                </div>""", unsafe_allow_html=True)

                if row.get("Status") == "Pending":
                    if st.button(f"Mark Done", key=f"done_{row['id']}"):
                        update_status("action_items", int(row["id"]), "Done")
                        
                        # --- TRIGGER EMAIL: ACTION ITEM COMPLETED ---
                        subject_ai = f"[Task Completed Alert] Action Item Resolved!"
                        body_ai = f"Hello Manager,\n\nThis is an automated update confirming an action item has been resolved.\n\nTask Details:\n• Task: {row.get('Task')}\n• Owner: {row.get('Owner', 'Unassigned')}\n• Meeting: {row.get('Meeting')}\n• Resolution Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                        
                        send_email_alert(subject=subject_ai, body=body_ai, to_email=manager_email)
                        st.rerun()

# ══════════════════════════════════════════
# PAGE: ESCALATIONS
# ══════════════════════════════════════════
elif page == "🚨  Escalations":
    st.markdown('<div class="page-title">🚨 Escalations</div>', unsafe_allow_html=True)
    df = get_table_df("escalations", 200)
    if df.empty:
        st.info("No escalations logged yet.")
    else:
        col1, col2 = st.columns(2)
        with col1: filter_sev = st.selectbox("Severity", ["All", "Critical", "High", "Medium", "Low"])
        with col2: filter_status = st.selectbox("Status", ["All", "Open", "Resolved"])

        filtered = df.copy()
        if filter_sev != "All": filtered = filtered[filtered["Severity"] == filter_sev]
        if filter_status != "All": filtered = filtered[filtered["Status"] == filter_status]

        for _, row in filtered.iterrows():
            sev = str(row.get("Severity","")).lower()
            cls = "critical" if sev == "critical" else ("high" if sev == "high" else ("medium" if sev == "medium" else "low"))
            badge_cls = "badge-red" if sev in ["critical","high"] else ("badge-yellow" if sev == "medium" else "badge-green")
            st.markdown(f"""
            <div class="intel-card {cls}">
                <div style="display:flex;justify-content:space-between;gap:8px;align-items:flex-start;">
                    <div style="color:#e2e8f0;font-weight:600;font-size:0.9rem;">{row.get('Issue','')}</div>
                    <span class="badge {badge_cls}">{row.get('Severity','')}</span>
                </div>
                <div style="color:#8892a4;font-size:0.78rem;margin-top:6px;display:flex;gap:12px;flex-wrap:wrap;">
                    <span>🔴 Raised by <strong style="color:#f87171">{row.get('Raised By','')}</strong></span>
                    {'<span>➡️ Assigned to <strong style="color:#a5b4fc">'+str(row.get('Assigned To',''))+'</strong></span>' if row.get('Assigned To') else ''}
                    <span>📁 {str(row.get('Meeting',''))[:40]}</span>
                    <span class="badge {'badge-yellow' if row.get('Status')=='Open' else 'badge-green'}">{row.get('Status','')}</span>
                </div>
            </div>""", unsafe_allow_html=True)

            if row.get("Status") == "Open":
                if st.button("Mark Resolved", key=f"res_{row['id']}"):
                    update_status("escalations", int(row["id"]), "Resolved")
                    
                    # --- TRIGGER EMAIL: ESCALATION RESOLVED ---
                    subject_esc = f"[Escalation Resolved Alert] Issue Closed!"
                    body_esc = f"Hello Manager,\n\nThis is an automated alert confirming that an escalation blocker has been formally closed.\n\nEscalation Details:\n• Blocker Issue: {row.get('Issue')}\n• Severity: {row.get('Severity')}\n• Raised By: {row.get('Raised By')}\n• Associated Meeting Context: {row.get('Meeting')}\n\nThis item is now marked as Resolved on the system."
                    
                    send_email_alert(subject=subject_esc, body=body_esc, to_email=manager_email)
                    st.rerun()

# ══════════════════════════════════════════
# PAGE: RISKS
# ══════════════════════════════════════════
elif page == "⚠️  Risks":
    st.markdown('<div class="page-title">⚠️ Risk Register</div>', unsafe_allow_html=True)
    df = get_table_df("risks", 200)
    if df.empty:
        st.info("No risks logged yet.")
    else:
        filter_sev = st.selectbox("Filter by Severity", ["All", "Critical", "High", "Medium", "Low"])
        filtered = df if filter_sev == "All" else df[df["Severity"] == filter_sev]

        for _, row in filtered.iterrows():
            sev = str(row.get("Severity","")).lower()
            cls = "critical" if sev == "critical" else ("high" if sev == "high" else ("medium" if sev == "medium" else "low"))
            badge_cls = "badge-red" if sev in ["critical","high"] else ("badge-yellow" if sev == "medium" else "badge-green")
            st.markdown(f"""
            <div class="intel-card {cls}">
                <div style="display:flex;justify-content:space-between;gap:8px;">
                    <div style="color:#e2e8f0;font-weight:500;font-size:0.9rem;">{row.get('Risk','')}</div>
                    <span class="badge {badge_cls}">{row.get('Severity','')}</span>
                </div>
                <div style="color:#8892a4;font-size:0.78rem;margin-top:6px;"><strong style="color:#e2e8f0">Impact:</strong> {row.get('Impact','')[:80]}</div>
                <div style="color:#8892a4;font-size:0.78rem;margin-top:3px;">Teams: <strong style="color:#a5b4fc">{row.get('Teams','')}</strong> · {str(row.get('Meeting',''))[:40]}</div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════
# PAGE: DECISIONS
# ══════════════════════════════════════════
elif page == "🎯  Decisions":
    st.markdown('<div class="page-title">🎯 Decision Log</div>', unsafe_allow_html=True)
    df = get_table_df("decisions", 200)
    if df.empty:
        st.info("No decisions logged yet.")
    else:
        for _, row in df.iterrows():
            st.markdown(f"""
            <div class="intel-card">
                <div style="color:#e2e8f0;font-weight:600;font-size:0.9rem;">📌 {row.get('Decision','')}</div>
                <div style="color:#8892a4;font-size:0.82rem;margin-top:6px;"><strong style="color:#e2e8f0">Rationale:</strong> {row.get('Rationale','')[:100]}</div>
                <div style="color:#8892a4;font-size:0.78rem;margin-top:4px;display:flex;gap:12px;flex-wrap:wrap;">
                    {'<span>👤 <strong style="color:#a5b4fc">'+str(row.get('Decision Maker',''))+'</strong></span>' if row.get('Decision Maker') else ''}
                    <span>📁 {str(row.get('Meeting',''))[:40]}</span>
                </div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════
# PAGE: PROJECTS
# ══════════════════════════════════════════
elif page == "📋  Projects":
    st.markdown('<div class="page-title">📋 Projects & Initiatives</div>', unsafe_allow_html=True)
    df = get_table_df("projects", 200)
    if df.empty:
        st.info("No projects logged yet.")
    else:
        for _, row in df.iterrows():
            status = str(row.get("Status","")).lower()
            cls = "high" if "delay" in status or "blocked" in status or "risk" in status else ("ok" if "track" in status or "complete" in status else "")
            st.markdown(f"""
            <div class="intel-card {cls}">
                <div style="display:flex;justify-content:space-between;gap:8px;">
                    <div style="color:#e2e8f0;font-weight:600;font-size:0.9rem;">🗂 {row.get('Project','')}</div>
                    <span class="badge badge-blue">{row.get('Status','')}</span>
                </div>
                <div style="color:#8892a4;font-size:0.82rem;margin-top:6px;">{row.get('Description','')[:120]}</div>
                <div style="color:#8892a4;font-size:0.78rem;margin-top:4px;">📁 {str(row.get('Meeting',''))[:40]}</div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════
# PAGE: STAKEHOLDERS
# ══════════════════════════════════════════
elif page == "👥  Stakeholders":
    st.markdown('<div class="page-title">👥 Stakeholder Map</div>', unsafe_allow_html=True)
    df = get_table_df("stakeholders", 200)
    if df.empty:
        st.info("No stakeholders logged yet.")
    else:
        conn = sqlite3.connect(INTEL_DB)
        people_df = pd.read_sql("""
            SELECT name, GROUP_CONCAT(DISTINCT role) as roles,
                   GROUP_CONCAT(DISTINCT responsibility) as responsibilities,
                   COUNT(DISTINCT meeting_id) as meeting_count
            FROM stakeholders GROUP BY LOWER(name) ORDER BY meeting_count DESC
        """, conn)
        conn.close()

        cols = st.columns(3)
        for i, (_, row) in enumerate(people_df.iterrows()):
            with cols[i % 3]:
                st.markdown(f"""
                <div class="intel-card" style="border-left-color:#a78bfa;">
                    <div style="color:#e2e8f0;font-weight:600;font-size:0.92rem;">👤 {row.get('name','')}</div>
                    <div style="color:#a5b4fc;font-size:0.78rem;margin-top:4px;">{str(row.get('roles',''))[:60]}</div>
                    <div style="color:#8892a4;font-size:0.75rem;margin-top:4px;">{str(row.get('responsibilities',''))[:80]}</div>
                    <div style="color:#6b7280;font-size:0.72rem;margin-top:6px;">Appeared in {row.get('meeting_count',0)} meeting(s)</div>
                </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════
# PAGE: QUERY INTELLIGENCE
# ══════════════════════════════════════════
elif page == "💬  Query Intelligence":
    st.markdown('<div class="page-title">💬 Query Intelligence</div>', unsafe_allow_html=True)
    suggestions = [
        "What are all open escalations?", "Show pending tasks assigned to Rahul",
        "Which projects are at risk?", "List all high-priority blockers",
        "What decisions were made about the payment integration?", "Which meetings discussed Vendor API issues?",
    ]
    cols = st.columns(3)
    for i, s in enumerate(suggestions):
        with cols[i % 3]:
            if st.button(s, key=f"sug_{i}", use_container_width=True):
                if "chat_messages" not in st.session_state: st.session_state.chat_messages = []
                st.session_state.chat_messages.append({"role": "user", "content": s})
                with st.spinner("Analyzing intelligence database..."):
                    answer = answer_nl_query(s, st.session_state.chat_messages[:-1])
                st.session_state.chat_messages.append({"role": "assistant", "content": answer})
                st.rerun()

    st.markdown("---")
    if "chat_messages" not in st.session_state: st.session_state.chat_messages = []

    if not st.session_state.chat_messages:
        st.markdown('<div style="text-align:center;padding:3rem;color:#4b5563;"><div style="font-size:2.5rem;margin-bottom:0.5rem;">🧠</div><div style="font-size:1rem;font-weight:500;color:#6b7280;">Ask any question about meetings, tasks, escalations, risks, or decisions.</div></div>', unsafe_allow_html=True)
    else:
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.chat_messages:
                if msg["role"] == "user":
                    st.markdown(f'<div class="chat-user">{msg["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-assistant"><strong style="color:#a5b4fc;font-size:0.8rem;">MeetingIQ</strong><br><br>{msg["content"]}</div>', unsafe_allow_html=True)

    user_input = st.chat_input("Ask about your organizational intelligence...")
    if user_input:
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        with st.spinner("Querying intelligence database..."):
            answer = answer_nl_query(user_input, st.session_state.chat_messages[:-1])
        st.session_state.chat_messages.append({"role": "assistant", "content": answer})
        st.rerun()

    if st.session_state.chat_messages:
        if st.button("🗑 Clear Chat"):
            st.session_state.chat_messages = []
            st.rerun()