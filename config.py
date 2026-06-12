import os
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────
INTEL_DB    = "meeting_intel.db"
CHROMA_PATH = "./chroma_meeting_db"
COLLECTION  = "meeting_transcripts"

# ── Email credentials ──────────────────────
SENDER_EMAIL = os.getenv("GMAIL_USER")
APP_PASSWORD  = os.getenv("GMAIL_PASS")

# ── Groq ───────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ── Duplicate Detection Thresholds ─────────
FUZZY_THRESHOLD = 0.75   # difflib ratio
TFIDF_THRESHOLD = 0.55   # cosine similarity