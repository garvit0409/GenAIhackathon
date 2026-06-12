import io
import os
import tempfile

import streamlit as st
from docx import Document as DocxReader
from langchain_community.document_loaders import PyPDFLoader

from resources import init_resources

# ══════════════════════════════════════════
# FILE PARSING
# ══════════════════════════════════════════

AUDIO_EXTENSIONS = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm"}


def parse_uploaded_file(uploaded_file) -> str:
    """
    Dispatch to the appropriate parser based on file extension.
    Returns the extracted text content as a string.
    """
    name = uploaded_file.name.lower()

    # ── Audio → Whisper transcription ─────────────────────────────────────
    if any(name.endswith(ext) for ext in AUDIO_EXTENSIONS):
        _, _, groq_audio_client = init_resources()
        if groq_audio_client is None:
            st.error("Audio processing is unavailable because the Groq client is not initialized.")
            return ""
        try:
            file_bytes = uploaded_file.read()
            transcription = groq_audio_client.audio.transcriptions.create(
                file=(uploaded_file.name, file_bytes),
                model="whisper-large-v3",
                response_format="text",
            )
            return transcription
        except Exception as e:
            st.error(f"Error transcribing audio: {e}")
            return ""

    # ── Plain text ─────────────────────────────────────────────────────────
    elif name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="ignore")

    # ── PDF ────────────────────────────────────────────────────────────────
    elif name.endswith(".pdf"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(uploaded_file.read())
            tmp_path = f.name
        loader = PyPDFLoader(tmp_path)
        pages  = loader.load()
        os.unlink(tmp_path)
        return "\n".join([p.page_content for p in pages])

    # ── DOCX ───────────────────────────────────────────────────────────────
    elif name.endswith(".docx"):
        doc = DocxReader(io.BytesIO(uploaded_file.read()))
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

    return ""