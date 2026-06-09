# 🧠 MeetingIQ — Organizational Intelligence Platform

**MeetingIQ** is an AI-powered organizational intelligence platform that transforms raw meeting transcripts, documents, and audio recordings into structured, actionable insights. Built with **Streamlit**, **Groq (LLaMA 3.3 & Whisper)**, and **ChromaDB**, it automatically extracts and tracks projects, action items, escalations, risks, and critical decisions.

It features a specialized **Duplicate Escalation Detection & Tracking System** to flag recurring issues across different meetings and an automated email engine to route real-time operational alerts directly to managers.

---

## 🚀 Key Features

* **Multi-Modal Ingestion:** Paste raw text, upload documents (`.txt`, `.pdf`, `.docx`), or upload audio files directly (`.mp3`, `.wav`, `.m4a`, etc.) for automatic transcription via **Whisper-Large-V3**.
* **AI Extraction Engine:** Powered by **LLaMA 3.3 (70B)** to parse unstructured text into highly accurate, structured JSON schemas.
* **Intelligent Duplicate Escalation Tracker:** Automatically detects if an incoming escalation matches a pre-existing open issue using fuzzy text matching ($>75\%$ similarity). It appends tracking history directly to the parent record and dynamically bumps severity tiers if necessary.
* **Natural Language Query (RAG):** Talk directly to your meeting database. Uses **ChromaDB** vector embeddings (`all-MiniLM-L6-v2`) combined with structured SQL constraints to provide hallucination-free answers about tasks, blockers, and decisions.
* **Operational Email Routing:** Automatically dispatches structured Markdown/Text emails using Gmail SMTP whenever new meetings are processed or action items/escalations are marked resolved.

---

## 🛠 Tech Stack

* **Frontend UI:** Streamlit (Custom Dark Theme UI)
* **LLM & Audio Transcription:** Groq API (`llama-3.3-70b-versatile` & `whisper-large-v3`)
* **Vector Database (RAG):** ChromaDB & HuggingFace Embeddings (`sentence-transformers/all-MiniLM-L6-v2`)
* **Structured Database:** SQLite3 + Pandas
* **Email Automation:** Python `smtplib` & `email.mime`

---

## 📂 Project Architecture

```plaintext
├── app.py                 # Core Streamlit application & backend pipelines
├── meeting_intel.db       # SQLite relational database (Auto-generated)
├── chroma_meeting_db/     # Chroma Vector DB persist directory (Auto-generated)
├── .env                   # Local environment credentials & API keys
└── README.md              # Project documentation
