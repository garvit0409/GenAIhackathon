import os
import sqlite3
from datetime import datetime

import streamlit as st
import pandas as pd

from config import GROQ_API_KEY, INTEL_DB
from database import init_db, get_dashboard_stats, get_table_df, update_status
from resources import init_resources, init_vector_store
from email_engine import send_email_alert, generate_meeting_summary_email_body
from parsing import parse_uploaded_file
from duplicate_engine import (
    get_duplicate_clusters,
    get_escalation_recurrence_timeline,
)
from ai_extraction import extract_intelligence, store_intelligence, answer_nl_query

# ══════════════════════════════════════════
# STARTUP CHECKS
# ══════════════════════════════════════════
if not GROQ_API_KEY:
    st.error("Missing GROQ_API_KEY in .env file.")
    st.stop()

init_db()

# Pre-warm cached resources so they're ready for all pages
embeddings, llm, groq_audio_client = init_resources()
vector_store = init_vector_store()


# ══════════════════════════════════════════
# PAGE CONFIG & GLOBAL STYLES
# ══════════════════════════════════════════
st.set_page_config(
    page_title="MeetingIQ — Intelligence Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
.metric-card.purple .metric-value { color: #c084fc; }
.intel-card { background: #1a1d2e; border: 1px solid #2a2d3e; border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 0.75rem; border-left: 4px solid #6366f1; }
.intel-card.high     { border-left-color: #f87171; }
.intel-card.medium   { border-left-color: #fbbf24; }
.intel-card.low      { border-left-color: #34d399; }
.intel-card.critical { border-left-color: #dc2626; background: #1f1a2e; }
.intel-card.recurring { border-left-color: #c084fc; background: #1d1a2e; }
.section-header { font-size: 1.1rem; font-weight: 600; color: #e2e8f0; margin-bottom: 1rem; padding-bottom: 0.5rem; border-bottom: 1px solid #2a2d3e; display: flex; align-items: center; gap: 8px; }
.badge { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.72rem; font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase; }
.badge-red    { background: #3f1a1a; color: #f87171; }
.badge-yellow { background: #3f2f0a; color: #fbbf24; }
.badge-green  { background: #0a3f2a; color: #34d399; }
.badge-blue   { background: #0a1f3f; color: #60a5fa; }
.badge-purple { background: #1f0a3f; color: #a78bfa; }
.badge-pink   { background: #2d0a2e; color: #e879f9; }
.chat-user { background: #1e293b; border-radius: 12px 12px 2px 12px; padding: 0.8rem 1rem; margin: 0.5rem 0; max-width: 75%; margin-left: auto; border: 1px solid #2a3f5f; color: #e2e8f0; }
.chat-assistant { background: #1a1d2e; border-radius: 12px 12px 12px 2px; padding: 0.8rem 1rem; margin: 0.5rem 0; max-width: 85%; border: 1px solid #2a2d3e; color: #e2e8f0; }
.page-title { font-size: 1.5rem; font-weight: 700; color: #e2e8f0; margin-bottom: 1.5rem; }
.stTextArea textarea, .stTextInput input { background: #1a1d2e !important; border: 1px solid #2a2d3e !important; color: #e2e8f0 !important; border-radius: 8px !important; }
.stButton button { background: #6366f1 !important; color: white !important; border-radius: 8px !important; font-weight: 600 !important; }
.logo-text { font-size: 1.3rem; font-weight: 700; color: #e2e8f0; }
.logo-accent { color: #6366f1; }
.dup-cluster { background: #1a1428; border: 1px solid #4a2d6e; border-radius: 12px; padding: 1.2rem 1.4rem; margin-bottom: 1.2rem; }
.dup-cluster-header { font-size: 0.95rem; font-weight: 600; color: #e2e8f0; display: flex; align-items: center; gap: 8px; margin-bottom: 0.8rem; }
.dup-occurrence { background: #12101e; border: 1px solid #2d2040; border-radius: 8px; padding: 0.7rem 1rem; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 10px; }
.score-bar-bg { background: #2a2d3e; border-radius: 999px; height: 6px; flex: 1; }
.method-icon { font-size: 0.7rem; background: #2d1f4e; color: #c084fc; padding: 2px 8px; border-radius: 999px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="logo-text">Meeting<span class="logo-accent">IQ</span></div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#8892a4;font-size:0.78rem;margin-bottom:1.5rem;">Organizational Intelligence Platform</div>', unsafe_allow_html=True)

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
        <div style="display:flex;justify-content:space-between;margin-bottom:4px;"><span style="color:#8892a4;font-size:0.82rem;">Recurring Issues</span><span style="color:#c084fc;font-weight:600;font-size:0.82rem;">{stats['recurring_escalations']}</span></div>
        <div style="display:flex;justify-content:space-between;margin-bottom:4px;"><span style="color:#8892a4;font-size:0.82rem;">Pending Tasks</span><span style="color:#fbbf24;font-weight:600;font-size:0.82rem;">{stats['pending_tasks']}</span></div>
        <div style="display:flex;justify-content:space-between;"><span style="color:#8892a4;font-size:0.82rem;">Critical Risks</span><span style="color:#f87171;font-weight:600;font-size:0.82rem;">{stats['critical_risks']}</span></div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["🏠  Dashboard", "➕  Ingest Meeting", "✅  Action Items", "🚨  Escalations",
         "🔄  Recurring Issues", "⚠️  Risks", "🎯  Decisions", "📋  Projects",
         "👥  Stakeholders", "💬  Query Intelligence"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown('<div style="color:#8892a4;font-size:0.72rem;">Powered by Groq LLaMA 3.3 · Whisper-Large-V3 · ChromaDB</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════
# SHARED UI HELPER
# ══════════════════════════════════════════
def render_dup_report(dup_report: list):
    if not dup_report:
        return
    st.markdown("---")
    st.markdown(f"""
    <div style="background:#1a1428;border:1px solid #6b21a8;border-radius:10px;padding:1rem 1.2rem;margin-bottom:1rem;">
        <div style="color:#c084fc;font-weight:700;font-size:1rem;margin-bottom:0.5rem;">
            🔄 Duplicate Escalation Detector — {len(dup_report)} match(es) found
        </div>
        <div style="color:#9ca3af;font-size:0.82rem;">
            The following escalations from this meeting were identified as re-occurrences of existing open issues.
            They've been linked to their originals and the recurrence counters have been updated.
        </div>
    </div>
    """, unsafe_allow_html=True)

    method_icons = {"exact": "🔗 Exact", "fuzzy": "〜 Fuzzy", "semantic": "🧠 Semantic"}

    for d in dup_report:
        pct = int(d['similarity'] * 100)
        method_label = method_icons.get(d["method"], d["method"])
        st.markdown(f"""
        <div style="background:#12101e;border:1px solid #4a2d6e;border-radius:10px;padding:1rem 1.2rem;margin-bottom:0.7rem;">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:12px;">
                <div>
                    <div style="color:#e2e8f0;font-size:0.88rem;font-weight:600;">🆕 This meeting: "{d['issue'][:90]}"</div>
                    <div style="color:#9ca3af;font-size:0.8rem;margin-top:4px;">
                        ↳ Matched to existing <span style="color:#a78bfa;font-weight:600;">[ID #{d['canonical_id']}]</span>: "{d['canonical_issue'][:90]}"
                    </div>
                </div>
                <div style="flex-shrink:0;text-align:right;">
                    <div style="color:#c084fc;font-weight:700;font-size:1.1rem;">{pct}%</div>
                    <div style="color:#6b7280;font-size:0.7rem;">match</div>
                </div>
            </div>
            <div style="display:flex;align-items:center;gap:10px;margin-top:10px;">
                <div style="background:#2a2d3e;border-radius:999px;height:5px;flex:1;">
                    <div style="background:{'#c084fc' if pct>=80 else '#fbbf24' if pct>=60 else '#60a5fa'};
                                width:{pct}%;height:5px;border-radius:999px;"></div>
                </div>
                <span style="background:#2d1f4e;color:#c084fc;padding:2px 8px;border-radius:999px;font-size:0.7rem;font-weight:600;">{method_label}</span>
                <span style="color:#8892a4;font-size:0.78rem;">🔁 Now raised {d['occurrences']}×</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════
if page == "🏠  Dashboard":
    st.markdown('<div class="page-title">🧠 Intelligence Dashboard</div>', unsafe_allow_html=True)

    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    kpis = [
        (col1, stats['meetings'],              "Meetings Ingested",  "info"),
        (col2, stats['projects'],              "Projects Tracked",   "info"),
        (col3, stats['pending_tasks'],         "Pending Tasks",      "warn"),
        (col4, stats['open_escalations'],      "Open Escalations",   "alert"),
        (col5, stats['recurring_escalations'], "Recurring Issues",   "purple"),
        (col6, stats['critical_risks'],        "Critical Risks",     "alert"),
        (col7, stats['decisions'],             "Decisions Logged",   "ok"),
    ]
    for col, val, label, cls in kpis:
        with col:
            st.markdown(f'<div class="metric-card {cls}"><div class="metric-value">{val}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if stats['recurring_escalations'] > 0:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#1a1428,#1e1535);border:1px solid #6b21a8;
                    border-radius:10px;padding:0.8rem 1.2rem;margin-bottom:1.2rem;
                    display:flex;align-items:center;gap:12px;">
            <span style="font-size:1.5rem;">🔄</span>
            <div>
                <span style="color:#c084fc;font-weight:700;">{stats['recurring_escalations']} persistent issue{'s' if stats['recurring_escalations']!=1 else ''}</span>
                <span style="color:#9ca3af;font-size:0.88rem;"> detected across multiple meetings — these unresolved blockers keep resurfacing.</span>
                <span style="color:#c084fc;font-size:0.82rem;cursor:pointer;"> → View Recurring Issues tab</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1.1, 0.9])

    with col_left:
        st.markdown('<div class="section-header">🚨 Recent Escalations</div>', unsafe_allow_html=True)
        df_esc = get_table_df("escalations", 5)
        if df_esc.empty:
            st.info("No escalations logged yet.")
        else:
            for _, row in df_esc.iterrows():
                sev   = str(row.get("Severity","")).lower()
                times = int(row.get("Times Raised", 1) or 1)
                is_dup = row.get("Duplicate Of") is not None and str(row.get("Duplicate Of","")) not in ["", "None", "nan"]
                is_recurring = (times > 1) and not is_dup

                cls = "critical" if sev=="critical" else ("high" if sev=="high" else ("medium" if sev=="medium" else "low"))
                if is_recurring: cls = "recurring"
                badge_cls = "badge-red" if sev in ["critical","high"] else ("badge-yellow" if sev=="medium" else "badge-green")

                recurring_html = ""
                if is_recurring:
                    recurring_html = f'<span class="badge badge-purple" style="font-size:0.68rem;margin-right:4px;">🔄 {times}× Recurring</span>'
                elif is_dup:
                    score = row.get("Similarity")
                    score_str = f" {int(float(score)*100)}%" if score and str(score) not in ["","None","nan"] else ""
                    recurring_html = f'<span class="badge badge-pink" style="font-size:0.68rem;margin-right:4px;">↩ Duplicate{score_str}</span>'

                st.markdown(f"""
                <div class="intel-card {cls}">
                    <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:8px;">
                        <div style="color:#e2e8f0;font-weight:500;font-size:0.88rem;">{recurring_html}{str(row.get('Issue',''))[:80]}</div>
                        <span class="badge {badge_cls}">{row.get('Severity','')}</span>
                    </div>
                    <div style="color:#8892a4;font-size:0.78rem;margin-top:6px;">Raised by <strong style="color:#a5b4fc">{row.get('Raised By','—')}</strong> · {str(row.get('Meeting',''))[:40]}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-header" style="margin-top:1.5rem;">⚠️ Active Risks</div>', unsafe_allow_html=True)
        df_risks = get_table_df("risks", 4)
        if df_risks.empty:
            st.info("No risks logged yet.")
        else:
            for _, row in df_risks.iterrows():
                sev = str(row.get("Severity","")).lower()
                cls = "critical" if sev=="critical" else ("high" if sev=="high" else "medium")
                st.markdown(f"""
                <div class="intel-card {cls}">
                    <div style="color:#e2e8f0;font-weight:500;font-size:0.88rem;">{str(row.get('Risk',''))[:80]}</div>
                    <div style="color:#8892a4;font-size:0.78rem;margin-top:4px;">Impact: {str(row.get('Impact',''))[:60]} · Teams: <strong style="color:#a5b4fc">{str(row.get('Teams',''))[:40]}</strong></div>
                </div>""", unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-header">✅ Pending Action Items</div>', unsafe_allow_html=True)
        df_ai      = get_table_df("action_items", 8)
        df_pending = df_ai[df_ai["Status"] == "Pending"] if not df_ai.empty else pd.DataFrame()
        if df_pending.empty:
            st.info("All tasks cleared! 🎉")
        else:
            for _, row in df_pending.iterrows():
                pri = str(row.get("Priority","")).lower()
                badge_cls = "badge-red" if pri=="high" else ("badge-yellow" if pri=="medium" else "badge-green")
                dl = row.get("Deadline","")
                st.markdown(f"""
                <div class="intel-card {'high' if pri=='high' else ('medium' if pri=='medium' else 'low')}">
                    <div style="color:#e2e8f0;font-size:0.86rem;font-weight:500;">{str(row.get('Task',''))[:70]}</div>
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
                    <div style="color:#e2e8f0;font-weight:500;font-size:0.86rem;">{str(row.get('Decision',''))[:70]}</div>
                    <div style="color:#8892a4;font-size:0.78rem;margin-top:4px;">{str(row.get('Rationale',''))[:60]}{' · <strong style="color:#a5b4fc">'+str(row.get('Decision Maker',''))+'</strong>' if row.get('Decision Maker') else ''}</div>
                </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════
# PAGE: INGEST MEETING
# ══════════════════════════════════════════
elif page == "➕  Ingest Meeting":
    st.markdown('<div class="page-title">➕ Ingest Meeting</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["✍️  Paste Text", "📎  Upload File / Audio", "🎬  Example Demo"])

    with tab1:
        st.markdown("Paste a meeting summary, transcript, or discussion notes below.")
        meeting_text = st.text_area("Meeting Content", height=280,
                                    placeholder="e.g. Rahul will coordinate with backend team...",
                                    label_visibility="collapsed")
        if st.button("🧠 Extract Intelligence", use_container_width=True):
            if meeting_text.strip():
                with st.spinner("Analyzing meeting content with AI..."):
                    intel = extract_intelligence(meeting_text)
                    if intel:
                        mid, dup_report = store_intelligence(intel, meeting_text, "text_paste")
                        st.success(f"✅ Meeting ingested successfully! (ID: {mid})")
                        render_dup_report(dup_report)
                        email_body = generate_meeting_summary_email_body(intel)
                        sent = send_email_alert(
                            subject=f"[MeetingIQ Summary] {intel.get('meeting_title','New Meeting')}",
                            body=email_body, to_email=manager_email,
                        )
                        if sent:
                            st.success("📩 Dashboard summary email sent to Manager!")
                        st.balloons()
                        st.markdown("### 📊 Extracted Intelligence Preview")
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.metric("Projects",     len(intel.get("projects",[])))
                            st.metric("Action Items", len(intel.get("action_items",[])))
                        with c2:
                            st.metric("Escalations",         len(intel.get("escalations",[])))
                            st.metric("↩ Duplicates Caught", len(dup_report))
                        with c3:
                            st.metric("Decisions",    len(intel.get("decisions",[])))
                            st.metric("Stakeholders", len(intel.get("stakeholders",[])))
            else:
                st.warning("Please enter some meeting content first.")

    with tab2:
        st.markdown("Upload documents or meeting audio recordings here.")
        uploaded = st.file_uploader("Upload meeting document or audio file",
                                    type=["txt","pdf","docx","mp3","mp4","mpeg","mpga","m4a","wav","webm"])
        if uploaded:
            name_lower = uploaded.name.lower()
            if any(name_lower.endswith(ext) for ext in [".mp3",".mp4",".mpeg",".mpga",".m4a",".wav",".webm"]):
                st.audio(uploaded, format='audio/wav')
            with st.spinner("Parsing file..."):
                content = parse_uploaded_file(uploaded)
            if content:
                st.success(f"Successfully processed `{uploaded.name}`!")
                with st.expander("Preview parsed content"):
                    st.text(content[:1200] + ("..." if len(content) > 1200 else ""))
                if st.button("🧠 Extract Intelligence from File", use_container_width=True):
                    with st.spinner("Analyzing with AI..."):
                        intel = extract_intelligence(content)
                        if intel:
                            mid, dup_report = store_intelligence(intel, content, f"file:{uploaded.name}")
                            st.success(f"✅ Ingested! Meeting ID: {mid}")
                            render_dup_report(dup_report)
                            email_body = generate_meeting_summary_email_body(intel)
                            send_email_alert(
                                subject=f"[MeetingIQ File Summary] {intel.get('meeting_title','New File Upload')}",
                                body=email_body, to_email=manager_email,
                            )
            else:
                st.error("Could not parse file content.")

    with tab3:
        st.markdown("**Try with this illustrative scenario from the problem statement:**")
        demo_text = """Team: Engineering + Product sync — Q3 Planning\nDate: Today\n\nThe payment integration project is significantly delayed because the Vendor API has been highly unstable over the past week, causing repeated failures in our staging environment. \n\nRahul (Backend Lead) will coordinate directly with the backend team and the vendor's technical contact before this Friday to resolve authentication issues. If this issue continues beyond Friday, it may critically impact the Phase-2 release scheduled for next month.\n\nPriya (Product Manager) escalated the concern formally to leadership, flagging this as a high-priority blocker.\n\nAdditionally, Amit confirmed that the new dashboard feature is on track for deployment by end of month. The design team (led by Sara) needs to finalize UI mockups by Wednesday so the frontend team can begin integration.\n\nDecision made: The team agreed to delay Phase-2 launch by two weeks to buffer for the API instability. This was decided by CTO Rajesh after reviewing risk scenarios.\n\nSarah raised a risk that if the backend delay extends, it will also impact the mobile team's release cycle since they depend on the same payment APIs.\n\nKey stakeholders: Rahul (Backend), Priya (Product), Amit (Frontend), Sara (Design), Rajesh (CTO)"""
        st.text_area("Demo meeting content:", value=demo_text, height=200, disabled=True)
        if st.button("🧠 Run Demo Extraction", use_container_width=True):
            with st.spinner("Extracting intelligence..."):
                intel = extract_intelligence(demo_text)
                if intel:
                    mid, dup_report = store_intelligence(intel, demo_text, "demo")
                    st.success(f"✅ Demo meeting ingested! (ID: {mid})")
                    render_dup_report(dup_report)
                    email_body = generate_meeting_summary_email_body(intel)
                    send_email_alert(
                        subject=f"[MeetingIQ Demo Summary] {intel.get('meeting_title')}",
                        body=email_body, to_email=manager_email,
                    )


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
        with col1: filter_status   = st.selectbox("Status",   ["All","Pending","Done"])
        with col2: filter_priority = st.selectbox("Priority", ["All","High","Medium","Low"])
        with col3: filter_owner    = st.text_input("Filter by owner", placeholder="e.g. Rahul")

        filtered = df.copy()
        if filter_status   != "All": filtered = filtered[filtered["Status"]   == filter_status]
        if filter_priority != "All": filtered = filtered[filtered["Priority"] == filter_priority]
        if filter_owner:             filtered = filtered[filtered["Owner"].str.contains(filter_owner, case=False, na=False)]

        for _, row in filtered.iterrows():
            pri = str(row.get("Priority","")).lower()
            badge_cls = "badge-red" if pri=="high" else ("badge-yellow" if pri=="medium" else "badge-green")
            card_cls  = pri if pri in ["high","medium","low"] else ""
            status_badge = "badge-green" if row.get("Status")=="Done" else "badge-yellow"
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
                        body_ai = (f"Hello Manager,\n\nAction item resolved.\n\n"
                                   f"• Task: {row.get('Task')}\n• Owner: {row.get('Owner','Unassigned')}\n"
                                   f"• Meeting: {row.get('Meeting')}\n• Resolved: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
                        send_email_alert("[Task Completed Alert] Action Item Resolved!", body_ai, manager_email)
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
        col1, col2, col3 = st.columns(3)
        with col1: filter_sev    = st.selectbox("Severity", ["All","Critical","High","Medium","Low"])
        with col2: filter_status = st.selectbox("Status",   ["All","Open","Resolved"])
        with col3: filter_type   = st.selectbox("Type",     ["All","Original","Duplicate"])

        filtered = df.copy()
        if filter_sev    != "All": filtered = filtered[filtered["Severity"] == filter_sev]
        if filter_status != "All": filtered = filtered[filtered["Status"]   == filter_status]
        if filter_type   == "Original":
            filtered = filtered[filtered["Duplicate Of"].isna() | (filtered["Duplicate Of"].astype(str).isin(["","None","nan"]))]
        elif filter_type == "Duplicate":
            filtered = filtered[~(filtered["Duplicate Of"].isna() | (filtered["Duplicate Of"].astype(str).isin(["","None","nan"])))]

        for _, row in filtered.iterrows():
            sev   = str(row.get("Severity","")).lower()
            times = int(row.get("Times Raised",1) or 1)
            is_dup = not (str(row.get("Duplicate Of","")) in ["","None","nan"])
            is_recurring = (times > 1) and not is_dup

            cls = "critical" if sev=="critical" else ("high" if sev=="high" else ("medium" if sev=="medium" else "low"))
            if is_recurring: cls = "recurring"
            badge_cls = "badge-red" if sev in ["critical","high"] else ("badge-yellow" if sev=="medium" else "badge-green")

            extra_badges = ""
            if is_recurring:
                extra_badges += f'<span class="badge badge-purple">🔄 {times}× Recurring</span> '
            if is_dup:
                score = row.get("Similarity")
                score_str = f" {int(float(score)*100)}%" if score and str(score) not in ["","None","nan"] else ""
                extra_badges += f'<span class="badge badge-pink">↩ Dup of #{int(float(row.get("Duplicate Of")))}{score_str}</span> '

            st.markdown(f"""
            <div class="intel-card {cls}">
                <div style="display:flex;justify-content:space-between;gap:8px;align-items:flex-start;">
                    <div>
                        <div style="margin-bottom:4px;">{extra_badges}</div>
                        <div style="color:#e2e8f0;font-weight:600;font-size:0.9rem;">{row.get('Issue','')}</div>
                    </div>
                    <span class="badge {badge_cls}">{row.get('Severity','')}</span>
                </div>
                <div style="color:#8892a4;font-size:0.78rem;margin-top:6px;display:flex;gap:12px;flex-wrap:wrap;">
                    <span>🔴 Raised by <strong style="color:#f87171">{row.get('Raised By','')}</strong></span>
                    {'<span>➡️ Assigned to <strong style="color:#a5b4fc">'+str(row.get('Assigned To',''))+'</strong></span>' if row.get('Assigned To') else ''}
                    <span>📁 {str(row.get('Meeting',''))[:40]}</span>
                    <span class="badge {'badge-yellow' if row.get('Status')=='Open' else 'badge-green'}">{row.get('Status','')}</span>
                    {('<span style="color:#8892a4;">🔍 via '+str(row.get('Detected Via',''))+'</span>') if is_dup and row.get('Detected Via') else ''}
                </div>
            </div>""", unsafe_allow_html=True)

            if row.get("Status") == "Open":
                if st.button("Mark Resolved", key=f"res_{row['id']}"):
                    update_status("escalations", int(row["id"]), "Resolved")
                    body_esc = (f"Hello Manager,\n\nEscalation closed.\n\n"
                                f"• Issue: {row.get('Issue')}\n• Severity: {row.get('Severity')}\n"
                                f"• Raised By: {row.get('Raised By')}\n• Meeting: {row.get('Meeting')}")
                    send_email_alert("[Escalation Resolved Alert] Issue Closed!", body_esc, manager_email)
                    st.rerun()


# ══════════════════════════════════════════
# PAGE: RECURRING ISSUES
# ══════════════════════════════════════════
elif page == "🔄  Recurring Issues":
    st.markdown('<div class="page-title">🔄 Recurring Escalation Clusters</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#1a1428;border:1px solid #4a2d6e;border-radius:10px;padding:1rem 1.2rem;margin-bottom:1.5rem;">
        <div style="color:#c084fc;font-weight:600;font-size:0.92rem;margin-bottom:4px;">How the detection engine works</div>
        <div style="color:#9ca3af;font-size:0.82rem;line-height:1.6;">
            Every escalation ingested is checked against all open originals using a <strong style="color:#e2e8f0;">three-pass algorithm</strong>:
            <br>
            <span style="background:#1f0a3f;color:#a78bfa;padding:1px 6px;border-radius:4px;font-size:0.76rem;">🔗 Pass 1 — Exact/Substring</span> &nbsp;
            <span style="background:#1f0a3f;color:#a78bfa;padding:1px 6px;border-radius:4px;font-size:0.76rem;">〜 Pass 2 — Fuzzy (difflib ≥75%)</span> &nbsp;
            <span style="background:#1f0a3f;color:#a78bfa;padding:1px 6px;border-radius:4px;font-size:0.76rem;">🧠 Pass 3 — TF-IDF Semantic (≥55%)</span>
            <br><br>
            Matches are stored as <em>duplicates</em> linked to the original. The occurrence counter climbs each time.
            Severity auto-escalates if a newer re-raise has a higher severity tier.
        </div>
    </div>
    """, unsafe_allow_html=True)

    clusters = get_duplicate_clusters()

    if not clusters:
        st.markdown("""
        <div style="text-align:center;padding:3rem;color:#4b5563;">
            <div style="font-size:2.5rem;margin-bottom:0.5rem;">✅</div>
            <div style="font-size:1rem;font-weight:500;color:#6b7280;">No recurring escalations detected yet.<br>
            Ingest more meetings with overlapping issues to see clusters here.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="color:#9ca3af;font-size:0.88rem;margin-bottom:1.2rem;">{len(clusters)} recurring issue cluster(s) found</div>', unsafe_allow_html=True)

        method_icons = {"exact": "🔗 Exact Match", "fuzzy": "〜 Fuzzy Match", "semantic": "🧠 Semantic Match"}
        sev_color    = {"critical":"#dc2626","high":"#f87171","medium":"#fbbf24","low":"#34d399"}

        for cluster in clusters:
            canon  = cluster["canonical"]
            dupes  = cluster["duplicates"]
            sev    = str(canon.get("severity","")).lower()
            s_col  = sev_color.get(sev, "#6366f1")
            times  = int(canon.get("occurrence_count",1))
            status = str(canon.get("status",""))

            timeline_df = get_escalation_recurrence_timeline(int(canon["id"]))

            st.markdown(f"""
            <div class="dup-cluster">
                <div class="dup-cluster-header">
                    <span style="background:#3b0764;color:#c084fc;padding:3px 10px;border-radius:999px;font-size:0.78rem;font-weight:700;">
                        🔄 {times}× across {len(dupes)+1} meeting(s)
                    </span>
                    <span style="background:{s_col}22;color:{s_col};padding:3px 10px;border-radius:999px;font-size:0.78rem;font-weight:700;">
                        {canon.get('severity','').upper()}
                    </span>
                    <span style="background:{'#0a3f2a' if status=='Resolved' else '#3f2f0a'};color:{'#34d399' if status=='Resolved' else '#fbbf24'};
                                 padding:3px 10px;border-radius:999px;font-size:0.78rem;font-weight:700;">
                        {status}
                    </span>
                </div>
                <div style="color:#e2e8f0;font-weight:600;font-size:0.95rem;margin-bottom:0.6rem;">
                    📌 "{canon.get('issue','')}"
                </div>
                <div style="color:#8892a4;font-size:0.78rem;margin-bottom:1rem;">
                    First raised by <strong style="color:#a5b4fc">{canon.get('raised_by','—')}</strong> in
                    <strong style="color:#e2e8f0">{str(canon.get('meeting',''))[:50]}</strong>
                    <span style="margin-left:8px;color:#6b7280;">({canon.get('first_seen','')})</span>
                </div>
            """, unsafe_allow_html=True)

            st.markdown('<div style="color:#9ca3af;font-size:0.78rem;font-weight:600;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.05em;">Recurrence Timeline</div>', unsafe_allow_html=True)

            if not timeline_df.empty:
                for idx, (_, t_row) in enumerate(timeline_df.iterrows()):
                    is_original = str(t_row.get("Duplicate Of","")) in ["","None","nan"]
                    score       = t_row.get("Similarity")
                    score_str   = f"{int(float(score)*100)}%" if score and str(score) not in ["","None","nan"] else "—"
                    method_raw  = str(t_row.get("Detected Via","")) or ""
                    method_str  = method_icons.get(method_raw, method_raw)

                    tag_html = ""
                    if is_original:
                        tag_html = '<span style="background:#0a1f3f;color:#60a5fa;padding:1px 8px;border-radius:4px;font-size:0.7rem;">ORIGINAL</span>'
                    else:
                        tag_html = f'<span style="background:#2d1f4e;color:#c084fc;padding:1px 8px;border-radius:4px;font-size:0.7rem;">{method_str} · {score_str}</span>'

                    connector = '<div style="width:1px;height:10px;background:#3a2d5e;margin:2px 0 2px 18px;"></div>' if idx < len(timeline_df)-1 else ''

                    st.markdown(f"""
                    <div style="background:#12101e;border:1px solid #2d2040;border-radius:8px;
                                padding:0.6rem 1rem;margin-bottom:0px;display:flex;align-items:center;gap:12px;">
                        <div style="width:8px;height:8px;border-radius:50%;background:{'#60a5fa' if is_original else '#c084fc'};flex-shrink:0;"></div>
                        <div style="flex:1;">
                            <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
                                {tag_html}
                                <span style="color:#e2e8f0;font-size:0.82rem;">{str(t_row.get('Meeting',''))[:55]}</span>
                            </div>
                            <div style="color:#6b7280;font-size:0.74rem;margin-top:2px;">
                                {str(t_row.get('Date',''))[:16]} · Raised by {t_row.get('Raised By','—')}
                            </div>
                        </div>
                        <div style="flex-shrink:0;">
                            <span style="color:#{'60a5fa' if is_original else '9ca3af'};font-size:0.8rem;font-weight:600;">#{t_row.get('id','')}</span>
                        </div>
                    </div>
                    {connector}
                    """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

            if status == "Open":
                canonical_id = int(canon["id"])
                if st.button(f"✅ Resolve entire cluster #{canonical_id}", key=f"cluster_resolve_{canonical_id}"):
                    conn = sqlite3.connect(INTEL_DB)
                    conn.execute("UPDATE escalations SET status='Resolved' WHERE id=?", (canonical_id,))
                    conn.execute("UPDATE escalations SET status='Resolved' WHERE duplicate_of=?", (canonical_id,))
                    conn.commit()
                    conn.close()
                    body_cluster = (
                        f"Hello Manager,\n\nA recurring escalation cluster has been fully resolved.\n\n"
                        f"• Issue: {canon.get('issue')}\n• Total recurrences: {times}\n"
                        f"• Severity: {canon.get('severity')}\n• First raised by: {canon.get('raised_by','—')}"
                    )
                    send_email_alert(
                        f"[Recurring Issue Resolved] Cluster #{canonical_id} Closed",
                        body_cluster, manager_email,
                    )
                    st.success(f"Cluster #{canonical_id} and all {len(dupes)} duplicate(s) marked Resolved!")
                    st.rerun()

            st.markdown("<br>", unsafe_allow_html=True)

        # ── Summary analytics ──────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="section-header">📊 Detection Method Breakdown</div>', unsafe_allow_html=True)

        conn = sqlite3.connect(INTEL_DB)
        df_methods = pd.read_sql("""
            SELECT detection_method as Method, COUNT(*) as Count
            FROM escalations WHERE duplicate_of IS NOT NULL
            GROUP BY detection_method
        """, conn)
        df_top = pd.read_sql("""
            SELECT e.issue as Issue, e.occurrence_count as Occurrences,
                   e.severity as Severity, e.status as Status,
                   m.title as 'First Seen In'
            FROM escalations e JOIN meetings m ON e.meeting_id=m.id
            WHERE e.duplicate_of IS NULL AND e.occurrence_count > 1
            ORDER BY e.occurrence_count DESC LIMIT 10
        """, conn)
        conn.close()

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Duplicates by detection pass**")
            if not df_methods.empty:
                st.dataframe(df_methods, use_container_width=True, hide_index=True)
            else:
                st.info("No duplicates detected yet.")
        with col_b:
            st.markdown("**Most-recurring issues (top 10)**")
            if not df_top.empty:
                st.dataframe(df_top, use_container_width=True, hide_index=True)
            else:
                st.info("No recurring issues yet.")


# ══════════════════════════════════════════
# PAGE: RISKS
# ══════════════════════════════════════════
elif page == "⚠️  Risks":
    st.markdown('<div class="page-title">⚠️ Risk Register</div>', unsafe_allow_html=True)
    df = get_table_df("risks", 200)
    if df.empty:
        st.info("No risks logged yet.")
    else:
        filter_sev = st.selectbox("Filter by Severity", ["All","Critical","High","Medium","Low"])
        filtered   = df if filter_sev == "All" else df[df["Severity"] == filter_sev]
        for _, row in filtered.iterrows():
            sev = str(row.get("Severity","")).lower()
            cls = "critical" if sev=="critical" else ("high" if sev=="high" else ("medium" if sev=="medium" else "low"))
            badge_cls = "badge-red" if sev in ["critical","high"] else ("badge-yellow" if sev=="medium" else "badge-green")
            st.markdown(f"""
            <div class="intel-card {cls}">
                <div style="display:flex;justify-content:space-between;gap:8px;">
                    <div style="color:#e2e8f0;font-weight:500;font-size:0.9rem;">{row.get('Risk','')}</div>
                    <span class="badge {badge_cls}">{row.get('Severity','')}</span>
                </div>
                <div style="color:#8892a4;font-size:0.78rem;margin-top:6px;"><strong style="color:#e2e8f0">Impact:</strong> {str(row.get('Impact',''))[:80]}</div>
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
                <div style="color:#8892a4;font-size:0.82rem;margin-top:6px;"><strong style="color:#e2e8f0">Rationale:</strong> {str(row.get('Rationale',''))[:100]}</div>
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
            cls = "high" if any(w in status for w in ["delay","block","risk"]) else ("ok" if any(w in status for w in ["track","complete"]) else "")
            st.markdown(f"""
            <div class="intel-card {cls}">
                <div style="display:flex;justify-content:space-between;gap:8px;">
                    <div style="color:#e2e8f0;font-weight:600;font-size:0.9rem;">🗂 {row.get('Project','')}</div>
                    <span class="badge badge-blue">{row.get('Status','')}</span>
                </div>
                <div style="color:#8892a4;font-size:0.82rem;margin-top:6px;">{str(row.get('Description',''))[:120]}</div>
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
        "What are all open escalations?",
        "Show pending tasks assigned to Rahul",
        "Which projects are at risk?",
        "List all high-priority blockers",
        "What decisions were made about the payment integration?",
        "Which issues have recurred across multiple meetings?",
    ]
    cols = st.columns(3)
    for i, s in enumerate(suggestions):
        with cols[i % 3]:
            if st.button(s, key=f"sug_{i}", use_container_width=True):
                if "chat_messages" not in st.session_state:
                    st.session_state.chat_messages = []
                st.session_state.chat_messages.append({"role": "user", "content": s})
                with st.spinner("Analyzing intelligence database..."):
                    answer = answer_nl_query(s, st.session_state.chat_messages[:-1])
                st.session_state.chat_messages.append({"role": "assistant", "content": answer})
                st.rerun()

    st.markdown("---")
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

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