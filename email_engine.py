import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

import streamlit as st

from config import SENDER_EMAIL, APP_PASSWORD


# ══════════════════════════════════════════
# CORE SEND FUNCTION
# ══════════════════════════════════════════

def send_email_alert(subject: str, body: str, to_email: str,
                     attachment=None, attachment_name: str = "") -> bool:
    if not SENDER_EMAIL or not APP_PASSWORD:
        st.error("Email configuration missing. Please verify GMAIL_USER and GMAIL_PASS in your environment.")
        return False
    if not to_email or to_email == "manager@example.com":
        st.warning("Skipping email alert: Please provide a valid recipient email address.")
        return False

    msg = MIMEMultipart()
    msg['From']    = SENDER_EMAIL
    msg['To']      = to_email
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


# ══════════════════════════════════════════
# EMAIL BODY BUILDERS
# ══════════════════════════════════════════

def generate_meeting_summary_email_body(intel: dict) -> str:
    title = intel.get("meeting_title", "Untitled Meeting")
    body  = f"Hello,\n\nA new meeting context has been ingested into MeetingIQ.\n\n"
    body += f"📌 MEETING TITLE: {title}\n"
    body += "═" * 40 + "\n\n"

    sections = [
        ("🗂 PROJECTS & INITIATIVES",    "projects",     ["name", "status", "description"]),
        ("✅ EXTRACTED ACTION ITEMS",     "action_items", ["task", "owner", "deadline", "priority"]),
        ("🚨 ESCALATIONS LOGGED",        "escalations",  ["issue", "raised_by", "assigned_to", "severity"]),
        ("⚠️ RISK REGISTER ENTRIES",     "risks",        ["description", "impact", "teams_involved", "severity"]),
        ("🎯 CRITICAL DECISIONS LOGGED", "decisions",    ["decision", "rationale", "decision_maker"]),
    ]
    for label, key, fields in sections:
        items = intel.get(key, [])
        body += f"{label}:\n"
        if not items:
            body += "  • None detected\n"
        else:
            for i, item in enumerate(items, 1):
                body += f"  {i}. "
                details = [f"{f.replace('_',' ').title()}: {item.get(f,'N/A')}" for f in fields if item.get(f)]
                body += " | ".join(details) + "\n"
        body += "\n"

    body += "This is an automated operational notification generated via MeetingIQ."
    return body