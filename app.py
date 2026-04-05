# app.py
from flask import Flask, render_template, request, jsonify
from nlp import CollegeChatbot
import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from waitress import serve
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

app = Flask(__name__)

bot = CollegeChatbot(
    college_name="Stanley College of Engineering and Technology for Women"
)


@app.route("/")
def index():
    return render_template("index.html", college_name=bot.college_name)


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    user_message = (data.get("message") or "").strip()

    if not user_message:
        return jsonify({"reply": "Please type a question.", "intent": "empty", "confidence": 0.0})

    reply, intent, confidence = bot.get_response(user_message)
    return jsonify(
        {
            "reply": reply,
            "intent": intent,
            "confidence": confidence
        }
    )


@app.route("/admission-form", methods=["POST"])
def admission_form():
    data = request.get_json()

    # Get all form fields
    name     = data.get("name", "N/A")
    phone    = data.get("phone", "N/A")
    email    = data.get("email", "N/A")
    city     = data.get("city", "N/A")
    course   = data.get("course", "N/A")
    tenth    = data.get("tenth", "N/A")
    twelfth  = data.get("twelfth", "N/A")
    exam     = data.get("exam", "N/A")
    source   = data.get("source", "N/A")
    message  = data.get("message", "N/A")

    # Load credentials from .env
    sender_email    = os.getenv("SENDER_EMAIL")
    sender_password = os.getenv("SENDER_PASSWORD")
    college_email   = os.getenv("COLLEGE_EMAIL", "hr@stanley.edu.in")

    subject = f"📋 New Admission Enquiry — {name} ({course})"

    body = f"""
New Admission Enquiry Received via Stanley College Chatbot
===========================================================

👤 PERSONAL DETAILS
--------------------
Name    : {name}
Phone   : {phone}
Email   : {email}
City    : {city}

📚 ACADEMIC DETAILS
--------------------
Course Interested : {course}
10th Percentage   : {tenth}%
12th Percentage   : {twelfth}%
Entrance Exam     : {exam}

📣 HOW DID THEY HEAR
---------------------
Source  : {source}

💬 MESSAGE
-----------
{message}

===========================================================
This enquiry was submitted via the Stanley College chatbot.
    """

    msg = MIMEMultipart()
    msg["From"]    = sender_email
    msg["To"]      = college_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        logging.info(f"✅ Admission email sent for {name}")
        return jsonify({"status": "success"})

    except Exception as e:
        logging.error(f"❌ Email error: {e}")
        return jsonify({"status": "error", "message": str(e)})


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)