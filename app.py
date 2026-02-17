# app.py
from flask import Flask, render_template, request, jsonify
from nlp import CollegeChatbot
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from waitress import serve

app = Flask(__name__)

bot = CollegeChatbot(
    college_name="Stanley College of Engineering and Technology for Women"
)


@app.route("/")
def index():
    return render_template("chatbot.html", college_name=bot.college_name)


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

    name = data.get("name")
    phone = data.get("phone")
    email = data.get("email")
    course = data.get("course")
    message = data.get("message")

    
    college_email = "admissions@stanleycollege.ac.in"

    sender_email = "YOUR_GMAIL@gmail.com"
    sender_password = "YOUR_APP_PASSWORD"

    subject = "New Admission Enquiry from Website Chatbot"

    body = f"""
    New Admission Enquiry Received:

    Name: {name}
    Phone: {phone}
    Email: {email}
    Course Interested: {course}

    Message:
    {message}
    """

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = college_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()

        return jsonify({"status": "success"})

    except Exception as e:
        print(e)
        return jsonify({"status": "error"})

if __name__ == "__main__":
    print("Starting production server on http://localhost:8000")
    serve(app, host="0.0.0.0", port=8000, threads=8)
