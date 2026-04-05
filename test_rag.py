# test_rag.py
# Run from ROOT directory only: pytest test_rag.py -v
# Install deps: pip install pytest

import pytest
from rag.rag_engine import RAGEngine
from nlp import CollegeChatbot

# ── shared instances ──────────────────────────────────────────────────────────
rag = RAGEngine()
bot = CollegeChatbot(college_name="Stanley College of Engineering and Technology for Women")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — RAG ENGINE  (your original smoke tests, now with assertions)
# ─────────────────────────────────────────────────────────────────────────────

def test_rag_eligibility():
    """RAG should return a non-empty answer about B.Tech eligibility."""
    answer, intent, conf = rag.ask("What is the eligibility for B.Tech admission?")
    assert isinstance(answer, str) and len(answer) > 10, "RAG returned empty answer"

def test_rag_how_to_apply():
    answer, intent, conf = rag.ask("How can I apply for admission?")
    assert isinstance(answer, str) and len(answer) > 10

def test_rag_college_timings():
    answer, intent, conf = rag.ask("What are the college timings?")
    assert isinstance(answer, str) and len(answer) > 10

def test_rag_confidence_range():
    """Confidence score must always be between 0 and 1."""
    _, _, conf = rag.ask("Tell me about placements")
    assert 0.0 <= conf <= 1.0, f"Unexpected confidence: {conf}"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — SMALL TALK  (greetings, pleasantries)
# ─────────────────────────────────────────────────────────────────────────────

def test_greeting_hello():
    reply, intent, _ = bot.get_response("hello")
    assert intent == "small_talk"

def test_greeting_hi():
    reply, intent, _ = bot.get_response("hi")
    assert intent == "small_talk"

def test_thank_you():
    reply, intent, _ = bot.get_response("thank you")
    assert intent == "small_talk"

def test_trivial_ok():
    _, intent, _ = bot.get_response("ok")
    assert intent == "acknowledge"

def test_trivial_yes():
    _, intent, _ = bot.get_response("yes")
    assert intent == "acknowledge"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — KEYWORD ROUTING  (fees, hostel, courses, etc.)
# ─────────────────────────────────────────────────────────────────────────────

def test_fees_query():
    reply, intent, _ = bot.get_response("what is the fee structure?")
    assert intent == "fees"
    assert len(reply) > 10

def test_hostel_query():
    _, intent, _ = bot.get_response("is hostel available?")
    assert intent == "hostel"

def test_courses_query():
    _, intent, _ = bot.get_response("what courses are offered?")
    assert intent == "courses_offered"

def test_placement_query():
    _, intent, _ = bot.get_response("tell me about placements")
    assert intent == "placements"

def test_scholarship_query():
    _, intent, _ = bot.get_response("are there any scholarships?")
    assert intent == "scholarships"

def test_library_query():
    _, intent, _ = bot.get_response("does the college have a library?")
    assert intent == "library"

def test_contact_query():
    _, intent, _ = bot.get_response("what is the contact number?")
    assert intent == "contact"

def test_timing_query():
    _, intent, _ = bot.get_response("what are the college timings?")
    assert intent == "timings"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — OUT OF DOMAIN  (should reject non-college queries)
# ─────────────────────────────────────────────────────────────────────────────

def test_out_of_domain_cricket():
    _, intent, _ = bot.get_response("who won the IPL 2024?")
    assert intent == "out_of_domain"

def test_out_of_domain_movie():
    _, intent, _ = bot.get_response("recommend me a good movie")
    assert intent == "out_of_domain"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — EDGE CASES  (empty, symbols, very long input)
# ─────────────────────────────────────────────────────────────────────────────

def test_empty_string():
    """Empty input should not crash — bot handles it gracefully."""
    try:
        reply, intent, _ = bot.get_response("")
        assert isinstance(reply, str)
    except Exception as e:
        pytest.fail(f"Bot crashed on empty input: {e}")

def test_whitespace_only():
    try:
        reply, intent, _ = bot.get_response("     ")
        assert isinstance(reply, str)
    except Exception as e:
        pytest.fail(f"Bot crashed on whitespace input: {e}")

def test_long_input():
    long_msg = "fees " * 50
    try:
        reply, intent, _ = bot.get_response(long_msg)
        assert isinstance(reply, str)
    except Exception as e:
        pytest.fail(f"Bot crashed on long input: {e}")

def test_special_characters():
    try:
        reply, intent, _ = bot.get_response("!@#$%^&*()")
        assert isinstance(reply, str)
    except Exception as e:
        pytest.fail(f"Bot crashed on special characters: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — FLASK ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    from app import app as flask_app
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as c:
        yield c

def test_home_route(client):
    res = client.get("/")
    assert res.status_code == 200

def test_chat_api_valid(client):
    import json
    res = client.post("/api/chat",
        data=json.dumps({"message": "what are the fees?"}),
        content_type="application/json"
    )
    assert res.status_code == 200
    data = json.loads(res.data)
    assert "reply" in data
    assert "intent" in data
    assert "confidence" in data

def test_chat_api_empty(client):
    import json
    res = client.post("/api/chat",
        data=json.dumps({"message": ""}),
        content_type="application/json"
    )
    assert res.status_code == 200
    data = json.loads(res.data)
    assert data["intent"] == "empty"