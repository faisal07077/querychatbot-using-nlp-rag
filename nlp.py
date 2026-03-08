# nlp.py
import json
import os
import numpy as np
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rag.rag_engine import RAGEngine
from faculty_handler import FacultyHandler


class CollegeChatbot:

    def __init__(
        self,
        college_name: str,
        faq_path: str = "data/faqs.json",
        threshold: float = 0.70
    ):
        self.college_name = college_name
        self.faq_path = faq_path
        self.threshold = threshold
        self.faqs = self._load_faqs()
        self.questions: List[str] = [item["question"] for item in self.faqs]

        if not self.questions:
            raise ValueError("FAQ knowledge base is empty.")

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.question_vectors = self.vectorizer.fit_transform(self.questions)
        self.rag_engine = RAGEngine()
        self.faculty_handler = FacultyHandler()

    def _load_faqs(self) -> List[dict]:
        if not os.path.exists(self.faq_path):
            raise FileNotFoundError(f"FAQ file not found at path: {self.faq_path}")
        with open(self.faq_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Invalid FAQ format. Expected a list of objects.")
        return data

    def _basic_small_talk(self, message: str):
        msg = message.lower().strip()

        # Greetings - only exact/clear greetings
        greet_words = ["hello", "hi", "hey"]
        if any(msg == word or msg.startswith(word + " ") for word in greet_words):
            return (
                f"Hello! I am the virtual assistant for {self.college_name}. "
                "You can ask me about admissions, exams, syllabus, fees, or faculty."
            )

        if any(phrase in msg for phrase in [
            "how are you", "how r you", "how are u",
            "how's it going", "what's up", "how do you do",
            "how r u", "khyriyat", "namaste"
        ]):
            return (
                "I'm doing well, thank you for asking. "
                "How can I assist you with college-related information today?"
            )

        if any(phrase in msg for phrase in ["good morning", "good afternoon", "good evening"]):
            return "Good day! How can I help you with college-related queries?"

        if msg in ["thank you", "thanks", "thank u", "thankyou"]:
            return "You're welcome. Let me know if you need any college information."

        if "who are you" in msg:
            return f"I am the AI assistant for {self.college_name}."

        return None

    def _is_college_domain_query(self, message: str) -> bool:
        domain_keywords = [
            "college", "stanley", "admission", "apply", "eligibility",
            "fees", "fee", "exam", "exams", "syllabus", "course",
            "b.tech", "branch", "department", "faculty",
            "timings", "hostel", "campus", "placement",
            "contact", "office", "university", "counseling",
            "scholarship", "scholarships", "library", "clubs",
            "canteen", "wifi", "sports", "principal", "ranking",
            "dress code", "facilities", "timing", "transport",
            "professor", "lecturer", "staff", "hod", "teacher",
            "vision", "mission", "accreditation", "about",
            "naac", "nba", "grade", "accreditation", "autonomous",
            "vision", "mission", "established", "founded", "affiliated"
        ]
        msg = message.lower()
        return any(keyword in msg for keyword in domain_keywords)

    def _handle_faculty_query(self, msg: str):
        """Handle faculty queries - only triggered by explicit faculty keywords or Dr/Mrs/Ms titles."""

        faculty_words = ["faculty", "lecturer", "staff", "hod", "head of department"]
        is_faculty_keyword = any(word in msg for word in faculty_words)

        # Check for title-based name search
        has_title = any(title in msg for title in ["dr.", "dr ", "mrs.", "mrs ", "ms.", "ms ", "mr.", "mr "])

        # Count/list queries
        if is_faculty_keyword:
            if any(word in msg for word in ["hod", "head of department", "head"]):
                return self.faculty_handler.get_hod()
            if any(word in msg for word in ["how many", "count", "total", "number"]):
                return self.faculty_handler.get_count()
            if any(word in msg for word in ["professor"]) and "assistant" not in msg and "associate" not in msg:
                return self.faculty_handler.get_professors()
            return self.faculty_handler.get_all_cse_faculty()

        # Name search only with title
        if has_title:
            result = self.faculty_handler.search_by_name(msg)
            if result:
                return result

        return None

    def get_response(self, user_message: str) -> Tuple[str, str, float]:

        msg = user_message.lower().strip()

        # Step 1: Ignore trivial inputs
        ignore_words = ["ok", "okay", "hmm", "yes", "no", "fine", "good"]
        if msg in ignore_words:
            return (
                "Acknowledged. You may ask about admissions, exams, syllabus, fees, or faculty.",
                "acknowledge",
                1.0
            )

        # Step 2: Small talk
        small_talk = self._basic_small_talk(user_message)
        if small_talk:
            return small_talk, "small_talk", 1.0

        # Step 2.5: Faculty queries
        faculty_response = self._handle_faculty_query(msg)
        if faculty_response:
            return faculty_response, "faculty", 1.0

        # Step 3: Keyword routing
        if any(word in msg for word in ["fee", "fees", "fee details"]):
            for faq in self.faqs:
                if faq.get("category") == "fees":
                    return faq["answer"], "fees", 1.0

        if any(word in msg for word in ["exam", "exams", "exam date", "schedule"]):
            for faq in self.faqs:
                if faq.get("category") == "exams":
                    return faq["answer"], "exams", 1.0

        if any(word in msg for word in ["syllabus", "subjects", "course structure"]):
            for faq in self.faqs:
                if faq.get("category") == "syllabus":
                    return faq["answer"], "syllabus", 1.0

        if any(word in msg for word in ["apply", "application", "how to join", "how to get admission"]):
            for faq in self.faqs:
                 if faq.get("intent") == "how_to_apply":
                    return faq["answer"], "how_to_apply", 1.0

        if any(word in msg for word in ["eligibility", "qualification", "criteria", "requirement"]):
            for faq in self.faqs:
                if faq.get("intent") == "eligibility":
                    return faq["answer"], "eligibility", 1.0

        if any(word in msg for word in ["courses", "programs", "branches", "b.tech", "offered"]):
            for faq in self.faqs:
                if faq.get("intent") == "courses_offered":
                    return faq["answer"], "courses_offered", 1.0

        if any(word in msg for word in ["scholarship", "scholarships"]):
            for faq in self.faqs:
                if faq.get("intent") == "scholarships":
                    return faq["answer"], "scholarships", 1.0

        if any(word in msg for word in ["library"]):
            for faq in self.faqs:
                if faq.get("intent") == "library":
                    return faq["answer"], "library", 1.0

        if any(word in msg for word in ["hostel"]):
            for faq in self.faqs:
                if faq.get("intent") == "hostel":
                    return faq["answer"], "hostel", 1.0

        if any(word in msg for word in ["placement", "placements"]):
            for faq in self.faqs:
                if faq.get("intent") == "placements":
                    return faq["answer"], "placements", 1.0

        if any(word in msg for word in ["located", "location", "address"]):
            for faq in self.faqs:
                if faq.get("intent") == "contact_info" and "located" in faq.get("question", "").lower():
                    return faq["answer"], "location", 1.0

        if any(word in msg for word in ["contact", "phone", "email"]):
            for faq in self.faqs:
                if faq.get("intent") == "contact_info" and "contact" in faq.get("question", "").lower():
                    return faq["answer"], "contact", 1.0

        if any(word in msg for word in ["timing", "timings", "time"]):
            for faq in self.faqs:
                if faq.get("intent") == "college_timings":
                    return faq["answer"], "timings", 1.0

        # Step 4: TF-IDF similarity
        user_vector = self.vectorizer.transform([user_message])
        similarities = cosine_similarity(user_vector, self.question_vectors)
        best_index = int(np.argmax(similarities))
        best_score = float(similarities[0, best_index])

        print(f"📊 TF-IDF Score: {best_score:.4f} | FAQ: {self.questions[best_index]}")

        # Step 5: RAG/Groq fallback
        if best_score < self.threshold:
            if not self._is_college_domain_query(user_message):
                return (
                    "I can assist only with college-related information such as admissions, fees, exams, syllabus, and campus details.",
                    "out_of_domain",
                    best_score
                )
            rag_answer, rag_intent, rag_conf = self.rag_engine.ask(user_message)
            return rag_answer, rag_intent, rag_conf

        # Step 6: High confidence FAQ
        matched_faq = self.faqs[best_index]
        return matched_faq["answer"], matched_faq.get("intent", "faq"), best_score
