# nlp.py
"""
Enterprise NLP Layer for College Query Chatbot

This module represents the core Natural Language Processing (NLP)
controller for the college chatbot system.

ARCHITECTURAL OVERVIEW
----------------------
The chatbot follows a layered decision-making approach:

1. Conversational handling (greetings, polite talk)
2. Deterministic keyword-based intent resolution
3. TF-IDF based semantic similarity matching
4. Retrieval-Augmented Generation (RAG) fallback
5. Explicit out-of-domain blocking

DESIGN PRINCIPLES
-----------------
- Deterministic responses have the highest priority
- Probabilistic methods are used only when required
- Retrieval is invoked only for domain-relevant queries
- Safety, explainability, and auditability are preserved
- Existing logic is preserved without modification

"""

import json
import os
import numpy as np
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# RAG engine (used only for low-confidence, domain-valid queries)
from rag.rag_engine import RAGEngine


class CollegeChatbot:
    """
    Central NLP orchestrator for the college chatbot.

    Responsibilities:
    - Manage FAQ knowledge base
    - Perform intent routing
    - Apply semantic similarity
    - Invoke RAG only when appropriate
    """

    def __init__(
        self,
        college_name: str,
        faq_path: str = "data/faqs.json",
        threshold: float = 0.25
    ):
        """
        Initialize chatbot configuration and NLP resources.

        Args:
            college_name (str): Name of the institution
            faq_path (str): Path to FAQ JSON file
            threshold (float): Similarity confidence threshold
        """

        self.college_name = college_name
        self.faq_path = faq_path
        self.threshold = threshold

        # Load FAQ knowledge base
        self.faqs = self._load_faqs()

        # Extract questions for vectorization
        self.questions: List[str] = [item["question"] for item in self.faqs]

        if not self.questions:
            raise ValueError(
                "FAQ knowledge base is empty. "
                "At least one FAQ entry is required."
            )

        # TF-IDF vectorizer for semantic similarity
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.question_vectors = self.vectorizer.fit_transform(self.questions)

        # Initialize RAG engine once
        self.rag_engine = RAGEngine()

    # ------------------------------------------------------------------
    # Knowledge Base Loading
    # ------------------------------------------------------------------

    def _load_faqs(self) -> List[dict]:
        """
        Load FAQ data from JSON storage.

        Returns:
            List[dict]: FAQ entries

        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If JSON structure is invalid
        """
        if not os.path.exists(self.faq_path):
            raise FileNotFoundError(
                f"FAQ file not found at path: {self.faq_path}"
            )

        with open(self.faq_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(
                "Invalid FAQ format. Expected a list of objects."
            )

        return data

    # ------------------------------------------------------------------
    # Conversational / Small Talk Handling
    # ------------------------------------------------------------------

    def _basic_small_talk(self, message: str):
        """
        Handle greetings, polite conversation, and identity queries.

        This method is intentionally lightweight and deterministic.
        It prevents harmless conversational inputs from reaching
        semantic or retrieval-based pipelines.

        Args:
            message (str): Raw user input

        Returns:
            str | None: Response if handled, else None
        """
        msg = message.lower().strip()

        # Greetings
        if any(word in msg for word in ["hello", "hi", "hey"]):
            return (
                f"Hello! I am the virtual assistant for {self.college_name}. "
                "You can ask me about admissions, exams, syllabus, or fees."
            )

        # Polite conversational phrases
        if any(phrase in msg for phrase in [
            "how are you",
            "how r you",
            "how are u",
            "how's it going",
            "what's up",
            "how do you do"
            "how r u"
            "khyriyat"
            "namaste"
            
        ]):
            return (
                "I'm doing well, thank you for asking. "
                "How can I assist you with college-related information today?"
            )

        # Time-based greetings
        if any(phrase in msg for phrase in [
            "good morning",
            "good afternoon",
            "good evening"
        ]):
            return (
                "Good day! How can I help you with college-related queries?"
            )

        # Appreciation
        if any(word in msg for word in ["thank", "thanks"]):
            return "You're welcome. Let me know if you need any college information."

        # Identity
        if "who are you" in msg:
            return f"I am the AI assistant for {self.college_name}."

        return None

    # ------------------------------------------------------------------
    # Domain Validation Guard
    # ------------------------------------------------------------------

    def _is_college_domain_query(self, message: str) -> bool:
        """
        Validate whether a query belongs to the college domain.

        This guard prevents irrelevant or inappropriate queries
        from invoking the RAG pipeline.

        Args:
            message (str): User input

        Returns:
            bool: True if domain-related, False otherwise
        """
        domain_keywords = [
            "college", "stanley", "admission", "apply", "eligibility",
            "fees", "fee", "exam", "exams", "syllabus", "course",
            "b.tech", "branch", "department", "faculty",
            "timings", "hostel", "campus", "placement",
            "contact", "office", "university", "counseling"
        ]

        msg = message.lower()
        return any(keyword in msg for keyword in domain_keywords)

    # ------------------------------------------------------------------
    # Main Response Controller
    # ------------------------------------------------------------------

    def get_response(self, user_message: str) -> Tuple[str, str, float]:
        """
        Generate a response for a given user query.

        Processing Order:
        1. Ignore trivial acknowledgements
        2. Small talk handling
        3. Deterministic keyword-based intents
        4. TF-IDF semantic similarity matching
        5. Domain-validated RAG fallback

        Args:
            user_message (str): User input

        Returns:
            Tuple[str, str, float]:
                - response text
                - intent label
                - confidence score
        """

        msg = user_message.lower().strip()

        # Step 1: Ignore trivial acknowledgement inputs
        ignore_words = ["ok", "okay", "hmm", "yes", "no", "fine", "good"]
        if msg in ignore_words:
            return (
                "Acknowledged. You may ask about admissions, exams, syllabus, or fees.",
                "acknowledge",
                1.0
            )

        # Step 2: Conversational handling
        small_talk = self._basic_small_talk(user_message)
        if small_talk:
            return small_talk, "small_talk", 1.0

        # Step 3: Deterministic keyword-based intent routing
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

        if any(word in msg for word in ["admission", "apply", "eligibility", "join"]):
            for faq in self.faqs:
                if faq.get("category") == "admissions":
                    return faq["answer"], "admissions", 1.0

        # Step 4: TF-IDF semantic similarity
        user_vector = self.vectorizer.transform([user_message])
        similarities = cosine_similarity(user_vector, self.question_vectors)

        best_index = int(np.argmax(similarities))
        best_score = float(similarities[0, best_index])

        # Step 5: Controlled RAG fallback
        if best_score < self.threshold:

            # Domain guard
            if not self._is_college_domain_query(user_message):
                return (
                    "I can assist only with college-related information such as admissions, fees, exams, syllabus, and campus details.",
                    "out_of_domain",
                    best_score
                )

            rag_response = self.rag_engine.ask(user_message)
            return rag_response, "rag_fallback", best_score

        # Step 6: High-confidence FAQ response
        matched_faq = self.faqs[best_index]
        return matched_faq["answer"], matched_faq.get("intent", "faq"), best_score
