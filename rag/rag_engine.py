from rag.retriever import RAGRetriever
from rag.generator import RAGGenerator
from rag.llm_service import OllamaLLMService

import logging
from typing import Tuple


# Raise threshold so only strong FAQ matches use RAG
# Anything below this goes to Groq
CONFIDENCE_THRESHOLD = 0.65


class RAGEngine:
    """
    Hybrid RAG + LLM fallback engine.
    """

    def __init__(self):
        self.retriever = RAGRetriever(k=5)
        self.generator = RAGGenerator()
        self.llm_service = OllamaLLMService()

    def ask(self, question: str) -> Tuple[str, str, float]:

        # Step 1: Retrieve context + similarity score
        retrieved_chunks, confidence = self.retriever.retrieve(question)

        logging.info(f"RAG Confidence: {confidence:.4f}")

        # Step 2: High confidence → FAQ matched well, use RAG
        if confidence >= CONFIDENCE_THRESHOLD:
            logging.info("✅ Using RAG generator")
            answer = self.generator.generate(retrieved_chunks, question)
            return answer, "rag", confidence

        # Step 3: Low confidence → FAQ failed, let Groq answer on its own
        logging.info(f"🤖 FAQ failed (confidence {confidence:.4f}) → Groq answering")
        llm_answer = self.llm_service.generate(
            question=question,
            context=None  # Groq uses its own knowledge
        )

        return llm_answer, "llm_fallback", confidence