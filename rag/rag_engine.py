from rag.retriever import RAGRetriever
from rag.generator import RAGGenerator


class RAGEngine:
    """
    Orchestrates retrieval and response synthesis.
    """

    def __init__(self):
        self.retriever = RAGRetriever(k=3)
        self.generator = RAGGenerator()

    def ask(self, question: str) -> str:
        retrieved_chunks = self.retriever.retrieve(question)
        return self.generator.generate(retrieved_chunks, question)
