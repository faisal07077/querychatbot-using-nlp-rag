import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer


class RAGRetriever:
    """
    Semantic retriever over FAISS vector store.
    """

    def __init__(self, k: int = 3):
        self.k = k
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = faiss.read_index("data/vector_store/index.faiss")

        with open("data/vector_store/store.pkl", "rb") as f:
            store = pickle.load(f)

        self.documents = store["documents"]
        self.metadata = store["metadata"]

    def retrieve(self, query: str):
        query_vec = self.model.encode([query])
        _, indices = self.index.search(np.array(query_vec), self.k)

        results = []
        for idx in indices[0]:
            results.append({
                "text": self.documents[idx],
                "meta": self.metadata[idx]
            })

        return results
