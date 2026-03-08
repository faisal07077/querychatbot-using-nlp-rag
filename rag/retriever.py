import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict


class RAGRetriever:
    """
    Semantic retriever over FAISS vector store.
    Compatible with IndexFlatL2 (lower distance = better).
    """

    def __init__(self, k: int = 5):
        self.k = k

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.index = faiss.read_index("data/vector_store/index.faiss")

        with open("data/vector_store/store.pkl", "rb") as f:
            store = pickle.load(f)

        self.documents = store["documents"]
        self.metadata = store["metadata"]

        # Intent keyword map for re-ranking
        self.intent_keywords = {
            "fee_payment":      ["pay online", "payment link", "pay fee", "online payment", "fee portal", "pay fees"],
            "fee_structure":    ["fee structure", "how much fee", "tuition fee", "course fee", "fee for"],
            "hostel":           ["hostel", "accommodation", "stay", "boarding"],
            "placement_stats":  ["placement percentage", "placement %", "how many placed"],
            "placement_package":["highest package", "salary", "lpa", "ctc", "package"],
            "recruiters":       ["top recruiters", "companies", "which companies", "who recruits"],
            "documents":        ["documents", "certificates", "what to bring", "papers needed"],
            "admission_process":["how to apply", "admission process", "how to join", "how to get admission"],
            "exam_schedule":    ["exam schedule", "timetable", "exam date", "when is exam"],
            "contact":          ["contact", "phone number", "email", "address", "reach"],
            "scholarship":      ["scholarship", "fee reimbursement", "financial aid"],
            "courses":          ["courses", "branches", "programs", "btech courses", "what courses"],
            "hostel":           ["hostel", "accommodation", "stay on campus"],
            "principal":        ["principal", "who is principal", "head of college"],
            "transport":        ["transport", "bus", "how to reach", "college bus"],
        }

    def _detect_intent(self, query: str) -> str:
        """Detect most likely intent from query keywords."""
        q = query.lower()
        best_intent = None
        best_score = 0
        for intent, keywords in self.intent_keywords.items():
            score = sum(1 for kw in keywords if kw in q)
            if score > best_score:
                best_score = score
                best_intent = intent
        return best_intent if best_score > 0 else None

    def retrieve(self, query: str) -> Tuple[List[Dict], float]:

        query_vec = self.model.encode([query])
        query_vec = np.array(query_vec).astype("float32")

        distances, indices = self.index.search(query_vec, self.k)

        retrieved_chunks = []
        for i, idx in enumerate(indices[0]):
            if idx < 0:
                continue
            retrieved_chunks.append({
                "text":     self.documents[idx],
                "meta":     self.metadata[idx],
                "distance": float(distances[0][i])
            })

        # ── Intent-based re-ranking ──
        detected_intent = self._detect_intent(query)
        if detected_intent:
            def rank_score(chunk):
                meta_intent = chunk.get("meta", {}).get("intent", "")
                # Boost chunks whose intent matches detected intent
                intent_boost = -1.0 if meta_intent == detected_intent else 0.0
                return chunk["distance"] + intent_boost
            retrieved_chunks.sort(key=rank_score)

        # Return top k=3 after re-ranking
        retrieved_chunks = retrieved_chunks[:3]

        top_distance = retrieved_chunks[0]["distance"] if retrieved_chunks else 999.0
        confidence = 1 / (1 + top_distance)

        return retrieved_chunks, confidence