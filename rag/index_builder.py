import json
import faiss
import pickle
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

FAQ_PATH = "data/faqs.json"
VECTOR_DIR = Path("data/vector_store")
VECTOR_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "all-MiniLM-L6-v2"

def main():
    print("🔹 Starting RAG index build")

    if not Path(FAQ_PATH).exists():
        print(f" FAQ file not found at {FAQ_PATH}")
        return

    with open(FAQ_PATH, "r", encoding="utf-8") as f:
        faqs = json.load(f)

    print(f"🔹 Loaded {len(faqs)} FAQ entries")

    if not faqs:
        print(" FAQ list is empty")
        return

    documents = []
    metadata = []

    for faq in faqs:
        doc = (
            f"Category: {faq.get('category')}\n"
            f"Question: {faq.get('question')}\n"
            f"Answer: {faq.get('answer')}"
        )
        documents.append(doc)
        metadata.append({
            "intent": faq.get("intent"),
            "category": faq.get("category")
        })

    print(" Building embeddings...")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(documents, show_progress_bar=True)

    print(" Creating FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    print(" Writing index to disk...")
    faiss.write_index(index, str(VECTOR_DIR / "index.faiss"))

    with open(VECTOR_DIR / "store.pkl", "wb") as f:
        pickle.dump(
            {"documents": documents, "metadata": metadata},
            f
        )

    print(" RAG index build completed successfully")

if __name__ == "__main__":
    main()
