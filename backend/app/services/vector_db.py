from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from app.core.config import settings
import os

class VectorDBService:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY)
        self.index_path = "faiss_index"
        self.vector_db = self._load_or_create_index()

    def _load_or_create_index(self):
        if os.path.exists(self.index_path):
            return FAISS.load_local(
                self.index_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
        else:
            # Initialize with a dummy document to create the index
            initial_doc = [Document(page_content="Initial memory seed.", metadata={"source": "system"})]
            db = FAISS.from_documents(initial_doc, self.embeddings)
            db.save_local(self.index_path)
            return db

    def add_memory(self, text: str, category: str = "fact", importance: int = 5, metadata: dict = {}):
        """
        Adds a new memory with rich metadata for intelligent retrieval.
        """
        import datetime
        full_metadata = {
            "category": category,
            "importance": importance,
            "timestamp": datetime.datetime.utcnow().timestamp(),
            **metadata
        }
        doc = Document(page_content=text, metadata=full_metadata)
        self.vector_db.add_documents([doc])
        self.vector_db.save_local(self.index_path)

    def search_memories(self, query: str, k: int = 5):
        """
        Searches for relevant memories and applies intelligent scoring.
        """
        results = self.vector_db.similarity_search_with_score(query, k=k)
        import datetime
        now = datetime.datetime.utcnow().timestamp()
        
        formatted_results = []
        for doc, vector_score in results:
            # --- INTELLIGENT SCORING ALGORITHM ---
            # 1. Similarity Score (normalized)
            # 2. Importance Weight (higher importance boosts score)
            # 3. Recency Decay (exponential decay over time)
            
            importance = doc.metadata.get("importance", 5)
            timestamp = doc.metadata.get("timestamp", now)
            
            # Time decay factor: (1 / (1 + days_since_memory))
            days_passed = (now - timestamp) / 86400
            recency_score = 1.0 / (1.0 + 0.1 * days_passed) # 0.1 is the decay rate
            
            # Final heuristic score (higher is better)
            # We invert vector_score because lower distance means higher similarity
            similarity_score = 1.0 / (1.0 + vector_score) 
            
            final_score = (similarity_score * 0.5) + (recency_score * 0.3) + (importance / 10.0 * 0.2)
            
            formatted_results.append({
                "content": doc.page_content,
                "category": doc.metadata.get("category"),
                "score": final_score,
                "metadata": doc.metadata
            })
            
        # Rank by final intelligent score
        formatted_results.sort(key=lambda x: x["score"], reverse=True)
        return formatted_results

vector_db_service = VectorDBService()
