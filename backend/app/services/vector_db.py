import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from app.core.config import settings
from rank_bm25 import BM25Okapi
import datetime

class HybridVectorDB:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY)
        self.index_path = "faiss_index"
        self.vector_db = self._load_or_create_vector_db()
        self.bm25 = None
        self.documents = []
        self._refresh_bm25()

    def _load_or_create_vector_db(self):
        if os.path.exists(self.index_path):
            return FAISS.load_local(self.index_path, self.embeddings, allow_dangerous_deserialization=True)
        else:
            texts = ["Initial memory seed"]
            db = FAISS.from_texts(texts, self.embeddings)
            db.save_local(self.index_path)
            return db

    def _refresh_bm25(self):
        """Refreshes the BM25 index from current documents in the vector store."""
        # Note: In a production system, we'd store documents in a separate DB for BM25 efficiency
        self.documents = list(self.vector_db.docstore._dict.values())
        if self.documents:
            tokenized_corpus = [doc.page_content.lower().split() for doc in self.documents]
            self.bm25 = BM25Okapi(tokenized_corpus)

    def add_memory(self, text: str, category: str = "fact", importance: int = 5, metadata: dict = {}):
        full_metadata = {
            "category": category,
            "importance": importance,
            "timestamp": datetime.datetime.utcnow().timestamp(),
            **metadata
        }
        doc = Document(page_content=text, metadata=full_metadata)
        self.vector_db.add_documents([doc])
        self.vector_db.save_local(self.index_path)
        self._refresh_bm25()

    def search_memories(self, query: str, k: int = 5):
        # 1. Semantic Search (FAISS)
        semantic_results = self.vector_db.similarity_search_with_score(query, k=k*2)
        
        # 2. Keyword Search (BM25)
        keyword_results = []
        if self.bm25:
            tokenized_query = query.lower().split()
            keyword_scores = self.bm25.get_scores(tokenized_query)
            top_n = sorted(range(len(keyword_scores)), key=lambda i: keyword_scores[i], reverse=True)[:k*2]
            keyword_results = [(self.documents[i], keyword_scores[i]) for i in top_n if keyword_scores[i] > 0]

        # 3. Reciprocal Rank Fusion (RRF)
        # RRF Score = sum(1 / (rank + k_param))
        k_param = 60
        rrf_scores = {}
        
        for rank, (doc, _) in enumerate(semantic_results):
            doc_id = doc.page_content
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (rank + k_param)
            
        for rank, (doc, _) in enumerate(keyword_results):
            doc_id = doc.page_content
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (rank + k_param)

        # Sort and take top K
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        # Find original document objects for metadata
        final_results = []
        for content, rrf_score in sorted_docs:
            # Find the first matching doc object
            doc = next((d for d in self.documents if d.page_content == content), None)
            if doc:
                final_results.append({
                    "content": doc.page_content,
                    "category": doc.metadata.get("category"),
                    "score": rrf_score,
                    "metadata": doc.metadata
                })
        
        return final_results

# Replace the old service
vector_db_service = HybridVectorDB()
