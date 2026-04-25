from typing import List, Dict, Any
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from app.services.vector_db import vector_db_service

class PromptBuilder:
    def __init__(self, max_history_tokens: int = 2000):
        self.max_history_tokens = max_history_tokens

    async def build_context(self, query: str, session_id: str) -> str:
        """
        Retrieves relevant long-term memories and ranks them.
        """
        memories = vector_db_service.search_memories(query, k=5)
        
        # Simple Ranking Logic: Filter by score threshold (e.g., L2 distance < 0.6)
        # Note: Score depends on embedding model; for FAISS-L2, lower is better.
        relevant_memories = [m for m in memories if m["score"] < 0.8]
        
        if not relevant_memories:
            return "No specific long-term context found."
            
        context_str = "\n".join([f"- {m['content']}" for m in relevant_memories])
        return f"Long-term memories related to '{query}':\n{context_str}"

    def get_system_prompt(self, context: str) -> str:
        return (
            "You are a sophisticated Custom AI Agent. "
            "You have access to the following long-term context about the user:\n"
            f"--- CONTEXT ---\n{context}\n---------------\n"
            "Use the context above to personalize your responses. "
            "If the context is irrelevant, prioritize the current conversation flow. "
            "Stay helpful, concise, and professional."
        )

    def prune_history(self, history: List[BaseMessage]) -> List[BaseMessage]:
        """
        Naive token management: Keep only the most recent messages.
        In production, use a tiktoken-based counter for precise pruning.
        """
        if len(history) > 10:
            return history[-10:]
        return history

prompt_builder = PromptBuilder()
