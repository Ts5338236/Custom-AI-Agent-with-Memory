from app.services.vector_db import vector_db_service
from app.services.memory import memory_manager
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from app.services.prompt_registry import prompt_registry
from typing import List

class PromptBuilder:
    def __init__(self, history_limit: int = 10):
        self.history_limit = history_limit

    async def build_context(self, query: str, session_id: str) -> str:
        """Fetches and formats relevant context from long-term memory."""
        relevant_memories = vector_db_service.search_memories(query, k=3)
        if not relevant_memories:
            return "No relevant past context found."
        
        context_str = "\n".join([f"- {m['content']}" for m in relevant_memories])
        return f"Long-term memories related to '{query}':\n{context_str}"

    def get_system_prompt(self, context: str, preferences: dict = {}) -> str:
        pref_str = "\n".join([f"- {k}: {v}" for k, v in preferences.items()]) if preferences else "No specific preferences learned yet."
        
        template = prompt_registry.get_prompt("agent_system")
        return template.format(context=context, preferences=pref_str)

    def prune_history(self, history: List[BaseMessage]) -> List[BaseMessage]:
        """Prunes the conversation history to stay within token limits."""
        if len(history) > self.history_limit:
            return history[-self.history_limit:]
        return history

prompt_builder = PromptBuilder()
