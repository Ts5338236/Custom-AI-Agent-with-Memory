from typing import List, Dict
from langchain_core.messages import BaseMessage
from collections import defaultdict

class ShortTermMemoryManager:
    def __init__(self, max_messages: int = 10):
        # In-memory store: {session_id: [messages]}
        # In production, this should be replaced with Redis
        self.store: Dict[str, List[BaseMessage]] = defaultdict(list)
        self.max_messages = max_messages

    def get_history(self, session_id: str) -> List[BaseMessage]:
        """Retrieves history for a specific session."""
        return self.store.get(session_id, [])

    def add_message(self, session_id: str, message: BaseMessage):
        """Adds a message to the session history and prunes if necessary."""
        self.store[session_id].append(message)
        
        # Keep only the last N messages
        if len(self.store[session_id]) > self.max_messages:
            self.store[session_id] = self.store[session_id][-self.max_messages:]

    def clear_history(self, session_id: str):
        """Clears history for a session."""
        if session_id in self.store:
            del self.store[session_id]

# Global singleton for in-memory storage
memory_manager = ShortTermMemoryManager()
