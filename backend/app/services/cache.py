import hashlib
import json
from typing import Optional

class ResponseCache:
    def __init__(self):
        # In production, use Redis
        self._cache = {}

    def _get_key(self, query: str, session_id: str) -> str:
        data = f"{query}:{session_id}"
        return hashlib.sha256(data.encode()).hexdigest()

    def get(self, query: str, session_id: str) -> Optional[str]:
        key = self._get_key(query, session_id)
        return self._cache.get(key)

    def set(self, query: str, session_id: str, response: str):
        key = self._get_key(query, session_id)
        self._cache[key] = response

response_cache = ResponseCache()
