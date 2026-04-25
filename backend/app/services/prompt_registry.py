import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class PromptRegistry:
    def __init__(self):
        # In a real production system, this would be backed by a DB or S3
        self._prompts: Dict[str, Dict[str, str]] = {
            "agent_system": {
                "v1": "You are a helpful AI assistant. Use the following context: {context}",
                "v2": "You are a sophisticated Custom AI Agent. "
                      "You have access to the following user profile and long-term context:\n\n"
                      "--- USER PREFERENCES ---\n{preferences}\n\n"
                      "--- LONG-TERM CONTEXT ---\n{context}\n"
                      "------------------------\n"
                      "Personalize your response based on the preferences above. "
                      "Stay helpful, concise, and professional.",
                "active": "v2"
            },
            "planner_system": {
                "v1": "You are a Planner Agent. Break the user's request into tasks. Tools: {tools}",
                "active": "v1"
            },
            "memory_analyzer": {
                "v1": "Analyze this text for facts and importance. Text: {text}",
                "active": "v1"
            }
        }

    def get_prompt(self, key: str, version: Optional[str] = None) -> str:
        """Retrieves a prompt by key and optional version."""
        if key not in self._prompts:
            logger.error(f"Prompt key '{key}' not found.")
            return ""
        
        v = version or self._prompts[key].get("active", "v1")
        prompt = self._prompts[key].get(v)
        
        if not prompt:
            logger.warning(f"Version '{v}' for prompt '{key}' not found. Falling back to v1.")
            return self._prompts[key].get("v1", "")
            
        return prompt

    def update_active_version(self, key: str, version: str):
        """Switches the active version of a prompt."""
        if key in self._prompts and version in self._prompts[key]:
            self._prompts[key]["active"] = version
            logger.info(f"Updated active version for '{key}' to '{version}'.")

prompt_registry = PromptRegistry()
