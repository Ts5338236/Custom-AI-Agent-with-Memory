import re
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

class PrivacyShield:
    def __init__(self):
        # Regex patterns for common PII
        self.patterns = {
            "EMAIL": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
            "PHONE": r"(\+\d{1,2}\s)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}",
            "CREDIT_CARD": r"\b(?:\d[ -]*?){13,16}\b",
            "SSN": r"\b\d{3}-\d{2}-\d{4}\b"
        }

    def redact(self, text: str) -> Tuple[str, Dict[str, str]]:
        """
        Redacts PII from text and returns the redacted text plus a mapping for restoration.
        """
        redacted_text = text
        mapping = {}
        
        for pii_type, pattern in self.patterns.items():
            matches = re.findall(pattern, redacted_text)
            for i, match in enumerate(matches):
                placeholder = f"[REDACTED_{pii_type}_{i}]"
                mapping[placeholder] = match
                redacted_text = redacted_text.replace(match, placeholder)
        
        if mapping:
            logger.info(f"Redacted {len(mapping)} PII items from input.")
            
        return redacted_text, mapping

    def restore(self, text: str, mapping: Dict[str, str]) -> str:
        """
        Restores redacted PII in a response if the LLM happened to repeat placeholders.
        """
        restored_text = text
        for placeholder, original in mapping.items():
            restored_text = restored_text.replace(placeholder, original)
        return restored_text

privacy_shield = PrivacyShield()
