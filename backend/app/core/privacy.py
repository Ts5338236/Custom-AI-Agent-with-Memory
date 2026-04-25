import re
from typing import Dict, Tuple

class PrivacyManager:
    def __init__(self):
        # Basic patterns for common PII
        self.patterns = {
            "EMAIL": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
            "PHONE": r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
            "CREDIT_CARD": r"\b(?:\d[ -]*?){13,16}\b",
        }

    def mask_pii(self, text: str) -> Tuple[str, Dict[str, str]]:
        """
        Scans text for PII and replaces it with placeholders.
        Returns the masked text and a mapping to restore it later.
        """
        mapping = {}
        masked_text = text
        
        for pii_type, pattern in self.patterns.items():
            matches = re.findall(pattern, masked_text)
            for i, match in enumerate(set(matches)):
                placeholder = f"[{pii_type}_{i}]"
                mapping[placeholder] = match
                masked_text = masked_text.replace(match, placeholder)
        
        return masked_text, mapping

    def unmask_pii(self, text: str, mapping: Dict[str, str]) -> str:
        """
        Restores the original PII from placeholders in the text.
        """
        unmasked_text = text
        for placeholder, original in mapping.items():
            unmasked_text = unmasked_text.replace(placeholder, original)
        return unmasked_text

privacy_manager = PrivacyManager()
