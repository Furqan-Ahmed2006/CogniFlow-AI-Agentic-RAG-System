import re


class Guardrails:
    @staticmethod
    def inspect_input(query: str) -> dict:
        """Prompt injection & malicious pattern detector."""
        if not query or not query.strip():
            return {"is_safe": False, "reason": "Empty query."}

        injection_patterns = [
            r"ignore previous instructions",
            r"ignore all instructions",
            r"system prompt",
            r"bypass guardrails",
            r"you are now DAN",
            r"jailbreak",
            r"reveal your prompt",
        ]

        for pattern in injection_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return {
                    "is_safe": False,
                    "reason": "Security Alert: Potential prompt injection detected.",
                }

        return {"is_safe": True, "reason": "Passed safety checks."}

    @staticmethod
    def verify_groundedness(response: str, context: list) -> bool:
        """Check if response has meaningful overlap with retrieved context."""
        if not response or not response.strip() or not context:
            return False

        response_lower = response.lower()
        for chunk in context:
            text = chunk.page_content if hasattr(chunk, "page_content") else str(chunk)
            words = [w for w in re.findall(r"\w+", text.lower()) if len(w) > 4]
            if any(w in response_lower for w in words):
                return True
        return False