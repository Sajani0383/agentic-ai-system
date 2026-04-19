import re

class InputValidator:
    """Sanitize prompts to prevent injection and filter malformed inputs"""
    
    @staticmethod
    def sanitize_query(raw_query: str) -> str:
        if not raw_query:
            return ""
        
        # 1. Strip whitespace
        cleaned = raw_query.strip()
        
        # 2. Limit length (Stop enormous payloads going into LLM)
        if len(cleaned) > 500:
            cleaned = cleaned[:500]
            
        # 3. Clean obvious script injections / HTML tags heavily
        cleaned = re.sub(r'<[^>]*>', '', cleaned)
        
        return cleaned

validator = InputValidator()
