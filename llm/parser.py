import json
import logging

def extract_json_payload(text):
    """Safely extracts JSON blocks preventing schema corruption and trace crashes."""
    cleaned = text.strip()
    
    if "```" in cleaned:
        parts = cleaned.split("```")
        cleaned = next((part for part in parts if "{" in part and "}" in part), cleaned)
        
    cleaned = cleaned.replace("json", "").replace("JSON", "").strip()
    
    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1
    
    if start == -1 or end <= 0:
        logging.warning("JSON Parser Failure: Output lacked valid block bounds.")
        raise ValueError("No JSON object found in model output")
        
    try:
        return json.loads(cleaned[start:end])
    except json.JSONDecodeError as err:
        logging.error(f"JSON Parser Decode Crash: {err} -> Output: {cleaned[start:end]}")
        raise ValueError("Corrupt JSON payload format")

def validate_action_schema(decision):
    """Enforce expected action keys returning False identically on broken LLM dicts."""
    if not isinstance(decision, dict):
        return False
    required_keys = {"action", "from", "to", "vehicles"}
    if not required_keys.issubset(decision.keys()):
        logging.warning(f"JSON Output bypassed validity boundaries, missing keys. Output: {decision}")
        return False
    return True
