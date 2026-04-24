import ast
import json
import logging
import re

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
        candidate = cleaned[start:end]
        relaxed = re.sub(r",(\s*[}\]])", r"\1", candidate)
        try:
            return json.loads(relaxed)
        except json.JSONDecodeError:
            pass
        try:
            return ast.literal_eval(candidate)
        except (SyntaxError, ValueError):
            logging.error(f"JSON Parser Decode Crash: {err} -> Output: {candidate[:800]}")
            raise ValueError("Corrupt JSON payload format")

def normalize_action_schema(decision):
    """Normalizes partial action dicts so near-valid LLM output can still be used safely."""
    if not isinstance(decision, dict):
        return decision
    normalized = dict(decision)
    action = str(normalized.get("action", "none") or "none").strip().lower()
    if action not in {"redirect", "none"}:
        action = "none"
    normalized["action"] = action
    normalized.setdefault("from", "")
    normalized.setdefault("to", "")
    try:
        normalized["vehicles"] = max(0, int(normalized.get("vehicles", 0) or 0))
    except (TypeError, ValueError):
        normalized["vehicles"] = 0
    if action == "none":
        normalized["vehicles"] = 0
    return normalized

def validate_action_schema(decision):
    """Enforce expected action keys returning False identically on broken LLM dicts."""
    if not isinstance(decision, dict):
        return False
    decision = normalize_action_schema(decision)
    required_keys = {"action", "from", "to", "vehicles"}
    if not required_keys.issubset(decision.keys()):
        logging.warning(f"JSON Output bypassed validity boundaries, missing keys. Output: {decision}")
        return False
    if decision.get("action") == "redirect" and (not decision.get("from") or not decision.get("to") or int(decision.get("vehicles", 0) or 0) <= 0):
        logging.warning(f"JSON Output failed redirect normalization checks. Output: {decision}")
        return False
    return True
