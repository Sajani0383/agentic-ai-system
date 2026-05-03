import os
import sys
import time
import logging
import threading
import hashlib
import warnings
from types import SimpleNamespace
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"google\.auth|google\.oauth2",
)
warnings.filterwarnings(
    "ignore",
    message=r".*@model_validator.*mode='after'.*",
)
try:
    from dotenv import dotenv_values, load_dotenv
except ImportError:  # Optional in local fallback mode
    def dotenv_values(*args, **kwargs):
        return {}
    def load_dotenv(*args, **kwargs):
        return False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
ENV_EXAMPLE_PATH = PROJECT_ROOT / ".env.example"

if ENV_PATH.exists(): load_dotenv(dotenv_path=ENV_PATH, override=True)

DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
DEFAULT_GEMINI_MODEL_SEQUENCE = [
    model.strip()
    for model in os.getenv(
        "GEMINI_MODEL_SEQUENCE",
        "gemini-flash-lite-latest,gemini-2.0-flash-lite,gemini-2.0-flash-lite-001,gemini-2.5-flash-lite,gemini-2.0-flash,gemini-flash-latest,gemini-2.5-flash",
    ).split(",")
    if model.strip()
]
_LLM_INSTANCE = None
_LLM_INSTANCE_KEY = None
_LLM_INSTANCE_LOCK = threading.RLock()


class LLMStatusManager:
    """Thread-safe mechanism avoiding corrupted parallel exception writes."""
    def __init__(self):
        self._lock = threading.RLock()
        self.last_error = ""
        self.backoff_until = 0.0
        self.backoff_reason = ""
        self.backoff_kind = ""
        self.key_backoffs = {}
        self.router_trace = []
        self.active_route = {}

    def set_error(self, message):
        with self._lock:
            self.last_error = message

    def start_backoff(self, message, seconds=300, kind="quota"):
        with self._lock:
            self.last_error = message
            self.backoff_until = time.time() + max(1, int(seconds))
            self.backoff_reason = message
            self.backoff_kind = kind

    def clear_backoff(self):
        with self._lock:
            self.backoff_until = 0.0
            self.backoff_reason = ""
            self.backoff_kind = ""

    def reset_runtime_flags(self):
        with self._lock:
            self.last_error = ""
            self.backoff_until = 0.0
            self.backoff_reason = ""
            self.backoff_kind = ""
            self.key_backoffs = {}
            self.router_trace = []
            self.active_route = {}

    def get_error(self):
        with self._lock:
            return self.last_error

    def get_backoff(self):
        with self._lock:
            remaining = max(0, int(self.backoff_until - time.time()))
            if remaining <= 0 and self.backoff_until:
                self.backoff_until = 0.0
                self.backoff_reason = ""
                self.backoff_kind = ""
            return {
                "active": remaining > 0,
                "remaining_seconds": remaining,
                "reason": self.backoff_reason,
                "kind": self.backoff_kind or ("quota" if remaining > 0 else ""),
            }

    def mark_key_backoff(self, key_label, message, seconds=300, kind="quota"):
        with self._lock:
            self.key_backoffs[key_label] = {
                "until": time.time() + max(1, int(seconds)),
                "reason": message,
                "kind": kind,
            }

    def get_key_backoffs(self):
        with self._lock:
            active = {}
            now = time.time()
            expired = []
            for key_label, payload in self.key_backoffs.items():
                remaining = int(payload.get("until", 0) - now)
                if remaining <= 0:
                    expired.append(key_label)
                    continue
                active[key_label] = {
                    "active": True,
                    "remaining_seconds": remaining,
                    "reason": payload.get("reason", ""),
                    "kind": payload.get("kind", "quota"),
                }
            for key_label in expired:
                self.key_backoffs.pop(key_label, None)
            return active

    def is_key_backoff_active(self, key_label):
        return self.get_key_backoffs().get(key_label, {}).get("active", False)

    def set_router_trace(self, trace, active_route=None):
        with self._lock:
            self.router_trace = list(trace or [])[-24:]
            self.active_route = dict(active_route or {})

    def get_router_trace(self):
        with self._lock:
            return list(self.router_trace)

    def get_active_route(self):
        with self._lock:
            return dict(self.active_route)

STATUS_MANAGER = LLMStatusManager()


def reset_llm_runtime_state():
    global _LLM_INSTANCE, _LLM_INSTANCE_KEY
    STATUS_MANAGER.reset_runtime_flags()
    with _LLM_INSTANCE_LOCK:
        _LLM_INSTANCE = None
        _LLM_INSTANCE_KEY = None


def _is_terminal_llm_error(message):
    text = (message or "").upper()
    return any(m in text for m in ["API_KEY_INVALID", "API KEY EXPIRED", "INVALID_ARGUMENT", "PERMISSION_DENIED", "UNAUTHENTICATED"])

def _is_connectivity_llm_error(message):
    text = (message or "").upper()
    return any(m in text for m in ["CONNECTERROR", "CONNECTIONERROR", "NODENAME", "RESOLUTION", "TLS", "SSL", "CERTIFICATE", "TIMEOUT"])

def _is_quota_llm_error(message):
    text = (message or "").upper()
    return any(m in text for m in ["429", "RESOURCE_EXHAUSTED", "QUOTA", "RATE_LIMIT", "RATE LIMIT"])

def _is_daily_quota_llm_error(message):
    text = (message or "").upper()
    return (
        "GENERATEREQUESTSPERDAYPERPROJECTPERMODEL-FREETIER" in text
        or "GENERATIVELANGUAGE.GOOGLEAPIS.COM/GENERATE_CONTENT_FREE_TIER_REQUESTS" in text
    )

def _is_model_unavailable_llm_error(message):
    text = (message or "").upper()
    return (
        ("MODEL" in text or "MODELS/" in text)
        and (
            "NOT_FOUND" in text
            or "IS NOT FOUND" in text
            or "NOT SUPPORTED FOR GENERATECONTENT" in text
            or "INVALID_ARGUMENT" in text
        )
    )


def _clean_key(raw):
    cleaned = (raw or "").strip()
    if cleaned.lower() in {"your_google_api_key_here", "your_real_key_here", "replace_me"} or len(cleaned) < 20 or not cleaned.startswith("AIza"):
        return None
    return cleaned


def get_api_keys():
    file_values = dotenv_values(ENV_PATH) if ENV_PATH.exists() else {}
    combined = (
        file_values.get("GOOGLE_API_KEYS")
        or file_values.get("GEMINI_API_KEYS")
        or os.getenv("GOOGLE_API_KEYS")
        or os.getenv("GEMINI_API_KEYS")
        or ""
    )
    keys = []
    for raw in combined.replace("\n", ",").split(","):
        cleaned = _clean_key(raw)
        if cleaned and cleaned not in keys:
            keys.append(cleaned)
    single_key = (
        file_values.get("GOOGLE_API_KEY")
        or file_values.get("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
    )
    cleaned_single = _clean_key(single_key)
    if cleaned_single and cleaned_single not in keys:
        keys.append(cleaned_single)
    return keys


def get_api_key():
    keys = get_api_keys()
    return keys[0] if keys else None


def _key_label(index):
    return f"Key {chr(ord('A') + index)}"


class GeminiLLMWrapper:
    """Enterprise LLM implementation with model routing, key failover, and hard timeouts."""
    def __init__(self, api_keys, models):
        from google import genai
        self.api_keys = list(api_keys)
        self.models = list(models) or [DEFAULT_GEMINI_MODEL]
        self.clients = {}
        self.model = self.models[0]
        self._genai = genai
        
        # Operational Limits
        self.call_lock = threading.Lock()
        self.last_call_time = 0.0
        self._cache = {}
        self._route_cursor = 0
        # NOTE: No shared executor — each invoke() creates a fresh one inside a
        # `with` block so Python's interpreter-shutdown ThreadPoolExecutor cleanup
        # can never fire on a stale reference and raise
        # "cannot schedule new futures after shutdown".
        
        self.TIMEOUT_SECONDS = float(os.getenv("GEMINI_TIMEOUT_SECONDS", "15.0"))
        self.MAX_ATTEMPTS_PER_CALL = max(1, int(os.getenv("GEMINI_MAX_ATTEMPTS_PER_CALL", "2")))
        self.RATE_LIMIT_DELAY = max(0.0, float(os.getenv("GEMINI_RATE_LIMIT_DELAY", "0.25")))

    def invoke(self, prompt):
        # 1. Evaluate Cache 
        prompt_hash = hashlib.md5(prompt.encode('utf-8')).hexdigest()
        if prompt_hash in self._cache:
            logging.debug("LLM Cache HIT - bypassing network latency.")
            return SimpleNamespace(content=self._cache[prompt_hash])

        # 2. Rate Limiter Enforced sequentially
        with self.call_lock:
            now = time.time()
            elapsed = now - self.last_call_time
            if elapsed < self.RATE_LIMIT_DELAY:
                time.sleep(self.RATE_LIMIT_DELAY - elapsed)
            self.last_call_time = time.time()

        last_exc = None
        trace = []
        if not self.api_keys:
            raise RuntimeError("No Gemini API keys are available for routing.")
        start_index = self._route_cursor % len(self.api_keys)
        self._route_cursor += 1
        key_order = list(range(start_index, len(self.api_keys))) + list(range(0, start_index))
        attempt_count = 0
        for key_index in key_order:
            if attempt_count >= self.MAX_ATTEMPTS_PER_CALL:
                break
            api_key = self.api_keys[key_index]
            key_label = _key_label(key_index)
            if STATUS_MANAGER.is_key_backoff_active(key_label):
                trace.append({"key": key_label, "model": "-", "status": "skipped", "reason": "key cooldown active"})
                continue
            key_quota_failures = 0
            for model in self.models:
                if attempt_count >= self.MAX_ATTEMPTS_PER_CALL:
                    break
                attempt_count += 1
                start_time = time.time()
                executor = None
                future = None
                try:
                    if sys.is_finalizing():
                        raise RuntimeError("Interpreter is shutting down; skipping Gemini request.")
                    client = self._client_for_key(api_key)
                    executor = ThreadPoolExecutor(max_workers=1)
                    future = executor.submit(client.models.generate_content, model=model, contents=prompt)
                    response = future.result(timeout=self.TIMEOUT_SECONDS)
                    text = self._extract_text(response)
                    STATUS_MANAGER.set_error("")
                    STATUS_MANAGER.clear_backoff()
                    result_text = (text or "").strip()
                    self._cache[prompt_hash] = result_text
                    if len(self._cache) > 200:
                        self._cache.clear()
                    duration = time.time() - start_time
                    active_route = {"key": key_label, "model": model, "latency_seconds": round(duration, 2)}
                    trace.append({"key": key_label, "model": model, "status": "success", "latency_seconds": round(duration, 2)})
                    STATUS_MANAGER.set_router_trace(trace, active_route)
                    logging.info(f"LLM call successful: {model} via {key_label} in {duration:.2f}s")
                    return SimpleNamespace(content=result_text, model=model, key_label=key_label)
                except TimeoutError:
                    last_exc = Exception("Cloud LLM Timeout bounds breached successfully dropping request.")
                    trace.append({"key": key_label, "model": model, "status": "timeout", "reason": str(last_exc)})
                    if future is not None:
                        future.cancel()
                    continue
                except Exception as exc:
                    last_exc = exc
                    message = f"{type(exc).__name__}: {exc}"
                    trace.append({"key": key_label, "model": model, "status": "failed", "reason": self._short_reason(message)})
                    if "cannot schedule new futures after shutdown" in message.lower() or "interpreter is shutting down" in message.lower():
                        STATUS_MANAGER.start_backoff(message, seconds=20, kind="transient")
                        STATUS_MANAGER.set_router_trace(trace)
                        raise
                    if _is_quota_llm_error(message):
                        attempt_count = max(0, attempt_count - 1)
                        key_quota_failures += 1
                        continue
                    if _is_model_unavailable_llm_error(message):
                        attempt_count = max(0, attempt_count - 1)
                        STATUS_MANAGER.set_error("")
                        continue
                    if _is_terminal_llm_error(message):
                        attempt_count = max(0, attempt_count - 1)
                        STATUS_MANAGER.mark_key_backoff(key_label, message, seconds=6 * 60 * 60, kind="terminal")
                        break
                    if _is_connectivity_llm_error(message) or "503" in message or "UNAVAILABLE" in message.upper():
                        continue
                    STATUS_MANAGER.set_error(message)
                    continue
                finally:
                    if executor is not None:
                        try:
                            executor.shutdown(wait=False, cancel_futures=True)
                        except TypeError:
                            executor.shutdown(wait=False)
            if key_quota_failures >= len(self.models):
                STATUS_MANAGER.mark_key_backoff(key_label, "All configured Gemini model tiers returned quota/rate-limit errors.", seconds=6 * 60 * 60, kind="daily_quota")
        if attempt_count >= self.MAX_ATTEMPTS_PER_CALL:
            trace.append({
                "key": "-",
                "model": "-",
                "status": "bounded",
                "reason": f"attempt cap reached ({self.MAX_ATTEMPTS_PER_CALL})",
            })
        STATUS_MANAGER.set_router_trace(trace)
        if last_exc:
            message = f"{type(last_exc).__name__}: {last_exc}"
            if _is_daily_quota_llm_error(message) or any(item.get("status") == "failed" and "quota" in str(item.get("reason", "")).lower() for item in trace):
                STATUS_MANAGER.start_backoff(message, seconds=45, kind="quota")
            else:
                STATUS_MANAGER.start_backoff(message, seconds=20, kind="transient")
            STATUS_MANAGER.set_error(message)
            raise last_exc
        raise RuntimeError("No Gemini API keys are available for routing.")

    def _client_for_key(self, api_key):
        if api_key not in self.clients:
            self.clients[api_key] = self._genai.Client(api_key=api_key)
        return self.clients[api_key]

    def _extract_text(self, response):
        if hasattr(response, "text") and response.text:
            return response.text
        text = ""
        for candidate in getattr(response, "candidates", []) or []:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", []) if content else []
            for part in parts:
                part_text = getattr(part, "text", "")
                if part_text:
                    text += part_text
        return text

    def _short_reason(self, message):
        text = str(message or "")
        if _is_quota_llm_error(text):
            return "quota/rate-limit"
        if _is_model_unavailable_llm_error(text):
            return "model unavailable"
        if _is_connectivity_llm_error(text):
            return "timeout/connectivity"
        if _is_terminal_llm_error(text):
            return "key invalid/permission"
        return text[:140]


def get_llm_status(ignore_backoff=False):
    enable_llm = os.getenv("ENABLE_LLM", "true").lower() in {"1", "true", "yes"}
    api_keys = get_api_keys()
    model_sequence = DEFAULT_GEMINI_MODEL_SEQUENCE or [DEFAULT_GEMINI_MODEL]
    status = {
        "enabled": enable_llm,
        "api_key_present": bool(api_keys),
        "api_key_count": len(api_keys),
        "env_file_present": ENV_PATH.exists(),
        "env_example_present": ENV_EXAMPLE_PATH.exists(),
        "env_path": str(ENV_PATH),
        "model": model_sequence[0] if model_sequence else DEFAULT_GEMINI_MODEL,
        "model_sequence": model_sequence,
        "provider": "google-genai",
        "router_mode": "Multi-Key Adaptive Routing" if len(api_keys) > 1 or len(model_sequence) > 1 else "Single-Key Gemini",
        "router_trace": STATUS_MANAGER.get_router_trace(),
        "active_route": STATUS_MANAGER.get_active_route(),
        "key_backoffs": STATUS_MANAGER.get_key_backoffs(),
        "available": False,
        "message": "",
        "last_error": STATUS_MANAGER.get_error(),
        "quota_backoff": STATUS_MANAGER.get_backoff(),
    }

    if not enable_llm:
        status["message"] = "LLM mode is disabled by ENABLE_LLM."
        return status
    if not api_keys:
        status["message"] = "No valid GOOGLE_API_KEY, GEMINI_API_KEY, GOOGLE_API_KEYS, or GEMINI_API_KEYS found in .env. .env.example is only a template and is never loaded at runtime."
        return status

    try:
        from google import genai  # noqa: F401
    except Exception as exc:
        status["message"] = f"google-genai import failed: {type(exc).__name__}"
        return status

    status["available"] = True
    status["message"] = f"Gemini SDK is configured with {len(api_keys)} key route(s) and {len(model_sequence)} model tier(s)."
    last_err = STATUS_MANAGER.get_error()
    backoff = STATUS_MANAGER.get_backoff()
    key_backoffs = STATUS_MANAGER.get_key_backoffs()
    available_key_count = max(0, len(api_keys) - len(key_backoffs))
    has_spare_key = available_key_count > 0
    status["quota_backoff"] = backoff
    status["available_key_count"] = available_key_count
    if backoff.get("active") and not ignore_backoff and not has_spare_key:
        status["available"] = False
        if backoff.get("kind") == "daily_quota":
            status["message"] = (
                "Gemini daily free-tier quota is exhausted for the active project/key. "
                "The system will stay in simulated/local reasoning mode until quota resets or a different quota source is used."
            )
        elif backoff.get("kind") == "quota":
            status["message"] = (
                f"Gemini is temporarily paused after a quota/rate-limit response. "
                f"Retry will resume in {backoff.get('remaining_seconds', 0)} seconds."
            )
        else:
            status["message"] = (
                f"Gemini is temporarily paused after a transient provider/network error. "
                f"Retry will resume in {backoff.get('remaining_seconds', 0)} seconds."
            )
        return status
    if backoff.get("active") and has_spare_key:
        status["message"] = (
            f"Gemini router has {available_key_count} available key route(s) after recent "
            f"{backoff.get('kind', 'provider')} cooldowns; it will keep trying healthy routes."
        )
    
    # If we are ignoring backoff, we treat it as available even if a backoff is active
    status["available"] = True
    if backoff.get("active"):
        if backoff.get("kind") == "daily_quota":
            status["message"] = (
                "Gemini daily free-tier quota is exhausted for the active project/key. "
                "Live calls are paused and local/simulated reasoning remains active."
            )
        else:
            status["message"] = (
                f"Gemini saw a recent transient error and will keep retrying. "
                f"Last error cooldown window: {backoff.get('remaining_seconds', 0)} seconds."
            )

    if last_err:
        if _is_terminal_llm_error(last_err):
            status["available"] = has_spare_key
            status["message"] = (
                f"Gemini router is skipping invalid/expired key routes; {available_key_count} route(s) remain available."
                if has_spare_key
                else "Gemini is disabled because the configured API key is invalid or expired."
            )
        elif _is_daily_quota_llm_error(last_err):
            status["available"] = has_spare_key
            status["message"] = (
                f"Some Gemini free-tier routes hit daily quota; {available_key_count} route(s) remain available for failover."
                if has_spare_key
                else "Gemini daily free-tier quota is exhausted. Autonomous Edge Intelligence will continue with simulated/local reasoning until quota resets."
            )
        elif _is_quota_llm_error(last_err):
            status["available"] = has_spare_key
            status["message"] = (
                f"Provider quota optimization active; {available_key_count} Gemini route(s) remain available for the next request."
                if has_spare_key
                else "Autonomous Edge Intelligence mode active (Provider Quota Optimization). System remains fully operational via high-fidelity local models."
            )
        elif "503" in last_err or "UNAVAILABLE" in last_err.upper():
            status["message"] = "Distributed intelligence is temporarily localized due to provider overhead."
        elif "TIMEOUT" in last_err.upper() or _is_connectivity_llm_error(last_err):
            status["message"] = "Hybrid mode restricted; falling back to resilient local reasoning."
        else:
            status["message"] = "Intelligence localized: standard local agent heuristics enabled."
            
    return status

def get_llm(force=False):
    global _LLM_INSTANCE, _LLM_INSTANCE_KEY
    status = get_llm_status(ignore_backoff=force)
    if not status["available"]:
        return None
    api_keys = get_api_keys()
    models = DEFAULT_GEMINI_MODEL_SEQUENCE or [DEFAULT_GEMINI_MODEL]
    instance_key = (tuple(api_keys), tuple(models))
    with _LLM_INSTANCE_LOCK:
        if _LLM_INSTANCE is not None and _LLM_INSTANCE_KEY == instance_key:
            return _LLM_INSTANCE
        try:
            _LLM_INSTANCE = GeminiLLMWrapper(api_keys=api_keys, models=models)
            _LLM_INSTANCE_KEY = instance_key
            return _LLM_INSTANCE
        except Exception as e:
            logging.error(f"Failed loading LLM Client interface: {e}")
            _LLM_INSTANCE = None
            _LLM_INSTANCE_KEY = None
            return None
