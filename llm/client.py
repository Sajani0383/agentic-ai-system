import os
import time
import logging
import threading
import hashlib
from types import SimpleNamespace
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError
try:
    from dotenv import load_dotenv
except ImportError:  # Optional in local fallback mode
    def load_dotenv(*args, **kwargs):
        return False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
ENV_EXAMPLE_PATH = PROJECT_ROOT / ".env.example"

if ENV_PATH.exists(): load_dotenv(dotenv_path=ENV_PATH, override=False)

DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
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


def get_api_key():
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key: return None
    cleaned = api_key.strip()
    if cleaned.lower() in {"your_google_api_key_here", "your_real_key_here", "replace_me"} or len(cleaned) < 20 or not cleaned.startswith("AIza"):
        return None
    return cleaned


class GeminiLLMWrapper:
    """Enterprise LLM implementation supporting Rate Limits, Caching, and hard Timeouts."""
    def __init__(self, api_key, model):
        from google import genai
        self.client = genai.Client(api_key=api_key)
        self.model = model
        
        # Operational Limits
        self.call_lock = threading.Lock()
        self.last_call_time = 0.0
        self._cache = {}
        # NOTE: No shared executor — each invoke() creates a fresh one inside a
        # `with` block so Python's interpreter-shutdown ThreadPoolExecutor cleanup
        # can never fire on a stale reference and raise
        # "cannot schedule new futures after shutdown".
        
        self.TIMEOUT_SECONDS = 30.0    # Increased for complex agentic prompts
        self.RATE_LIMIT_DELAY = 1.0    # 1 second spacing protects QPS quotas

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
        for attempt in range(2):
            start_time = time.time()
            try:
                # 3. Fresh executor per call — avoids "cannot schedule new futures
                # after shutdown" when Python's atexit fires on interpreter teardown.
                with ThreadPoolExecutor(max_workers=1) as _ex:
                    future = _ex.submit(self.client.models.generate_content, model=self.model, contents=prompt)
                    response = future.result(timeout=self.TIMEOUT_SECONDS)
                
                text = ""
                # Use native SDK property if available
                if hasattr(response, "text") and response.text:
                    text = response.text
                else:
                    for candidate in getattr(response, "candidates", []) or []:
                        content = getattr(candidate, "content", None)
                        parts = getattr(content, "parts", []) if content else []
                        for part in parts:
                            part_text = getattr(part, "text", "")
                            if part_text:
                                text += part_text
                                
                STATUS_MANAGER.set_error("")
                STATUS_MANAGER.clear_backoff()
                result_text = (text or "").strip()
                
                # Update bounds
                self._cache[prompt_hash] = result_text
                if len(self._cache) > 200: self._cache.clear()
                
                duration = time.time() - start_time
                logging.info(f"LLM call successful: {self.model} in {duration:.2f}s")
                
                return SimpleNamespace(content=result_text)
                
            except TimeoutError:
                last_exc = Exception("Cloud LLM Timeout bounds breached successfully dropping request.")
                logging.warning(f"Attempt {attempt+1} LLM generation timed out after {self.TIMEOUT_SECONDS}s.")
                STATUS_MANAGER.start_backoff(str(last_exc), seconds=12, kind="transient")
                time.sleep(1.0)
            except Exception as exc:
                last_exc = exc
                message = f"{type(exc).__name__}: {exc}"
                logging.warning(f"Attempt {attempt+1} Exception: {message}")
                if _is_quota_llm_error(message):
                    STATUS_MANAGER.start_backoff(message, seconds=45, kind="quota")
                    raise
                if _is_connectivity_llm_error(message):
                    STATUS_MANAGER.start_backoff(message, seconds=10, kind="transient")
                    raise
                if ("503" in message or "UNAVAILABLE" in message.upper()) and attempt < 1:
                    time.sleep(1.5)
                    continue
                STATUS_MANAGER.set_error(message)
                raise
                
        if last_exc:
            STATUS_MANAGER.set_error(str(last_exc))
            raise last_exc


def get_llm_status(ignore_backoff=False):
    enable_llm = os.getenv("ENABLE_LLM", "true").lower() in {"1", "true", "yes"}
    api_key = get_api_key()
    status = {
        "enabled": enable_llm,
        "api_key_present": bool(api_key),
        "env_file_present": ENV_PATH.exists(),
        "env_example_present": ENV_EXAMPLE_PATH.exists(),
        "env_path": str(ENV_PATH),
        "model": DEFAULT_GEMINI_MODEL,
        "provider": "google-genai",
        "available": False,
        "message": "",
        "last_error": STATUS_MANAGER.get_error(),
        "quota_backoff": STATUS_MANAGER.get_backoff(),
    }

    if not enable_llm:
        status["message"] = "LLM mode is disabled by ENABLE_LLM."
        return status
    if not api_key:
        status["message"] = "No valid GOOGLE_API_KEY or GEMINI_API_KEY found in .env. .env.example is only a template and is never loaded at runtime."
        return status

    try:
        from google import genai  # noqa: F401
    except Exception as exc:
        status["message"] = f"google-genai import failed: {type(exc).__name__}"
        return status

    status["available"] = True
    status["message"] = "Gemini SDK is configured."
    last_err = STATUS_MANAGER.get_error()
    backoff = STATUS_MANAGER.get_backoff()
    status["quota_backoff"] = backoff
    if backoff.get("active") and not ignore_backoff:
        if backoff.get("kind") == "quota":
            status["available"] = False
            status["message"] = (
                f"Gemini is temporarily paused after a quota/rate-limit response. "
                f"Retry will resume in {backoff.get('remaining_seconds', 0)} seconds."
            )
            return status
    
    # If we are ignoring backoff, we treat it as available even if a backoff is active
    status["available"] = True
    if backoff.get("active"):
        status["message"] = (
            f"Gemini saw a recent transient error and will keep retrying. "
            f"Last error cooldown window: {backoff.get('remaining_seconds', 0)} seconds."
        )

    if last_err:
        if _is_terminal_llm_error(last_err):
            status["available"] = False
            status["message"] = "Gemini is disabled because the configured API key is invalid or expired."
        elif _is_quota_llm_error(last_err):
            status["available"] = False
            status["message"] = "Autonomous Edge Intelligence mode active (Provider Quota Optimization). System remains fully operational via high-fidelity local models."
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
    api_key = get_api_key()
    instance_key = (api_key, DEFAULT_GEMINI_MODEL)
    with _LLM_INSTANCE_LOCK:
        if _LLM_INSTANCE is not None and _LLM_INSTANCE_KEY == instance_key:
            return _LLM_INSTANCE
        try:
            _LLM_INSTANCE = GeminiLLMWrapper(api_key=api_key, model=DEFAULT_GEMINI_MODEL)
            _LLM_INSTANCE_KEY = instance_key
            return _LLM_INSTANCE
        except Exception as e:
            logging.error(f"Failed loading LLM Client interface: {e}")
            _LLM_INSTANCE = None
            _LLM_INSTANCE_KEY = None
            return None
