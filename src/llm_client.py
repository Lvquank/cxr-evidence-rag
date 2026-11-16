"""Small, production-ready client for one-shot LLM report generation.

The client deliberately keeps the request stateless: classifier/retriever evidence
is sent once, not copied into a second ``context`` object. It uses the OpenAI
Responses API through the standard library, so no extra SDK dependency is needed.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import mimetypes
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


def _load_project_env() -> None:
    """Load the repository .env for command-line callers without logging secrets."""
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path, override=False)
    except ImportError:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass
class LLMConfig:
    api_key: Optional[str] = None
    base_url: str = "https://api.openai.com/v1"
    # Luna is the high-volume/cost-sensitive default. Explicit models are kept.
    model: str = "gpt-5.6-luna"
    temperature: Optional[float] = None
    max_tokens: int = 8192
    reasoning_effort: str = "high"
    timeout: int = 120
    retry_attempts: int = 3
    retry_delay: float = 1.0
    cache_enabled: bool = True
    cache_dir: str = "data/.cache/llm"
    dry_run: bool = False


@dataclass
class LLMResponse:
    text: str
    response_id: Optional[str] = None
    model: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    cached: bool = False


class LLMClientBase:
    """Small compatibility interface for existing callers."""

    def send_request(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        *,
        system: Optional[str] = None,
        image_path: Optional[str] = None,
        image_detail: str = "high",
        response_schema: Optional[Dict[str, Any]] = None,
    ) -> str:
        raise NotImplementedError

    def send_request_with_context(
        self, prompt: str, evidence: Dict[str, Any], **kwargs: Any
    ) -> str:
        raise NotImplementedError


class OpenAIResponsesClient(LLMClientBase):
    """OpenAI Responses API client with local exact-request caching."""

    def __init__(self, config: LLMConfig):
        self.config = config
        _load_project_env()
        self.config.api_key = (
            self.config.api_key
            or os.getenv("LLM_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("GPT_TOKEN")  # Legacy project variable; prefer OPENAI_API_KEY for new setups.
        )
        self.session_history: List[Dict[str, Any]] = []
        self.last_response: Optional[LLMResponse] = None
        self._cache_dir = Path(self.config.cache_dir)

    def send_request(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        *,
        system: Optional[str] = None,
        image_path: Optional[str] = None,
        image_detail: str = "high",
        response_schema: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate one response. ``context`` remains for legacy callers only."""
        if context is not None:
            prompt = f"{prompt}\n{json.dumps(context, ensure_ascii=False, separators=(',', ':'))}"
        response = self._request(
            system or "",
            prompt,
            image_path=image_path,
            image_detail=image_detail,
            response_schema=response_schema,
        )
        self.last_response = response
        self.session_history.append({
            "response_id": response.response_id,
            "model": response.model or self.config.model,
            "usage": response.usage,
            "cached": response.cached,
            "timestamp": time.time(),
        })
        return response.text

    def send_request_with_context(
        self, prompt: str, evidence: Dict[str, Any], **kwargs: Any
    ) -> str:
        """Legacy helper; use ``send_request`` when evidence is already in prompt."""
        return self.send_request(
            prompt,
            context=evidence,
            system=kwargs.get("system"),
            image_path=kwargs.get("image_path"),
            image_detail=kwargs.get("image_detail", "high"),
            response_schema=kwargs.get("response_schema"),
        )

    def _cache_key(
        self,
        system: str,
        prompt: str,
        image_fingerprint: Optional[str],
        image_detail: str,
        response_schema: Optional[Dict[str, Any]],
    ) -> str:
        payload = json.dumps(
            {
                "model": self.config.model,
                "reasoning_effort": self.config.reasoning_effort,
                "temperature": self.config.temperature,
                "system": system,
                "prompt": prompt,
                "max_tokens": self.config.max_tokens,
                "image_fingerprint": image_fingerprint,
                "image_detail": image_detail,
                "response_schema": response_schema,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.json"

    def _load_cache(self, key: str) -> Optional[LLMResponse]:
        if not self.config.cache_enabled:
            return None
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return LLMResponse(**data, cached=True)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            logger.warning("Ignoring unreadable LLM cache entry: %s", path)
            return None

    def _save_cache(self, key: str, response: LLMResponse) -> None:
        if not self.config.cache_enabled:
            return
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        payload = asdict(response)
        payload["cached"] = False
        self._cache_path(key).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def _image_input(image_path: str) -> tuple[str, str]:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"LLM image input not found: {path}")
        mime_type = mimetypes.guess_type(path.name)[0]
        if mime_type not in {"image/png", "image/jpeg", "image/webp", "image/gif"}:
            raise ValueError(f"Unsupported image type for LLM input: {path.suffix or path.name}")
        content = path.read_bytes()
        fingerprint = hashlib.sha256(content).hexdigest()
        encoded = base64.b64encode(content).decode("ascii")
        return f"data:{mime_type};base64,{encoded}", fingerprint

    def _request(
        self,
        system: str,
        prompt: str,
        *,
        image_path: Optional[str] = None,
        image_detail: str = "high",
        response_schema: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        if image_detail not in {"low", "high", "original", "auto"}:
            raise ValueError("image_detail must be low, high, original, or auto.")
        image_url, image_fingerprint = (None, None)
        if image_path:
            image_url, image_fingerprint = self._image_input(image_path)
        key = self._cache_key(system, prompt, image_fingerprint, image_detail, response_schema)
        cached = self._load_cache(key)
        if cached is not None:
            logger.info("LLM exact-request cache hit")
            return cached
        if self.config.dry_run:
            return LLMResponse(text="[dry-run: request was not sent]", model=self.config.model)
        if not self.config.api_key:
            raise RuntimeError("Set LLM_API_KEY or OPENAI_API_KEY before calling the LLM API (GPT_TOKEN is accepted for this legacy project).")

        payload: Dict[str, Any] = {
            "model": self.config.model,
            "instructions": system or None,
            "input": prompt,
            "max_output_tokens": self.config.max_tokens,
            "store": False,
            "prompt_cache_key": "image-rag-radiology-v5",
        }
        if self.config.reasoning_effort:
            payload["reasoning"] = {"effort": self.config.reasoning_effort}
        if self.config.temperature is not None:
            payload["temperature"] = self.config.temperature
        if response_schema is not None:
            payload["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "radiology_prediction",
                    "strict": True,
                    "schema": response_schema,
                }
            }
        if image_url:
            payload["input"] = [{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": image_url, "detail": image_detail},
                ],
            }]
        payload = {key: value for key, value in payload.items() if value is not None}

        url = f"{self.config.base_url.rstrip('/')}/responses"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        retryable_statuses = {408, 409, 429, 500, 502, 503, 504}
        last_error: Optional[Exception] = None
        for attempt in range(self.config.retry_attempts):
            try:
                request = Request(
                    url,
                    data=body,
                    headers={
                        "Authorization": f"Bearer {self.config.api_key}",
                        "Content-Type": "application/json",
                    },
                    method="POST",
                )
                with urlopen(request, timeout=self.config.timeout) as raw_response:
                    result = json.loads(raw_response.read().decode("utf-8"))
                response = LLMResponse(
                    text=self._extract_output_text(result),
                    response_id=result.get("id"),
                    model=result.get("model", self.config.model),
                    usage=result.get("usage"),
                )
                self._save_cache(key, response)
                return response
            except HTTPError as error:
                last_error = error
                if error.code not in retryable_statuses:
                    detail = error.read().decode("utf-8", errors="replace")
                    raise RuntimeError(f"LLM API HTTP {error.code}: {detail}") from error
            except (URLError, TimeoutError, json.JSONDecodeError) as error:
                last_error = error
            if attempt + 1 < self.config.retry_attempts:
                time.sleep(self.config.retry_delay * (2**attempt))
        raise RuntimeError(f"LLM request failed after {self.config.retry_attempts} attempts: {last_error}")

    @staticmethod
    def _extract_output_text(result: Dict[str, Any]) -> str:
        if result.get("status") == "incomplete":
            reason = (result.get("incomplete_details") or {}).get("reason", "unknown")
            raise RuntimeError(f"LLM API response incomplete (reason={reason}).")
        if isinstance(result.get("output_text"), str) and result["output_text"].strip():
            return result["output_text"].strip()
        parts: List[str] = []
        for item in result.get("output", []):
            for content in item.get("content", []):
                if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                    parts.append(content["text"])
        text = "".join(parts).strip()
        if not text:
            incomplete_reason = (result.get("incomplete_details") or {}).get("reason")
            usage = result.get("usage") or {}
            output_details = usage.get("output_tokens_details") or {}
            diagnostics = [f"status={result.get('status', 'unknown')}"]
            if incomplete_reason:
                diagnostics.append(f"incomplete_reason={incomplete_reason}")
            if usage.get("output_tokens") is not None:
                diagnostics.append(f"output_tokens={usage['output_tokens']}")
            if output_details.get("reasoning_tokens") is not None:
                diagnostics.append(f"reasoning_tokens={output_details['reasoning_tokens']}")
            raise RuntimeError("LLM API returned no output text (" + ", ".join(diagnostics) + ").")
        return text

    def get_last_response_metadata(self) -> Optional[Dict[str, Any]]:
        return asdict(self.last_response) if self.last_response else None

    def get_session_history(self) -> List[Dict[str, Any]]:
        return list(self.session_history)


class LLMClientFactory:
    @staticmethod
    def create_client(model_name: Optional[str] = None, config: Optional[LLMConfig] = None) -> LLMClientBase:
        config = config or LLMConfig(model=model_name or LLMConfig.model)
        if model_name:
            config.model = model_name
        return OpenAIResponsesClient(config)


def create_llm_client(
    model: str = "gpt-5.6-luna",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: int = 8192,
    **kwargs: Any,
) -> LLMClientBase:
    """Create an OpenAI Responses client. Extra LLMConfig fields are accepted."""
    config = LLMConfig(
        model=model,
        api_key=api_key,
        base_url=base_url or "https://api.openai.com/v1",
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )
    return LLMClientFactory.create_client(config=config)
