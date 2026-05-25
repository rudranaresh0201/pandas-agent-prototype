import logging
import os
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_MODEL = "meta-llama/llama-3.1-8b-instruct"
_API_URL = "https://openrouter.ai/api/v1/chat/completions"
_MAX_RETRIES = 3


class LLMClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.model = _MODEL
        self.api_url = _API_URL

    def __repr__(self) -> str:
        masked = self.api_key[:8] + "..." if self.api_key else "NOT SET"
        return f"LLMClient(model='{self.model}', key='{masked}')"

    def call(self, prompt: str, temperature: float = 0, max_tokens: int = 500) -> str:
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set in environment")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        last_error: Optional[Exception] = None
        for attempt in range(_MAX_RETRIES):
            try:
                resp = requests.post(
                    self.api_url, headers=headers, json=payload, timeout=30
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except Exception as exc:
                last_error = exc
                logger.warning(f"LLM attempt {attempt + 1}/{_MAX_RETRIES} failed: {exc}")
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(2**attempt)

        raise RuntimeError(
            f"LLM call failed after {_MAX_RETRIES} attempts: {last_error}"
        )
