from __future__ import annotations

import json
import os
from typing import Any

import requests


class OpenAIResponsesTransport:
    def __init__(self, model: str, api_key_env: str) -> None:
        self.model = model
        self.api_key_env = api_key_env

    def generate(self, messages: list[dict[str, str]], schema: dict[str, Any]) -> str:
        api_key = os.getenv(self.api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(f"Missing API key in environment variable {self.api_key_env}")
        payload = {
            "model": self.model,
            "input": messages,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema["name"],
                    "strict": schema["strict"],
                    "schema": schema["schema"],
                }
            },
        }
        response = requests.post(
            "https://api.openai.com/v1/responses",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps(payload),
            timeout=180,
        )
        response.raise_for_status()
        data = response.json()
        output_text = data.get("output_text")
        if output_text:
            return output_text
        for item in data.get("output", []):
            for content in item.get("content", []):
                text = content.get("text")
                if text:
                    return text
        raise RuntimeError("No text output returned by the OpenAI Responses API")
