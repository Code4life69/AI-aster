from __future__ import annotations

import re


SECRET_PATTERNS = [
    (re.compile(r"(?i)(api[_-]?key\s*[=:]\s*[\"']?)([A-Za-z0-9_\-]{8,})"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(token\s*[=:]\s*[\"']?)([A-Za-z0-9_\-]{8,})"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(password\s*[=:]\s*[\"']?)([^\"'\s]+)"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(secret\s*[=:]\s*[\"']?)([^\"'\s]+)"), r"\1[REDACTED]"),
]


def redact_secrets(text: str) -> str:
    cleaned = text
    for pattern, replacement in SECRET_PATTERNS:
        cleaned = pattern.sub(replacement, cleaned)
    return cleaned


def seems_binary(data: bytes) -> bool:
    if b"\x00" in data[:2048]:
        return True
    if not data:
        return False
    text_bytes = sum(byte in b"\t\n\r\f\b" or 32 <= byte <= 126 for byte in data[:2048])
    return text_bytes / min(len(data[:2048]), 2048) < 0.75
