"""Small file-backed API key rotation helpers."""

from __future__ import annotations

import os
import threading
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_LOCK = threading.Lock()

ENV_BY_POOL = {
    "openai": (
        "OPENAI_API_KEY",
        "REASONING_API_KEY",
        "VERIFIER_API_KEY",
        "EXPERIENCE_API_KEY",
        "EXPERIENCE_EMBEDDING_API_KEY",
    ),
    "serp": ("SERPAPI_KEY",),
    "jina": ("JINA_API_KEY",),
}

KEY_FILE_ENV = {
    "openai": "OPENAI_KEY_FILE",
    "serp": "SERPAPI_KEY_FILE",
    "jina": "JINA_KEY_FILE",
}

OPENAI_ROTATE_MARKERS = (
    "每小时请求数已达上限",
    "requests per hour",
    "hourly request",
    "request quota",
    "quota",
    "insufficient_quota",
    "credit",
    "billing",
)

SERP_ROTATE_MARKERS = (
    "quota",
    "credit",
    "balance",
    "limit exceeded",
    "payment required",
    "insufficient",
    "unauthorized",
    "forbidden",
)

JINA_ROTATE_MARKERS = (
    "402",
    "40203",
    "insufficientbalanceerror",
    "account balance not enough",
    "quota",
    "credit",
    "balance",
)


def resolve_key_file(raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    path = Path(raw_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def configure_key_file(pool: str, raw_path: str | None) -> str:
    path = resolve_key_file(raw_path)
    if path is None:
        return ""
    os.environ[KEY_FILE_ENV[pool]] = str(path)
    return current_key(pool)


def _key_file(pool: str) -> Path | None:
    return resolve_key_file(os.environ.get(KEY_FILE_ENV[pool]))


def _read_keys(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def current_key(pool: str) -> str:
    path = _key_file(pool)
    if path is None:
        return ""
    with _LOCK:
        keys = _read_keys(path)
        if not keys:
            raise RuntimeError(f"No usable {pool} key left in {path}")
        key = keys[0]
    apply_key(pool, key)
    return key


def apply_key(pool: str, key: str) -> None:
    for env_name in ENV_BY_POOL[pool]:
        os.environ[env_name] = key


def _masked_key(key: str | None) -> str:
    if key and len(key) > 10:
        return f"{key[:6]}...{key[-4:]}"
    return "<current>"


def rotate_current_key(pool: str, current_key: str | None = None) -> str:
    """Move a temporarily limited key to the end of the file and use the next key."""
    path = _key_file(pool)
    if path is None:
        return ""

    with _LOCK:
        keys = _read_keys(path)
        if not keys:
            raise RuntimeError(f"No usable {pool} key left in {path}")

        rotated_key = current_key if current_key in keys else keys[0]
        if len(keys) > 1:
            keys.remove(rotated_key)
            keys.append(rotated_key)
            path.write_text("\n".join(keys) + "\n", encoding="utf-8")
        next_key = keys[0]

    apply_key(pool, next_key)
    print(f"[KeyPool] Rotated {pool} key {_masked_key(rotated_key)}; switched to next key.")
    return next_key


def retire_current_key(pool: str, exhausted_key: str | None = None) -> str:
    path = _key_file(pool)
    if path is None:
        return ""

    with _LOCK:
        keys = _read_keys(path)
        if exhausted_key and exhausted_key in keys:
            keys.remove(exhausted_key)
        elif keys:
            keys.pop(0)
        path.write_text("\n".join(keys) + ("\n" if keys else ""), encoding="utf-8")
        if not keys:
            raise RuntimeError(f"No usable {pool} key left in {path}")
        next_key = keys[0]

    apply_key(pool, next_key)
    print(f"[KeyPool] Retired {pool} key {_masked_key(exhausted_key)}; switched to next key.")
    return next_key


def openai_should_rotate(status_code: int | None, message: str) -> bool:
    text = f"{status_code or ''} {message or ''}".lower()
    return any(marker.lower() in text for marker in OPENAI_ROTATE_MARKERS)


def serp_should_rotate(status_code: int | None, message: str) -> bool:
    text = f"{status_code or ''} {message or ''}".lower()
    return status_code in {401, 402, 403, 429} or any(marker in text for marker in SERP_ROTATE_MARKERS)


def jina_should_rotate(status_code: int | None, message: str) -> bool:
    text = f"{status_code or ''} {message or ''}".lower()
    return status_code == 402 or any(marker in text for marker in JINA_ROTATE_MARKERS)
