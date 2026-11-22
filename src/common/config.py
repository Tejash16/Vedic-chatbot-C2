"""Read shared/config.yaml (written by session C)."""
from __future__ import annotations

import functools

import yaml

from src.common import load as L

CONFIG = L.ROOT / "shared" / "config.yaml"


@functools.lru_cache(maxsize=1)
def config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def table(table_id: str) -> dict:
    return config()["tables"][table_id]
