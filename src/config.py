from __future__ import annotations
from pathlib import Path
import yaml

def load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p.resolve()}")

    text = p.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Config is empty: {p.resolve()}")

    cfg = yaml.safe_load(text)
    if cfg is None:
        raise ValueError(f"Config parsed to None (check YAML): {p.resolve()}")

    if not isinstance(cfg, dict):
        raise TypeError(f"Config must be a dict, got {type(cfg)}")

    return cfg
