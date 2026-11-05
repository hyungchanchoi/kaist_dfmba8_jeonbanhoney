"""설정 로딩: config/config.toml 을 읽고, config/local.toml 이 있으면 그 값으로 덮어씁니다."""
from __future__ import annotations

import copy
import tomllib
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "config.toml"


def _deep_update(base: dict, new: dict) -> dict:
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


class Config:
    """설정 값 묶음. cfg["collect"]["start_date"] 처럼 읽고, cfg.path("raw_videos") 로 경로를 얻습니다."""

    def __init__(self, data: dict[str, Any], root: Path):
        self.data = data
        self.root = Path(root)

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def section(self, name: str) -> dict[str, Any]:
        return self.data.get(name, {})

    def path(self, key: str) -> Path:
        value = self.data["paths"].get(key, "")
        if not value:
            raise FileNotFoundError(
                f"설정 paths.{key} 가 비어 있습니다. config/local.toml 에 경로를 적어 주세요."
            )
        p = Path(value)
        return p if p.is_absolute() else self.root / p

    def has_path(self, key: str) -> bool:
        return bool(self.data["paths"].get(key, ""))


def load_config(path: str | Path | None = None) -> Config:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path, "rb") as f:
        data = tomllib.load(f)
    root = cfg_path.resolve().parents[1]
    local = cfg_path.with_name("local.toml")
    if local.exists():
        with open(local, "rb") as f:
            _deep_update(data, tomllib.load(f))
    return Config(copy.deepcopy(data), root)
