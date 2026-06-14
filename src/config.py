"""Configuration loading and validation.

Loads `config.yaml`, applies CLI overrides, exposes a typed dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@dataclass
class Config:
    p3sam_model_path: str = "tencent/Hunyuan3D-Part"
    xpart_model_path: str = "tencent/Hunyuan3D-Part"
    enable_xpart: bool = False
    export_mode: str = "merged"
    input_dir: str = "input"
    output_dir: str = "output"
    device: str = "cuda"
    log_level: str = "INFO"

    def validate(self) -> None:
        if self.export_mode not in {"merged", "split"}:
            raise ValueError(
                f"export_mode must be 'merged' or 'split', got {self.export_mode!r}"
            )
        if self.device not in {"cuda", "cpu"}:
            raise ValueError(f"device must be 'cuda' or 'cpu', got {self.device!r}")
        if self.log_level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
            raise ValueError(f"invalid log_level: {self.log_level!r}")

    def resolved_input_dir(self) -> Path:
        return _resolve(self.input_dir)

    def resolved_output_dir(self) -> Path:
        return _resolve(self.output_dir)


def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def load_config(path: Path | str | None = None) -> Config:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    data: dict[str, Any] = {}
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    known = {f.name for f in fields(Config)}
    filtered = {k: v for k, v in data.items() if k in known}
    cfg = Config(**filtered)
    cfg.validate()
    return cfg


def apply_overrides(cfg: Config, overrides: dict[str, Any]) -> Config:
    """Return a new Config with non-None override values applied."""
    known = {f.name for f in fields(Config)}
    payload = {k: getattr(cfg, k) for k in known}
    for k, v in overrides.items():
        if v is None:
            continue
        if k not in known:
            raise KeyError(f"unknown config field: {k}")
        payload[k] = v
    new = Config(**payload)
    new.validate()
    return new
