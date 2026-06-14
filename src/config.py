"""Configuration loading and validation.

Loads `config.yaml`, applies CLI overrides, exposes a typed dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@dataclass
class Config:
    hy3d_part_root: str = ""
    p3sam_ckpt_path: str = ""
    python_executable: str = ""

    export_mode: str = "merged"
    input_dir: str = "input"
    output_dir: str = "output"

    p3sam_point_num: int = 100000
    p3sam_threshold: float = 0.95
    p3sam_seed: int = 42
    p3sam_clean_mesh: int = 1
    p3sam_post_process: int = 0

    log_level: str = "INFO"

    def validate(self) -> None:
        if self.export_mode not in {"merged", "split"}:
            raise ValueError(
                f"export_mode must be 'merged' or 'split', got {self.export_mode!r}"
            )
        if self.log_level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
            raise ValueError(f"invalid log_level: {self.log_level!r}")
        if not self.hy3d_part_root:
            raise ValueError(
                "hy3d_part_root is empty — set it to your local clone of "
                "Tencent-Hunyuan/Hunyuan3D-Part (see INSTALL.md)."
            )
        if not self.p3sam_ckpt_path:
            raise ValueError(
                "p3sam_ckpt_path is empty — point it at p3sam.safetensors "
                "(or last.ckpt) (see INSTALL.md §4 Poids)."
            )

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
