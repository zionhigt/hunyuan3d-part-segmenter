"""CLI: segment a single GLB.

Example:
    python src/single.py --glb input/vehicle.glb
    python src/single.py --glb input/vehicle.glb --export-mode split
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.config import Config, apply_overrides, load_config
from src.export import export_parts
from src.segmenter import PartSegmenter, UpstreamNotInstalled


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Segment a single GLB into parts.")
    p.add_argument("--glb", required=True, help="Path to input .glb")
    p.add_argument("--config", default=None, help="Path to config.yaml (optional)")
    p.add_argument("--hy3d-part-root", default=None, help="Override hy3d_part_root")
    p.add_argument("--p3sam-ckpt-path", default=None, help="Override p3sam_ckpt_path")
    p.add_argument("--output-dir", default=None, help="Override output_dir")
    p.add_argument(
        "--export-mode",
        choices=["merged", "split"],
        default=None,
        help="Override export_mode",
    )
    p.add_argument("--log-level", default=None)
    return p


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


def run(cfg: Config, glb_path: Path) -> int:
    output_dir = cfg.resolved_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        segmenter = PartSegmenter(
            hy3d_part_root=cfg.hy3d_part_root,
            p3sam_ckpt_path=cfg.p3sam_ckpt_path,
            python_executable=cfg.python_executable,
            point_num=cfg.p3sam_point_num,
            threshold=cfg.p3sam_threshold,
            seed=cfg.p3sam_seed,
            clean_mesh=cfg.p3sam_clean_mesh,
            post_process=cfg.p3sam_post_process,
            prompt_bs=cfg.p3sam_prompt_bs,
        )
    except UpstreamNotInstalled as exc:
        logging.error("%s", exc)
        return 2

    parts = segmenter.segment(glb_path)
    export_parts(parts, output_dir, stem=glb_path.stem, mode=cfg.export_mode)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = load_config(args.config)
    cfg = apply_overrides(
        cfg,
        {
            "hy3d_part_root": args.hy3d_part_root,
            "p3sam_ckpt_path": args.p3sam_ckpt_path,
            "output_dir": args.output_dir,
            "export_mode": args.export_mode,
            "log_level": args.log_level,
        },
    )
    _configure_logging(cfg.log_level)

    glb = Path(args.glb)
    if not glb.is_file():
        logging.error("input file not found: %s", glb)
        return 1
    return run(cfg, glb)


if __name__ == "__main__":
    sys.exit(main())
