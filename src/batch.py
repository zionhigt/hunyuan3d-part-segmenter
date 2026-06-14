"""CLI: batch-segment every .glb under input_dir.

Skips files already segmented in output_dir (resume-friendly).
Failures on individual files are logged but do not stop the batch.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from tqdm import tqdm

from src.config import Config, apply_overrides, load_config
from src.export import export_parts
from src.segmenter import PartSegmenter, UpstreamNotInstalled


logger = logging.getLogger("batch")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batch-segment all GLBs under input/.")
    p.add_argument("--config", default=None)
    p.add_argument("--hy3d-part-root", default=None)
    p.add_argument("--p3sam-ckpt-path", default=None)
    p.add_argument("--input-dir", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--export-mode", choices=["merged", "split"], default=None)
    p.add_argument("--log-level", default=None)
    p.add_argument(
        "--force",
        action="store_true",
        help="Reprocess files even if output already exists",
    )
    return p


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


def _already_done(output_dir: Path, stem: str, mode: str) -> bool:
    if mode == "merged":
        return (output_dir / f"{stem}.glb").exists()
    target = output_dir / stem
    return target.is_dir() and any(target.glob("*.glb"))


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = load_config(args.config)
    cfg = apply_overrides(
        cfg,
        {
            "hy3d_part_root": args.hy3d_part_root,
            "p3sam_ckpt_path": args.p3sam_ckpt_path,
            "input_dir": args.input_dir,
            "output_dir": args.output_dir,
            "export_mode": args.export_mode,
            "log_level": args.log_level,
        },
    )
    _configure_logging(cfg.log_level)

    input_dir = cfg.resolved_input_dir()
    output_dir = cfg.resolved_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.glb"))
    if not files:
        logger.warning("no .glb files in %s", input_dir)
        return 0

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
        )
    except UpstreamNotInstalled as exc:
        logger.error("%s", exc)
        return 2

    n_ok = n_skip = n_fail = 0
    for glb in tqdm(files, desc="segmenting", unit="mesh"):
        if not args.force and _already_done(output_dir, glb.stem, cfg.export_mode):
            logger.info("skip (already done): %s", glb.name)
            n_skip += 1
            continue
        t0 = time.time()
        try:
            parts = segmenter.segment(glb)
            export_parts(parts, output_dir, stem=glb.stem, mode=cfg.export_mode)
            logger.info("OK %s (%.1fs, %d parts)", glb.name, time.time() - t0, len(parts))
            n_ok += 1
        except Exception as exc:  # noqa: BLE001
            logger.exception("FAIL %s: %s", glb.name, exc)
            n_fail += 1

    logger.info("batch done — ok=%d skip=%d fail=%d", n_ok, n_skip, n_fail)
    return 0 if n_fail == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
