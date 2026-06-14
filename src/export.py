"""Export segmented parts to GLB.

Two modes:
- merged: a single GLB containing each part as a named submesh.
- split:  one GLB per part inside `<output>/<stem>/`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import trimesh

from src.segmenter import SegmentedPart


logger = logging.getLogger(__name__)


def export_parts(
    parts: list[SegmentedPart],
    output_dir: Path,
    stem: str,
    mode: str = "merged",
) -> list[Path]:
    """Write parts to disk. Returns the list of files written."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if mode == "merged":
        return [_export_merged(parts, output_dir / f"{stem}.glb")]
    if mode == "split":
        return _export_split(parts, output_dir / stem)
    raise ValueError(f"unknown export mode: {mode!r}")


def _export_merged(parts: list[SegmentedPart], out: Path) -> Path:
    scene = trimesh.Scene()
    for p in parts:
        scene.add_geometry(p.mesh, node_name=p.label, geom_name=p.label)
    scene.export(out.as_posix())
    logger.info("Wrote %s (%d parts)", out.name, len(parts))
    return out


def _export_split(parts: list[SegmentedPart], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for p in parts:
        path = out_dir / f"{p.label}.glb"
        p.mesh.export(path.as_posix())
        written.append(path)
    logger.info("Wrote %d files to %s/", len(written), out_dir.name)
    return written
