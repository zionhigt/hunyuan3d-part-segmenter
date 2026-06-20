"""PartSegmenter — orchestrates Tencent Hunyuan3D-Part / P3-SAM by shelling out
to the upstream `demo/auto_mask.py` script.

Why subprocess and not in-process import?
The upstream repo is a collection of demo scripts (no setup.py, no package),
and the model file lives at `P3-SAM/model.py` with a `demo/` folder importing
it via relative path tricks. Wrapping it in a subprocess is the only stable
contract Tencent actually documents.

`--output_path` is interpreted as a DIRECTORY by auto_mask.py. The final
segmented outputs land at:
    <output_path>/auto_mask_mesh_final.glb            (segmented mesh)
    <output_path>/auto_mask_mesh_final_face_ids.npy   (one int per face)
    <output_path>/auto_mask_mesh_final_aabb.npy       (per-part AABB)
plus a bunch of intermediate `_org` / `_filtered_*` / `point_pca` files we
ignore.

X-Part is NOT supported here: the upstream README marks its public weights as
TODO. When/if Tencent ships them, add an X-Part stage in this file.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh


logger = logging.getLogger(__name__)


class UpstreamNotInstalled(RuntimeError):
    """Raised when Tencent's Hunyuan3D-Part repo or P3-SAM weights are missing."""


@dataclass
class SegmentedPart:
    mesh: trimesh.Trimesh
    label: str
    index: int


class PartSegmenter:
    """Segment a GLB by invoking P3-SAM's demo/auto_mask.py."""

    def __init__(
        self,
        hy3d_part_root: str | Path,
        p3sam_ckpt_path: str | Path,
        python_executable: str = "",
        point_num: int = 50000,
        threshold: float = 0.95,
        seed: int = 42,
        clean_mesh: int = 1,
        post_process: int = 0,
        prompt_bs: int = 1,
    ) -> None:
        self.hy3d_part_root = Path(hy3d_part_root)
        self.demo_script = self.hy3d_part_root / "P3-SAM" / "demo" / "auto_mask.py"
        self.ckpt_path = Path(p3sam_ckpt_path)
        self.python = python_executable or sys.executable

        if not self.demo_script.is_file():
            raise UpstreamNotInstalled(
                f"P3-SAM script not found at {self.demo_script}. "
                "Clone https://github.com/Tencent-Hunyuan/Hunyuan3D-Part and set "
                "hy3d_part_root (see INSTALL.md)."
            )
        if not self.ckpt_path.is_file():
            raise UpstreamNotInstalled(
                f"P3-SAM checkpoint not found at {self.ckpt_path}. "
                "Download p3sam.safetensors from "
                "https://huggingface.co/tencent/Hunyuan3D-Part (see INSTALL.md §4)."
            )

        self.point_num = point_num
        self.threshold = threshold
        self.seed = seed
        self.clean_mesh = clean_mesh
        self.post_process = post_process
        self.prompt_bs = prompt_bs

    def segment(self, glb_path: str | Path) -> list[SegmentedPart]:
        glb_path = Path(glb_path).resolve()
        if not glb_path.is_file():
            raise FileNotFoundError(glb_path)

        logger.info("Segmenting %s", glb_path.name)
        t0 = time.time()

        with tempfile.TemporaryDirectory(prefix="p3sam_") as td:
            out_dir = Path(td)
            cmd = [
                self.python,
                str(self.demo_script),
                "--ckpt_path", str(self.ckpt_path),
                "--mesh_path", str(glb_path),
                "--output_path", str(out_dir),
                "--point_num", str(self.point_num),
                "--threshold", str(self.threshold),
                "--seed", str(self.seed),
                "--clean_mesh", str(self.clean_mesh),
                "--post_process", str(self.post_process),
                "--save_mid_res", "0",
                "--show_info", "0",
                "--prompt_bs", str(self.prompt_bs),
            ]
            logger.debug("running: %s", " ".join(cmd))
            try:
                subprocess.run(
                    cmd,
                    cwd=str(self.demo_script.parent),
                    check=True,
                )
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(
                    f"P3-SAM auto_mask.py exited with code {exc.returncode}"
                ) from exc

            out_glb = out_dir / "auto_mask_mesh_final.glb"
            face_ids_path = out_dir / "auto_mask_mesh_final_face_ids.npy"
            if not out_glb.is_file() or not face_ids_path.is_file():
                raise RuntimeError(
                    "P3-SAM did not produce expected outputs "
                    f"({out_glb.name}, {face_ids_path.name}) in {td}"
                )

            segmented_mesh = _as_single_mesh(
                trimesh.load(str(out_glb), force="mesh", process=False)
            )
            face_ids = np.load(face_ids_path).astype(np.int64).reshape(-1)

        parts = _split_by_labels(segmented_mesh, face_ids)
        logger.info(
            "Segmentation done in %.2fs — %d parts (mesh %d faces)",
            time.time() - t0,
            len(parts),
            len(segmented_mesh.faces),
        )
        return [
            SegmentedPart(mesh=m, label=f"part_{i:03d}", index=i)
            for i, m in enumerate(parts)
        ]


def _as_single_mesh(obj: object) -> trimesh.Trimesh:
    if isinstance(obj, trimesh.Trimesh):
        return obj
    if isinstance(obj, trimesh.Scene):
        geoms = list(obj.geometry.values())
        if not geoms:
            raise ValueError("Scene contains no geometry")
        return trimesh.util.concatenate(geoms)
    raise TypeError(f"Cannot coerce {type(obj).__name__} to a single Trimesh")


def _split_by_labels(
    mesh: trimesh.Trimesh, labels: np.ndarray
) -> list[trimesh.Trimesh]:
    if labels.size != len(mesh.faces):
        raise ValueError(
            f"face_ids length {labels.size} != mesh faces {len(mesh.faces)}"
        )
    parts: list[trimesh.Trimesh] = []
    for lbl in np.unique(labels):
        face_idx = np.where(labels == lbl)[0]
        sub = mesh.submesh([face_idx], append=True)
        if isinstance(sub, list):
            sub = trimesh.util.concatenate(sub) if sub else None
        if sub is None or len(sub.faces) == 0:
            continue
        parts.append(sub)
    if not parts:
        raise RuntimeError("Segmentation produced 0 parts")
    return parts
