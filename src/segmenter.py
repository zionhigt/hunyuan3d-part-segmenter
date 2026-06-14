"""PartSegmenter — thin wrapper around Tencent Hunyuan3D-Part (P3-SAM + optional X-Part).

This module does NOT reimplement the models. It imports the upstream package
(installed separately, see INSTALL.md) and orchestrates the segmentation of
a single GLB into a list of `trimesh.Trimesh` parts.

Upstream API surface is not stable across releases — we therefore probe a few
known entry points and surface a clear error if none matches.
"""

from __future__ import annotations

import importlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


logger = logging.getLogger(__name__)


class UpstreamNotInstalled(RuntimeError):
    """Raised when Tencent's Hunyuan3D-Part package can't be imported."""


@dataclass
class SegmentedPart:
    mesh: trimesh.Trimesh
    label: str
    index: int


class PartSegmenter:
    """Load P3-SAM (+ optional X-Part) and segment a GLB into named parts."""

    def __init__(
        self,
        p3sam_model_path: str,
        xpart_model_path: str | None = None,
        enable_xpart: bool = False,
        device: str = "cuda",
    ) -> None:
        self.p3sam_model_path = p3sam_model_path
        self.xpart_model_path = xpart_model_path
        self.enable_xpart = enable_xpart
        self.device = device

        self._p3sam: Any | None = None
        self._xpart: Any | None = None

        self._load_p3sam()
        if enable_xpart:
            self._load_xpart()

    # -------- model loading --------

    def _load_p3sam(self) -> None:
        module = _try_import_any(["P3SAM", "p3sam", "hunyuan3d_part.p3sam"])
        if module is None:
            raise UpstreamNotInstalled(
                "Could not import P3-SAM. Install Tencent Hunyuan3D-Part first "
                "(see INSTALL.md, section 4)."
            )
        ctor = _resolve_callable(
            module,
            ["P3SAM", "P3SAMPredictor", "Predictor", "build_p3sam", "load_model"],
        )
        if ctor is None:
            raise UpstreamNotInstalled(
                f"Imported {module.__name__} but found no known P3-SAM entry "
                "point. Upstream API may have changed — check the repo README."
            )
        logger.info("Loading P3-SAM from %s", self.p3sam_model_path)
        t0 = time.time()
        self._p3sam = _instantiate(ctor, self.p3sam_model_path, self.device)
        logger.info("P3-SAM loaded in %.2fs", time.time() - t0)

    def _load_xpart(self) -> None:
        module = _try_import_any(["XPart", "xpart", "hunyuan3d_part.xpart"])
        if module is None:
            raise UpstreamNotInstalled(
                "enable_xpart=True but X-Part is not importable. Install the "
                "X-Part component from Hunyuan3D-Part (see INSTALL.md)."
            )
        ctor = _resolve_callable(
            module,
            ["XPart", "XPartPredictor", "Predictor", "build_xpart", "load_model"],
        )
        if ctor is None:
            raise UpstreamNotInstalled(
                f"Imported {module.__name__} but found no known X-Part entry point."
            )
        logger.info("Loading X-Part from %s", self.xpart_model_path)
        t0 = time.time()
        self._xpart = _instantiate(ctor, self.xpart_model_path, self.device)
        logger.info("X-Part loaded in %.2fs", time.time() - t0)

    # -------- public API --------

    def segment(self, glb_path: str | Path) -> list[SegmentedPart]:
        """Segment a GLB into parts. Returns a list of SegmentedPart."""
        glb_path = Path(glb_path)
        if not glb_path.is_file():
            raise FileNotFoundError(glb_path)

        logger.info("Segmenting %s", glb_path.name)
        scene_or_mesh = trimesh.load(str(glb_path), force="mesh", process=False)
        mesh = _as_single_mesh(scene_or_mesh)
        logger.info("Input mesh: %d vertices / %d faces", len(mesh.vertices), len(mesh.faces))

        t0 = time.time()
        peak_mb = _reset_vram_peak()

        labels = self._run_p3sam(mesh)
        parts = _split_by_labels(mesh, labels)

        if self.enable_xpart and self._xpart is not None:
            parts = [self._run_xpart(p) for p in parts]

        elapsed = time.time() - t0
        peak_mb_after = _read_vram_peak(peak_mb)
        logger.info(
            "Segmentation done in %.2fs — %d parts — VRAM peak ~%s",
            elapsed,
            len(parts),
            f"{peak_mb_after:.0f} MiB" if peak_mb_after is not None else "n/a",
        )
        return [
            SegmentedPart(mesh=m, label=_default_label(i), index=i)
            for i, m in enumerate(parts)
        ]

    # -------- model dispatch --------

    def _run_p3sam(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """Invoke P3-SAM and return a per-face integer label array."""
        assert self._p3sam is not None
        candidates = ("segment", "predict", "__call__", "infer", "run")
        for name in candidates:
            fn = getattr(self._p3sam, name, None)
            if callable(fn):
                logger.debug("Calling P3-SAM via .%s()", name)
                result = fn(mesh)
                return _extract_face_labels(result, n_faces=len(mesh.faces))
        raise RuntimeError(
            f"P3-SAM object {type(self._p3sam).__name__} exposes none of "
            f"{candidates}. Upstream API may have changed."
        )

    def _run_xpart(self, part: trimesh.Trimesh) -> trimesh.Trimesh:
        assert self._xpart is not None
        candidates = ("regenerate", "decompose", "predict", "__call__", "run")
        for name in candidates:
            fn = getattr(self._xpart, name, None)
            if callable(fn):
                logger.debug("Calling X-Part via .%s()", name)
                out = fn(part)
                return _as_single_mesh(out)
        raise RuntimeError(
            f"X-Part object {type(self._xpart).__name__} exposes none of "
            f"{candidates}."
        )


# ---------------- helpers ----------------


def _try_import_any(names: list[str]) -> Any | None:
    for n in names:
        try:
            return importlib.import_module(n)
        except ImportError:
            continue
    return None


def _resolve_callable(module: Any, names: list[str]) -> Any | None:
    for n in names:
        obj = getattr(module, n, None)
        if callable(obj):
            return obj
    return None


def _instantiate(ctor: Any, model_path: str, device: str) -> Any:
    """Try a few common signatures used by upstream classes."""
    attempts = [
        lambda: ctor(model_path=model_path, device=device),
        lambda: ctor(pretrained=model_path, device=device),
        lambda: ctor(model_path, device=device),
        lambda: ctor(model_path),
        lambda: ctor(),
    ]
    last_err: Exception | None = None
    for fn in attempts:
        try:
            return fn()
        except TypeError as exc:
            last_err = exc
            continue
    raise RuntimeError(
        f"Could not instantiate {ctor!r} with known signatures. Last error: {last_err}"
    )


def _as_single_mesh(obj: Any) -> trimesh.Trimesh:
    if isinstance(obj, trimesh.Trimesh):
        return obj
    if isinstance(obj, trimesh.Scene):
        geoms = list(obj.geometry.values())
        if not geoms:
            raise ValueError("Scene contains no geometry")
        return trimesh.util.concatenate(geoms)
    if isinstance(obj, (list, tuple)) and obj and isinstance(obj[0], trimesh.Trimesh):
        return trimesh.util.concatenate(list(obj))
    raise TypeError(f"Cannot coerce {type(obj).__name__} to a single Trimesh")


def _extract_face_labels(result: Any, n_faces: int) -> np.ndarray:
    """Coerce whatever the upstream model returned into a (n_faces,) int array."""
    if isinstance(result, np.ndarray):
        arr = result
    elif isinstance(result, dict):
        for key in ("face_labels", "labels", "segmentation", "seg"):
            if key in result:
                arr = np.asarray(result[key])
                break
        else:
            raise ValueError(f"Unrecognized P3-SAM output dict keys: {list(result)}")
    elif hasattr(result, "cpu"):
        arr = result.cpu().numpy()
    else:
        arr = np.asarray(result)

    arr = arr.astype(np.int64).reshape(-1)
    if arr.size != n_faces:
        raise ValueError(
            f"P3-SAM returned {arr.size} labels but mesh has {n_faces} faces"
        )
    return arr


def _split_by_labels(mesh: trimesh.Trimesh, labels: np.ndarray) -> list[trimesh.Trimesh]:
    parts: list[trimesh.Trimesh] = []
    for lbl in np.unique(labels):
        face_mask = labels == lbl
        sub = mesh.submesh([np.where(face_mask)[0]], append=True)
        if isinstance(sub, list):
            sub = trimesh.util.concatenate(sub) if sub else None
        if sub is None or len(sub.faces) == 0:
            continue
        parts.append(sub)
    if not parts:
        raise RuntimeError("Segmentation produced 0 parts")
    return parts


def _default_label(i: int) -> str:
    return f"part_{i:03d}"


def _reset_vram_peak() -> float | None:
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            return 0.0
    except ImportError:
        pass
    return None


def _read_vram_peak(_sentinel: float | None) -> float | None:
    if _sentinel is None:
        return None
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024**2
    except ImportError:
        pass
    return None
