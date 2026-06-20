"""Apply Windows-compat patches to a Tencent Hunyuan3D-Part clone.

Idempotent: re-running is a no-op once patches are applied.

Usage (from the segmenter repo root, hy3d-part env active):
    python scripts/patch_hy3d_part.py
    python scripts/patch_hy3d_part.py C:/path/to/Hunyuan3D-Part

Default hy3d_part_root is read from config.yaml.

Patches:
1. XPart/partgen/models/sonata/model.py: force enable_flash=False in load().
   PTv3's SerializedAttention has a complete non-flash fallback path
   (regular matmul+softmax+attn@v), so disabling flash is functionally safe
   and avoids the flash-attn wheel pain on Windows.
2. P3-SAM/model.py: replace the hard-coded Linux path '/root/sonata' with
   None, falling back to HuggingFace's default cache (~/.cache/sonata/ckpt).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


PATCH_SONATA_OLD = """    # 关闭flash attention
    # ckpt["config"]['enable_flash'] = False"""

PATCH_SONATA_NEW = """    # Windows-compat patch (applied by hunyuan3d-part-segmenter):
    # force flash attention OFF — no flash_attn wheel for torch+cu124 on
    # Windows. PTv3's SerializedAttention has a non-flash fallback path
    # (regular matmul+softmax+attn@v) that produces equivalent output.
    ckpt["config"]['enable_flash'] = False"""

PATCH_P3SAM_OLD = (
    'self.sonata = sonata.load("sonata", repo_id="facebook/sonata", '
    "download_root='/root/sonata')"
)

PATCH_P3SAM_NEW = (
    'self.sonata = sonata.load("sonata", repo_id="facebook/sonata", '
    "download_root=None)"
)

# Shadow-priority patch: P3-SAM/model.py originally appends XPart/partgen to
# sys.path. If the env was cloned from hy3d (Hunyuan3D-2.1), the clone carries
# a .pth that injects hy3dpaint/ early in sys.path. hy3dpaint has utils/ and
# models/ as regular packages, which win against XPart/partgen's namespace
# packages. We switch the append to an insert(0, ...) so XPart/partgen takes
# priority, AND we drop empty __init__.py files in XPart/partgen/{utils,models}
# to make them regular packages too (priority over any sibling namespace).
PATCH_SYSPATH_OLD = (
    "sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'XPart/partgen'))"
)

PATCH_SYSPATH_NEW = (
    "sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'XPart/partgen'))"
)

# Tencent left two hardcoded debug overrides at the top of mesh_sam() that
# pin point_num=100000 / prompt_num=400 regardless of what the CLI flags
# requested. The forward pass allocates a [point_num, prompt_bs, 1027] fp32
# tensor (~25 GB at default values), causing OOM on a 20 GB GPU when
# flash_attn is disabled. We comment them out so --point_num / --prompt_num
# actually take effect.
PATCH_AUTOMASK_OLD = """    point_num = 100000
    prompt_num = 400"""

PATCH_AUTOMASK_NEW = """    # Patched by hunyuan3d-part-segmenter: was forcing point_num=100000 /
    # prompt_num=400 regardless of CLI flags. Let function parameters flow.
    # point_num = 100000
    # prompt_num = 400"""


def patch_file(path: Path, old: str, new: str, label: str) -> str:
    if not path.is_file():
        return f"[SKIP] {label}: file not found ({path})"
    text = path.read_text(encoding="utf-8")
    if new in text:
        return f"[OK]   {label}: already patched"
    if old not in text:
        return f"[FAIL] {label}: original snippet not found in {path}"
    path.write_text(text.replace(old, new), encoding="utf-8")
    return f"[DONE] {label}: patched {path}"


def ensure_init_py(path: Path, label: str) -> str:
    if not path.parent.is_dir():
        return f"[SKIP] {label}: parent dir not found ({path.parent})"
    if path.is_file():
        return f"[OK]   {label}: already present"
    path.write_text("", encoding="utf-8")
    return f"[DONE] {label}: created {path}"


def resolve_root(arg_root: str | None) -> Path:
    if arg_root:
        return Path(arg_root)
    cfg = Path(__file__).resolve().parent.parent / "config.yaml"
    if not cfg.is_file():
        sys.exit(f"config.yaml not found at {cfg}; pass hy3d_part_root as arg.")
    with cfg.open(encoding="utf-8") as f:
        return Path(yaml.safe_load(f)["hy3d_part_root"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hy3d_part_root", nargs="?", default=None)
    args = parser.parse_args()

    root = resolve_root(args.hy3d_part_root).resolve()
    if not root.is_dir():
        sys.exit(f"hy3d_part_root not found: {root}")

    print(f"hy3d_part_root = {root}\n")
    results = [
        patch_file(
            root / "XPart" / "partgen" / "models" / "sonata" / "model.py",
            PATCH_SONATA_OLD,
            PATCH_SONATA_NEW,
            "sonata/model.py: disable flash_attn",
        ),
        patch_file(
            root / "P3-SAM" / "model.py",
            PATCH_P3SAM_OLD,
            PATCH_P3SAM_NEW,
            "P3-SAM/model.py: fix /root/sonata path",
        ),
        patch_file(
            root / "P3-SAM" / "model.py",
            PATCH_SYSPATH_OLD,
            PATCH_SYSPATH_NEW,
            "P3-SAM/model.py: sys.path.append -> insert(0)",
        ),
        patch_file(
            root / "P3-SAM" / "demo" / "auto_mask.py",
            PATCH_AUTOMASK_OLD,
            PATCH_AUTOMASK_NEW,
            "auto_mask.py: unhardcode point_num/prompt_num",
        ),
        ensure_init_py(
            root / "XPart" / "partgen" / "utils" / "__init__.py",
            "XPart/partgen/utils/__init__.py",
        ),
        ensure_init_py(
            root / "XPart" / "partgen" / "models" / "__init__.py",
            "XPart/partgen/models/__init__.py",
        ),
    ]
    print("\n".join(results))
    ok = all(("[DONE]" in r) or ("[OK]" in r) for r in results)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
