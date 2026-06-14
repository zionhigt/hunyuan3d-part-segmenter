# hunyuan3d-part-segmenter

Maillon de **segmentation par parties** d'un mesh 3D (GLB). Wrapper **subprocess** au-dessus du script officiel [P3-SAM `demo/auto_mask.py`](https://github.com/Tencent-Hunyuan/Hunyuan3D-Part/tree/main/P3-SAM) de Tencent. Ce projet est **indépendant** : il n'importe aucun code d'un autre maillon du pipeline et ne ré-implémente pas le modèle. Le contrat est strictement fichier : un GLB monobloc en entrée, un (ou des) GLB(s) segmenté(s) en sortie — directement exploitables pour le rigging.

> **X-Part n'est pas supporté** : les poids publics sont marqués `TODO` dans le README X-Part upstream. P3-SAM seul suffit pour produire des parties exploitables.

## Flux macro

```
[projet génération hy3d]  →  GLB monobloc
      │  (fichier, pas d'import)
      ▼
[CE PROJET]  P3-SAM (via subprocess sur auto_mask.py)  →  GLB multi-parties nommées
      │  (fichier, pas d'import)
      ▼
[projet rigging Blender headless]  →  armature, pivots, export Godot
```

## Quickstart

```bash
conda activate hy3d-part
python src/check_env.py                     # diagnostic torch/CUDA/VRAM
# 1. Configure hy3d_part_root et p3sam_ckpt_path dans config.yaml
python src/single.py --glb input/vehicle.glb
python src/batch.py                         # traite tout input/
python src/single.py --glb input/vehicle.glb --export-mode split
```

Installation complète (Windows 11 / Shadow Power) : voir [`INSTALL.md`](./INSTALL.md). **Important** : `pip install -r requirements.txt` de ce projet **n'installe pas** Hunyuan3D-Part — il y a une procédure à part (Sonata, build du kernel CUDA `chamfer3D`, poids P3-SAM via HuggingFace), tout est dans `INSTALL.md` §4.

## Paramètres `config.yaml`

| Clé                 | Type    | Défaut                                                         | Rôle |
|---------------------|---------|----------------------------------------------------------------|------|
| `hy3d_part_root`    | str     | `C:/Users/Shadow/Hunyuan3D-Part`                               | Racine du clone local du repo Tencent. |
| `p3sam_ckpt_path`   | str     | `.../P3-SAM/weights/p3sam.safetensors`                         | Chemin vers le checkpoint P3-SAM. |
| `python_executable` | str     | `""` (= interpréteur courant)                                  | Forcer un autre `python` pour lancer `auto_mask.py`. |
| `export_mode`       | str     | `merged`                                                       | `merged` = un GLB multi-parties nommées ; `split` = un GLB par partie. |
| `input_dir`         | str     | `input`                                                        | Dossier source des `.glb`. |
| `output_dir`        | str     | `output`                                                       | Dossier de sortie. |
| `p3sam_point_num`   | int     | `100000`                                                       | Sampling P3-SAM. Baisser pour réduire VRAM. |
| `p3sam_threshold`   | float   | `0.95`                                                         | Seuil de fusion des parties. |
| `p3sam_seed`        | int     | `42`                                                           | Graine. |
| `p3sam_clean_mesh`  | int     | `1`                                                            | Laisser P3-SAM nettoyer le mesh avant segmentation. |
| `p3sam_post_process`| int     | `0`                                                            | Post-traitement P3-SAM. |
| `log_level`         | str     | `INFO`                                                         | `DEBUG`, `INFO`, `WARNING`, `ERROR`. |

Toute clé peut être surchargée en CLI (`--export-mode split`, `--hy3d-part-root ...`, etc.).

## Modes d'export

- **`merged`** : un seul fichier `<stem>.glb` contenant chaque partie comme **mesh distinct nommé** (`part_000`, `part_001`, …). À l'import dans Blender, chaque partie devient un objet distinct.
- **`split`** : un sous-dossier `<output>/<stem>/` contenant un GLB par partie.

> Note nommage : P3-SAM expose des indices de parties, pas des labels sémantiques fins (« roue avant gauche »). Les noms sémantiques restent à affiner en aval (pipeline rigging).

## Comment ça marche

Pour chaque GLB en entrée, ce projet :

1. Lance `<hy3d_part_root>/P3-SAM/demo/auto_mask.py` via `subprocess.run(...)` avec les flags lus dans `config.yaml`.
2. Lit le mesh segmenté (`<stem>.glb`) et les `face_ids` (`<stem>_face_ids.npy`) que le script écrit dans un dossier temporaire.
3. Découpe le mesh en sous-meshes par `face_id` et exporte selon `export_mode`.

Aucun import in-process des modèles Tencent — le repo upstream n'expose pas de package Python propre, le contrat documenté est la CLI `auto_mask.py`.

## Matériel cible

Pensé/testé pour **Shadow PC Power** : NVIDIA **RTX A4500 (20 Go VRAM)**, 28 Go RAM, **Windows 11 natif**. P3-SAM tient confortablement dans 20 Go ; pour réduire la VRAM, baisser `p3sam_point_num`. La décimation des meshes se fait en amont (côté pipeline de génération), pas ici.

## Structure

```
src/
  segmenter.py   # PartSegmenter : subprocess vers auto_mask.py + split par face_id
  single.py      # CLI : 1 GLB
  batch.py       # CLI : tout input/, reprise/skip, tqdm
  export.py      # export merged / split
  config.py      # config.yaml + surcharge CLI
  check_env.py   # diagnostic CUDA/VRAM
scripts/
  run_batch.bat  # lanceur Windows (active hy3d-part puis lance batch)
config.yaml
input/  output/
```

## Licence

MIT — voir [`LICENSE`](./LICENSE). Les poids Hunyuan3D-Part sont distribués sous leur propre licence par Tencent, consulter le repo upstream.
