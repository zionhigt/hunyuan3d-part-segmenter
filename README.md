# hunyuan3d-part-segmenter

Maillon de **segmentation par parties** d'un mesh 3D (GLB). Wrapper d'orchestration autour du pipeline open-source [Tencent **Hunyuan3D-Part**](https://github.com/Tencent-Hunyuan/Hunyuan3D-Part) : **P3-SAM** pour la détection/segmentation, **X-Part** (optionnel) pour la régénération propre des parties. Ce projet est **indépendant** : il n'importe aucun code d'un autre maillon du pipeline. Le contrat est strictement basé sur des fichiers : un GLB monobloc en entrée, un (ou des) GLB(s) segmenté(s) en sortie — directement exploitables pour le rigging.

## Flux macro

```
[projet génération hy3d]  →  GLB monobloc
      │  (fichier, pas d'import)
      ▼
[CE PROJET]  P3-SAM (+ X-Part option)  →  GLB multi-parties nommées
      │  (fichier, pas d'import)
      ▼
[projet rigging Blender headless]  →  armature, pivots, export Godot
```

## Quickstart

```bash
conda activate hy3d-part
python src/check_env.py                     # diagnostic torch/CUDA/VRAM
python src/single.py --glb input/vehicle.glb
python src/batch.py                         # traite tout input/
python src/single.py --glb input/vehicle.glb --enable-xpart   # ajoute X-Part
```

Installation complète (Windows 11 / Shadow Power) : voir [`INSTALL.md`](./INSTALL.md).

## Paramètres `config.yaml`

| Clé                | Type    | Défaut                       | Rôle |
|--------------------|---------|------------------------------|------|
| `p3sam_model_path` | str     | `tencent/Hunyuan3D-Part`     | Repo HF (ou chemin local) des poids P3-SAM. |
| `xpart_model_path` | str     | `tencent/Hunyuan3D-Part`     | Repo HF (ou chemin local) des poids X-Part *light*. |
| `enable_xpart`     | bool    | `false`                      | Active l'étape X-Part (plus lourd, risque OOM). |
| `export_mode`      | str     | `merged`                     | `merged` = un GLB multi-parties nommées ; `split` = un GLB par partie. |
| `input_dir`        | str     | `input`                      | Dossier source des `.glb`. |
| `output_dir`       | str     | `output`                     | Dossier de sortie. |
| `device`           | str     | `cuda`                       | `cuda` ou `cpu` (cpu inutilisable en pratique). |
| `log_level`        | str     | `INFO`                       | `DEBUG`, `INFO`, `WARNING`, `ERROR`. |

Toute clé peut être surchargée en CLI (`--enable-xpart`, `--export-mode split`, etc.).

## Modes d'export

- **`merged`** : un seul fichier `<stem>.glb` contenant chaque partie comme **mesh distinct nommé** (`part_000`, `part_001`, … ou labels sémantiques quand disponibles). À l'import dans Blender, chaque partie devient un objet distinct.
- **`split`** : un sous-dossier `<output>/<stem>/` contenant un GLB par partie. Pratique pour le diff ou la régénération individuelle.

> Note nommage : P3-SAM expose des indices de parties, pas des labels sémantiques fins (« roue avant gauche »). Les noms sémantiques restent à affiner en aval (pipeline rigging).

## Matériel cible

Pensé/testé pour **Shadow PC Power** : NVIDIA **RTX A4500 (20 Go VRAM)**, 28 Go RAM, **Windows 11 natif**. P3-SAM tient confortablement dans 20 Go ; **X-Part est plus gourmand** et peut OOM sur les meshes denses — voir le dépannage dans `INSTALL.md`. La décimation des meshes se fait en amont (côté pipeline de génération), pas ici.

## Structure

```
src/
  segmenter.py   # PartSegmenter (charge P3-SAM, X-Part optionnel)
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
