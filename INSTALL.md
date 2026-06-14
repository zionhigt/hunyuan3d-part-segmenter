# INSTALL — Windows 11 natif / Shadow PC Power (RTX A4500, 20 Go VRAM, 28 Go RAM)

Procédure complète et reproductible sur une machine vierge. Le projet est un **wrapper subprocess** au-dessus du script officiel `P3-SAM/demo/auto_mask.py` du repo Tencent. Tout tourne dans un environnement conda **dédié** `hy3d-part`, **distinct** de l'environnement de génération (`hy3d`).

> **X-Part n'est PAS supporté ici** : les poids publics sont marqués `TODO` dans le README X-Part upstream au moment de l'écriture. Ce projet utilise uniquement **P3-SAM**, ce qui est suffisant pour produire des parties exploitables pour le rigging.

---

## 1. Prérequis système

- **Miniconda** : <https://www.anaconda.com/download/success>
- **Git pour Windows** : <https://git-scm.com/download/win>
- **Visual Studio Build Tools 2022** — workload « *Desktop development with C++* » **complet**. Nécessaire pour compiler le kernel CUDA `chamfer3D` (étape obligatoire P3-SAM). Cocher tout le workload, puis **redémarrer le terminal** pour que `cl.exe` / `link.exe` soient sur le `PATH`. Téléchargement : <https://visualstudio.microsoft.com/visual-cpp-build-tools/>
- **Pilote NVIDIA** : déjà présent sur Shadow. Vérifier :
  ```cmd
  nvidia-smi
  ```
  Le chiffre « **CUDA Version** » en haut à droite = **version CUDA MAX supportée par le pilote**. La config testée upstream est **CUDA 12.1 / PyTorch 2.4.0+cu121** ; le pilote Shadow (typiquement CUDA 12.8) couvre largement.

---

## 2. Environnement conda DÉDIÉ

```cmd
conda create -n hy3d-part python=3.10 -y
conda activate hy3d-part
```

> **Insistance** : `hy3d-part` est SÉPARÉ de l'env de génération `hy3d`. Avant tout `pip install`, vérifier que le prompt affiche bien `(hy3d-part)`.

---

## 3. PyTorch (config testée upstream)

```cmd
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
```

Vérification :

```cmd
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Attendu : `2.4.0+cu121 12.1 True NVIDIA RTX A4500`.

> Si tu veux dévier (cu124, etc.), vérifie d'abord la compat avec les kernels CUDA de `chamfer3D` ; en cas de doute, reste sur la combo officielle ci-dessus.

---

## 4. Installation de Hunyuan3D-Part (P3-SAM)

Liens officiels :
- **Dépôt GitHub** : <https://github.com/Tencent-Hunyuan/Hunyuan3D-Part>
- **Composant P3-SAM** : <https://github.com/Tencent-Hunyuan/Hunyuan3D-Part/tree/main/P3-SAM>
- **Poids HuggingFace** : <https://huggingface.co/tencent/Hunyuan3D-Part>
- **Org GitHub** : <https://github.com/Tencent-Hunyuan>

Clés de recherche (si les URLs changent) :
- GitHub : `Tencent-Hunyuan Hunyuan3D-Part`
- HuggingFace : `tencent Hunyuan3D-Part`
- Papier : `P3-SAM Native 3D Part Segmentation` (arXiv 2509.06784)

### 4.1 Clone

```cmd
cd C:\Users\Shadow
git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-Part.git
```

Tu obtiens `C:\Users\Shadow\Hunyuan3D-Part\` avec à l'intérieur `P3-SAM\`, `XPart\`, etc.

### 4.2 Dépendance externe : Sonata (Facebook Research)

P3-SAM utilise le framework **Sonata**. À installer en suivant son propre README :

```cmd
git clone https://github.com/facebookresearch/sonata.git
cd sonata
REM Suivre les instructions du README Sonata pour l'install (pip install -e . dans la plupart des cas).
```

### 4.3 Dépendances pip de P3-SAM

> Il n'y a **PAS** de `requirements.txt` à la racine de Hunyuan3D-Part. Les paquets sont listés dans le README de `P3-SAM/`.

```cmd
pip install viser fpsample trimesh numba gradio
```

### 4.4 Build du kernel CUDA `chamfer3D` (obligatoire)

```cmd
cd C:\Users\Shadow\Hunyuan3D-Part\P3-SAM\utils\chamfer3D
python setup.py install
```

> Échec ici = Build Tools VS 2022 « Desktop C++ » incomplets, **ou** mismatch CUDA toolkit / wheel torch. Vérifier `cl.exe` accessible (`where cl`) et que le CUDA toolkit (`nvcc --version`) correspond bien à la version cu121 de torch.

### 4.5 Poids P3-SAM

Téléchargement HuggingFace dans `Hunyuan3D-Part\P3-SAM\weights\` :

```cmd
huggingface-cli download tencent/Hunyuan3D-Part p3sam.safetensors --local-dir C:\Users\Shadow\Hunyuan3D-Part\P3-SAM\weights
```

(Connecte-toi avec `huggingface-cli login` si l'accès aux poids requiert l'acceptation des conditions.)

Optionnel : rediriger le cache HF vers un disque avec plus d'espace (extension stockage Shadow jusqu'à 5 To) :

```cmd
setx HF_HOME "D:\hf_cache"
```

(Fermer/rouvrir le terminal pour appliquer.)

### 4.6 Test rapide de P3-SAM (en dehors de ce projet)

```cmd
cd C:\Users\Shadow\Hunyuan3D-Part\P3-SAM\demo
python auto_mask.py --ckpt_path ..\weights\p3sam.safetensors --mesh_path assets\1.glb --output_path results\1
```

Tu dois voir apparaître `results\1.glb` (mesh colorisé par partie), `results\1.ply`, `results\1_aabb.npy`, `results\1_face_ids.npy`. Si oui, P3-SAM est opérationnel ; on peut passer à ce projet.

---

## 5. Installation de ce projet

```cmd
cd C:\Users\Shadow
git clone https://github.com/zionhigt/hunyuan3d-part-segmenter.git
cd hunyuan3d-part-segmenter
pip install -r requirements.txt
```

> Le `requirements.txt` de **ce** projet n'installe que des libs Python génériques (`trimesh`, `pyyaml`, `tqdm`, …). Il **n'installe pas** Hunyuan3D-Part — ça, c'est l'étape 4.

Éditer `config.yaml` pour pointer sur ton clone Tencent et tes poids :

```yaml
hy3d_part_root: "C:/Users/Shadow/Hunyuan3D-Part"
p3sam_ckpt_path: "C:/Users/Shadow/Hunyuan3D-Part/P3-SAM/weights/p3sam.safetensors"
```

> Pas besoin de `PYTHONPATH` ni de `pip install -e` — ce projet **shell out** sur `auto_mask.py`, il a juste besoin du chemin vers le clone et vers le checkpoint.

---

## 6. Vérification

```cmd
python src\check_env.py
```

Le diagnostic affiche : CUDA disponible, **NVIDIA RTX A4500**, ~20 Go VRAM, versions torch / CUDA.

---

## 7. Premier test

```cmd
REM Déposer un GLB monobloc dans input\, puis :
python src\single.py --glb input\vehicle.glb

REM Batch sur tout input\ :
python src\batch.py

REM Mode split (un GLB par partie au lieu d'un GLB multi-meshes) :
python src\single.py --glb input\vehicle.glb --export-mode split
```

Lanceur Windows tout-en-un :

```cmd
scripts\run_batch.bat
```

---

## 8. Dépannage

| Symptôme | Cause probable | Action |
|---|---|---|
| `torch.cuda.is_available() == False` | mauvais wheel CUDA / pilote non chargé | réinstaller la combo cu121 §3 ; vérifier `nvidia-smi`. |
| `chamfer3D` ne compile pas (`cl.exe`/`nvcc` errors) | Build Tools « Desktop C++ » incomplets, terminal non relancé, ou mismatch CUDA toolkit/torch | réinstaller le workload VS complet, fermer/rouvrir le terminal, vérifier `where cl` et `nvcc --version`. |
| `P3-SAM script not found at .../P3-SAM/demo/auto_mask.py` | mauvais `hy3d_part_root` | vérifier le chemin dans `config.yaml`. |
| `P3-SAM checkpoint not found` | `p3sam_ckpt_path` faux ou poids non téléchargés | refaire §4.5. |
| `ModuleNotFoundError: sonata` (ou équivalent) en lançant `auto_mask.py` | Sonata pas installé | refaire §4.2. |
| `OutOfMemoryError` | mesh trop dense | réduire la densité en amont (décimation côté pipeline de génération) ou baisser `p3sam_point_num` dans `config.yaml`. |
| Cache HF sature `C:` | poids gros | rediriger `HF_HOME` (§4.5). |
| Lenteur extrême | fallback CPU | vérifier `nvidia-smi` **pendant** l'exécution ; GPU à 0 % = fallback CPU silencieux. |
| L'env actif n'est pas `hy3d-part` | activation oubliée | `conda activate hy3d-part`. |
| Veux X-Part | poids upstream `TODO` | non supporté ; attendre que Tencent publie les poids X-Part puis ouvrir une issue ici. |
