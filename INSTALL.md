# INSTALL — Windows 11 natif / Shadow PC Power (RTX A4500, 20 Go VRAM, 28 Go RAM)

Procédure complète et reproductible sur une machine vierge. Le projet est un **wrapper subprocess** au-dessus du script officiel `P3-SAM/demo/auto_mask.py` du repo Tencent. Tout tourne dans un environnement conda **dédié** `hy3d-part`, **distinct** de l'environnement de génération (`hy3d`).

> **X-Part n'est PAS supporté ici** : les poids publics sont marqués `TODO` dans le README X-Part upstream au moment de l'écriture. Ce projet utilise uniquement **P3-SAM**, ce qui est suffisant pour produire des parties exploitables pour le rigging.

## Comment utiliser ce guide

Chaque section se termine par un bloc **✅ Vérification** : exécute la commande et **garde la sortie**. Si elle ne correspond pas à l'attendu, **arrête-toi** et colle la sortie ici — on patchera ce doc plutôt que de continuer avec une base cassée.

Toutes les commandes sont à exécuter dans **Anaconda Prompt** (Démarrer → "Anaconda Prompt"), **pas** PowerShell ni le terminal VSCode tant que conda n'est pas configuré pour eux. Les commandes commencent par `(env)` dans les blocs où un env doit être actif.

---

## 1. Prérequis système

À installer **dans cet ordre** :

1. **Pilote NVIDIA** — déjà présent sur Shadow, rien à faire.
2. **Miniconda** (Python 3.10+) — <https://www.anaconda.com/download/success>. Cocher *"Add Miniconda3 to my PATH environment variable"* à l'installation **simplifie tout** sur Shadow (sinon il faut passer par "Anaconda Prompt" exclusivement).
3. **Git pour Windows** — <https://git-scm.com/download/win>. Garder les options par défaut.
4. **Visual Studio Build Tools 2022** — <https://visualstudio.microsoft.com/visual-cpp-build-tools/>. Dans l'installeur, cocher le workload **"Desktop development with C++"** *complet* (laisser tous les composants par défaut sélectionnés, en particulier *MSVC v143*, *Windows 11 SDK*, *C++ CMake tools*).
5. **CUDA Toolkit 12.1** — <https://developer.nvidia.com/cuda-12-1-0-download-archive>. C'est **`nvcc` qui compile `chamfer3D`** ; le driver seul ne suffit **pas**. Choisir Windows / x86_64 / 11 / exe (local). À l'install, accepter les composants par défaut (`nvcc`, runtime, headers).

> **Important** : après l'install de VS Build Tools **et** du CUDA Toolkit, **fermer et rouvrir** ton Anaconda Prompt pour que `cl.exe`, `link.exe`, `nvcc.exe` soient visibles sur le `PATH`.

### ✅ Vérification §1

```cmd
nvidia-smi
where cl
where nvcc
nvcc --version
```

Attendu :
- `nvidia-smi` : tableau avec **RTX A4500**, ligne d'en-tête `CUDA Version: 12.x` (12.8 typique sur Shadow — c'est la **version max supportée par le driver**, pas la version installée du toolkit).
- `where cl` : un chemin du type `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.4x.x\bin\Hostx64\x64\cl.exe`.
- `where nvcc` : `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin\nvcc.exe`.
- `nvcc --version` : ligne `Cuda compilation tools, release 12.1, ...`.

> Si `where cl` ne trouve rien : tu as ouvert un terminal **avant** que VS soit installé, ou tu utilises PowerShell sans hook. Lance **"x64 Native Tools Command Prompt for VS 2022"** depuis le menu Démarrer — il pré-positionne `cl.exe` sur le PATH ; puis ré-active conda dedans.

---

## 2. Environnement conda DÉDIÉ

```cmd
conda create -n hy3d-part python=3.10 -y
conda activate hy3d-part
```

> **Insistance** : `hy3d-part` est SÉPARÉ de l'env de génération `hy3d`. Avant tout `pip install`, vérifier que le prompt affiche bien `(hy3d-part)`.

### ✅ Vérification §2

```cmd
python --version
where python
```

Attendu : `Python 3.10.x` et un chemin contenant `envs\hy3d-part\python.exe`.

---

## 3. PyTorch (combo testée upstream)

```cmd
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
```

### ✅ Vérification §3

```cmd
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Attendu : `2.4.0+cu121 12.1 True NVIDIA RTX A4500`.

> Si `torch.cuda.is_available()` ➜ `False` : driver KO ou wheel CPU installé par erreur. Refaire `pip install` **avec** `--index-url`.
> Ne dévie pas vers cu124/cu128 sans raison — la combo `chamfer3D` upstream est calibrée pour cu121.

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

> On installe Hunyuan3D-Part dans `C:\Users\Shadow\` ; si tu redirires ailleurs (extension stockage Shadow), adapte ensuite `config.yaml`.

### 4.1 Clone du repo Tencent

```cmd
cd C:\Users\Shadow
git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-Part.git
```

Tu obtiens `C:\Users\Shadow\Hunyuan3D-Part\` avec à l'intérieur `P3-SAM\`, `XPart\`, etc.

### ✅ Vérification §4.1

```cmd
dir C:\Users\Shadow\Hunyuan3D-Part\P3-SAM\demo\auto_mask.py
```

Attendu : le fichier est listé (pas `File Not Found`).

### 4.2 Dépendances pip de P3-SAM (+ HuggingFace)

> Il n'y a **PAS** de `requirements.txt` à la racine de Hunyuan3D-Part. Les paquets sont listés dans le README de `P3-SAM/`. On ajoute `huggingface_hub` ici pour pouvoir télécharger les poids à l'étape 4.5.

```cmd
pip install viser fpsample trimesh numba gradio huggingface_hub
```

### 4.3 Dépendance externe : Sonata (Facebook Research)

P3-SAM importe `sonata`. C'est un repo **sans wheel PyPI**, à installer en mode éditable. Le README Sonata bouge ; la séquence robuste sur Shadow :

```cmd
cd C:\Users\Shadow
git clone https://github.com/facebookresearch/sonata.git
cd sonata
pip install -e .
```

> **Si `pip install -e .` casse** sur une dépendance C++ (typiquement `flash-attn`, `pointops`, `torch-scatter`) : note l'erreur exacte et arrête-toi. Ces deps sont sensibles à la combo CUDA/torch et on patchera ce doc en fonction du message. Ne PAS forcer avec `--no-deps` à l'aveugle, ça reportera l'échec à l'import runtime.

### ✅ Vérification §4.3

```cmd
python -c "import sonata; print(sonata.__file__)"
```

Attendu : un chemin vers `C:\Users\Shadow\sonata\sonata\__init__.py` (ou équivalent).

### 4.4 Build du kernel CUDA `chamfer3D` (obligatoire)

```cmd
cd C:\Users\Shadow\Hunyuan3D-Part\P3-SAM\utils\chamfer3D
python setup.py install
```

La compilation prend ~1-3 min ; il y aura beaucoup de warnings C++/nvcc, c'est normal. Ce qui compte c'est la dernière ligne (`Finished processing dependencies for chamfer-3D-...`).

### ✅ Vérification §4.4

```cmd
python -c "import chamfer_3D; print('chamfer3D OK')"
```

Attendu : `chamfer3D OK`. (Le nom exact du module peut varier — si erreur, colle-la, on ajustera.)

> Causes d'échec classiques :
> - `cl.exe` introuvable ➜ §1 incomplet, terminal pas relancé.
> - `unsupported Microsoft Visual Studio version` ➜ il faut un compilo MSVC compatible avec CUDA 12.1 (MSVC 14.4x livré avec VS Build Tools 2022 marche).
> - `nvcc fatal : Cannot find compiler 'cl.exe' in PATH` ➜ lancer la commande depuis **"x64 Native Tools Command Prompt for VS 2022"** puis `conda activate hy3d-part` dedans.
> - mismatch CUDA ➜ `nvcc --version` doit dire 12.1 ; `torch.version.cuda` doit dire `12.1`.

### 4.5 Poids P3-SAM

Optionnel mais recommandé sur Shadow (cache HF peut dépasser plusieurs Go) — rediriger le cache HF vers le disque étendu :

```cmd
setx HF_HOME "D:\hf_cache"
```

Puis **fermer et rouvrir** l'Anaconda Prompt et `conda activate hy3d-part`.

Téléchargement HuggingFace dans `Hunyuan3D-Part\P3-SAM\weights\` :

```cmd
huggingface-cli download tencent/Hunyuan3D-Part p3sam.safetensors --local-dir C:\Users\Shadow\Hunyuan3D-Part\P3-SAM\weights
```

> Si HF demande un token (acceptation des conditions sur la page modèle) : `huggingface-cli login` puis recommencer.

### ✅ Vérification §4.5

```cmd
dir C:\Users\Shadow\Hunyuan3D-Part\P3-SAM\weights\p3sam.safetensors
```

Attendu : fichier listé, taille de l'ordre de plusieurs centaines de Mo.

### 4.6 Test rapide de P3-SAM (en dehors de ce projet)

```cmd
cd C:\Users\Shadow\Hunyuan3D-Part\P3-SAM\demo
python auto_mask.py --ckpt_path ..\weights\p3sam.safetensors --mesh_path assets\1.glb --output_path results\1
```

### ✅ Vérification §4.6

```cmd
dir results\1.glb results\1.ply results\1_aabb.npy results\1_face_ids.npy
```

Attendu : les 4 fichiers sont listés. Si oui, P3-SAM est opérationnel — on peut passer à ce projet.

---

## 5. Installation de ce projet

```cmd
cd C:\Users\Shadow
git clone https://github.com/zionhigt/hunyuan3d-part-segmenter.git
cd hunyuan3d-part-segmenter
pip install -r requirements.txt
```

> Le `requirements.txt` de **ce** projet n'installe que des libs Python génériques (`trimesh`, `pyyaml`, `tqdm`, …). Il **n'installe pas** Hunyuan3D-Part — ça, c'est l'étape 4.

Éditer `config.yaml` pour pointer sur ton clone Tencent et tes poids (slashs `/`, pas `\`, et **pas** d'espaces non échappés) :

```yaml
hy3d_part_root: "C:/Users/Shadow/Hunyuan3D-Part"
p3sam_ckpt_path: "C:/Users/Shadow/Hunyuan3D-Part/P3-SAM/weights/p3sam.safetensors"
```

> Pas besoin de `PYTHONPATH` ni de `pip install -e` — ce projet **shell out** sur `auto_mask.py`, il a juste besoin du chemin vers le clone et vers le checkpoint.

---

## 6. Vérification finale

```cmd
python src\check_env.py
```

### ✅ Vérification §6

Attendu dans la sortie :
- `cuda.is_available()` ➜ `True`
- `GPU[0]` ➜ `NVIDIA RTX A4500`
- `GPU[0] VRAM` ➜ ~20 GiB
- `torch` ➜ `2.4.0+cu121`

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

### ✅ Vérification §7

À la première run, `nvidia-smi` (dans un **second** terminal, pendant l'exécution) doit montrer un process Python qui consomme de la VRAM. Si la VRAM reste à 0 et que c'est lent ➜ fallback CPU silencieux, retour §3.

---

## 8. Dépannage

| Symptôme | Cause probable | Action |
|---|---|---|
| `torch.cuda.is_available() == False` | mauvais wheel CUDA / pilote non chargé | réinstaller la combo cu121 §3 ; vérifier `nvidia-smi`. |
| `chamfer3D` ne compile pas (`cl.exe`/`nvcc` errors) | Build Tools « Desktop C++ » incomplets, CUDA Toolkit pas installé, ou terminal non relancé | refaire §1 *complet*, fermer/rouvrir le terminal, vérifier `where cl` / `where nvcc` / `nvcc --version`. |
| `nvcc fatal : Cannot find compiler 'cl.exe'` | `cl.exe` pas sur le PATH du shell courant | utiliser "x64 Native Tools Command Prompt for VS 2022" puis `conda activate hy3d-part`. |
| `ModuleNotFoundError: sonata` au lancement de `auto_mask.py` | Sonata pas installé ou env actif différent | refaire §4.3 dans `(hy3d-part)`. |
| `pip install -e .` de Sonata casse sur `flash-attn`/`pointops`/`torch-scatter` | dep C++ non-prebuilt pour cu121+win | **m'envoyer le message d'erreur exact**, on patchera §4.3 avec la wheel ou la version pinnée qui marche. |
| `P3-SAM script not found at .../P3-SAM/demo/auto_mask.py` | mauvais `hy3d_part_root` | vérifier le chemin dans `config.yaml` (slashs `/`). |
| `P3-SAM checkpoint not found` | `p3sam_ckpt_path` faux ou poids non téléchargés | refaire §4.5. |
| `huggingface-cli: command not found` | `huggingface_hub` pas installé dans l'env actif | `pip install huggingface_hub` dans `(hy3d-part)`. |
| `403`/`Access denied` sur HF | conditions d'utilisation pas acceptées | aller sur <https://huggingface.co/tencent/Hunyuan3D-Part>, accepter, puis `huggingface-cli login`. |
| `OutOfMemoryError` | mesh trop dense | baisser `p3sam_point_num` dans `config.yaml`, ou décimer le mesh en amont. |
| Cache HF sature `C:` | poids gros | rediriger `HF_HOME` (§4.5). |
| Lenteur extrême | fallback CPU | vérifier `nvidia-smi` **pendant** l'exécution ; GPU à 0 % = fallback CPU silencieux. |
| L'env actif n'est pas `hy3d-part` | activation oubliée | `conda activate hy3d-part`. |
| Veux X-Part | poids upstream `TODO` | non supporté ; attendre que Tencent publie les poids X-Part puis ouvrir une issue ici. |
