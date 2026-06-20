# INSTALL — Shadow PC Power (RTX A4500, Windows 11)

Procédure **calquée sur l'env `hy3d`** (Hunyuan3D-2.1) déjà fonctionnel sur cette machine, plutôt que sur la combo papier upstream. On clone `hy3d → hy3d-part` pour hériter d'une base validée (Python 3.10, **torch 2.5.1+cu124**, CUDA 12.4), puis on couche les deps spécifiques à P3-SAM.

> **X-Part n'est PAS supporté** : poids upstream `TODO`. Ce projet utilise uniquement **P3-SAM**.

## Comment utiliser ce guide

Chaque section finit par un bloc **✅ Vérification** : exécute, colle la sortie si ça coince — on patchera ce doc plutôt que de continuer sur un sol pourri. Toutes les commandes tournent dans **Anaconda Prompt**, sauf §4.4 (`chamfer3D`) qui exige **`scripts\dev_shell.bat`** pour avoir `cl.exe` sur le PATH.

---

## 1. Prérequis (déjà présents sur ta machine — à vérifier)

Ton diagnostic montre :
- Driver NVIDIA **572.16** / CUDA max **12.8** (`nvidia-smi`) ✅
- CUDA Toolkit **12.4** (`nvcc --version`) ✅
- VS Build Tools 2022 : `cl.exe` à `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe` ✅
- `vcvars64.bat` : `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat` ✅
- conda env **hy3d** opérationnel (torch 2.5.1+cu124, Python 3.10.20) ✅
- Hunyuan3D-2.1 cloné à `C:\Users\Shadow\Hunyuan3D-2.1\` ✅

Rien à installer ici. Si un de ces 6 points n'est plus vrai, refais le diag : `scripts\diag_hy3d.bat > hy3d_diag.txt 2>&1`.

---

## 2. Repartir propre : supprimer l'ancien `hy3d-part`

L'env `hy3d-part` actuel a torch 2.6 et pas d'outils build. On le nuke pour cloner `hy3d` à la place.

```cmd
conda deactivate
conda env remove -n hy3d-part
```

### ✅ Vérification §2

```cmd
conda env list
```

Attendu : `hy3d-part` **n'apparaît plus**. Tu dois voir `base`, `hy3d`, `kontext` uniquement.

---

## 3. Cloner `hy3d` → `hy3d-part`

Depuis Anaconda Prompt :

```cmd
cd C:\Users\Shadow\hunyuan3d-part-segmenter
scripts\setup_hy3d_part.bat
```

Le script :
1. Clone `hy3d` → `hy3d-part` (~3-5 min, copie quelques Go).
2. Active `hy3d-part`.
3. Vérifie que torch+CUDA sont vivants après clone.
4. Ajoute `viser` + `fpsample` (les seules deps P3-SAM absentes de `hy3d`).
5. Joue `pip install -r requirements.txt` du projet (déjà 99% redondant avec `hy3d`, c'est juste une safety net).

### ✅ Vérification §3

```cmd
conda activate hy3d-part
python -c "import torch, trimesh, viser, fpsample, huggingface_hub; print('torch', torch.__version__, '| cuda', torch.cuda.is_available()); print('layered deps OK')"
```

Attendu :
```
torch 2.5.1+cu124 | cuda True
layered deps OK
```

> Si un import échoue, **arrête** et colle l'erreur — on ajustera `setup_hy3d_part.bat` ou `requirements.txt` plutôt que de bricoler à la main.

---

## 4. Hunyuan3D-Part : clone + chamfer3D + Sonata + poids

Tout ce qui suit s'installe **dans** ou **à côté de** `C:\Users\Shadow\Hunyuan3D-Part\`. Ce repo Tencent est séparé de Hunyuan3D-2.1.

Liens :
- GitHub : <https://github.com/Tencent-Hunyuan/Hunyuan3D-Part>
- HuggingFace : <https://huggingface.co/tencent/Hunyuan3D-Part>
- Forks Windows à investiguer (tu utilises déjà ce pattern pour 2.1 via `lzz19980125/Hunyuan3D-2.1-Windows`) : chercher un `Hunyuan3D-Part-Windows` ou équivalent **avant** de te battre avec le build natif. Si un tel fork existe et fournit des wheels prebuilt, dis-le moi — on le préfère.

### 4.1 Clone du repo Tencent

```cmd
cd C:\Users\Shadow
git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-Part.git
```

### ✅ Vérification §4.1

```cmd
dir C:\Users\Shadow\Hunyuan3D-Part\P3-SAM\demo\auto_mask.py
dir C:\Users\Shadow\Hunyuan3D-Part\P3-SAM\utils\chamfer3D\setup.py
```

Attendu : les deux fichiers existent.

### 4.2 Sonata — pas besoin du repo Facebook

Bonne surprise : le code Sonata est **bundle** dans `Hunyuan3D-Part/XPart/partgen/models/sonata/`. Le README P3-SAM dit d'installer le repo `facebookresearch/sonata`, mais c'est trompeur — `model.py` du repo Tencent ajoute `XPart/partgen` au `sys.path` et fait `from models import sonata`, donc il utilise sa copie locale.

**On évite donc** de cloner facebookresearch/sonata, **et surtout** d'installer `flash-attn` (pas de wheel Windows). À la place :

1. **Installer les 2 deps natives de Sonata qui ont des wheels Windows** :

```cmd
conda activate hy3d-part
pip install spconv-cu124
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
```

2. **Patcher le Sonata bundle pour désactiver `flash_attn`** (le fallback PyTorch standard est déjà implémenté dans `SerializedAttention.forward`, branche `if not self.enable_flash:`). Le patcher règle aussi un chemin Linux codé en dur (`/root/sonata`) dans `P3-SAM/model.py` :

```cmd
cd C:\Users\Shadow\hunyuan3d-part-segmenter
python scripts\patch_hy3d_part.py
```

Sortie attendue :
```
[DONE] sonata/model.py: disable flash_attn: patched ...
[DONE] P3-SAM/model.py: fix /root/sonata path: patched ...
```

(Si tu relances : `[OK] ... already patched`.)

### ✅ Vérification §4.2

```cmd
python -c "import torch; import spconv.pytorch as spconv; import torch_scatter; print('spconv OK | torch_scatter', torch_scatter.__version__)"
```

Attendu : `spconv OK | torch_scatter <version>`.

> Si `pip install spconv-cu124` ou `torch-scatter` casse ➜ colle l'erreur. Pour torch-scatter, l'index PyG cache un sous-dossier `torch-2.5.1+cu124` qui contient la wheel pour Python 3.10 Windows.

### 4.3 Build de `chamfer3D` (kernel CUDA, depuis le dev shell)

`cl.exe` n'est pas sur le PATH d'un Anaconda Prompt classique. Le script `dev_shell.bat` règle ça en chargeant `vcvars64.bat`.

```cmd
cd C:\Users\Shadow\hunyuan3d-part-segmenter
scripts\dev_shell.bat
```

Tu obtiens un nouveau shell où `cl` et `nvcc` sont dispo, et `hy3d-part` est actif. **Dans ce shell**, build :

```cmd
cd C:\Users\Shadow\Hunyuan3D-Part\P3-SAM\utils\chamfer3D
python setup.py install
```

### ✅ Vérification §4.3

Toujours dans le dev shell :

```cmd
python -c "import chamfer_3D; print('chamfer3D OK:', chamfer_3D.__file__)"
```

Attendu : `chamfer3D OK: ...site-packages\chamfer_3D...`. Si le nom du module diffère (ex. `chamfer3D`, `chamfer`), `pip show chamfer-3D` te le dira.

> Le shell normal (sans dev_shell) suffit pour la suite — l'extension est compilée une fois, puis importable partout.

### 4.4 Poids P3-SAM

Si tu ne l'as pas déjà fait, rediriger le cache HF vers le disque étendu (extension Shadow) :

```cmd
setx HF_HOME "D:\hf_cache"
```

Fermer/rouvrir Anaconda Prompt, `conda activate hy3d-part`, puis :

```cmd
huggingface-cli download tencent/Hunyuan3D-Part p3sam.safetensors --local-dir C:\Users\Shadow\Hunyuan3D-Part\P3-SAM\weights
```

### ✅ Vérification §4.4

```cmd
dir C:\Users\Shadow\Hunyuan3D-Part\P3-SAM\weights\p3sam.safetensors
```

Attendu : fichier listé, plusieurs centaines de Mo.

### 4.5 Smoke test P3-SAM en standalone

```cmd
cd C:\Users\Shadow\Hunyuan3D-Part\P3-SAM\demo
python auto_mask.py --ckpt_path ..\weights\p3sam.safetensors --mesh_path assets\1.glb --output_path results\1
```

### ✅ Vérification §4.5

```cmd
dir results\1.glb results\1.ply results\1_aabb.npy results\1_face_ids.npy
```

Attendu : les 4 sorties existent. Si oui → P3-SAM tourne, on peut brancher ce projet.

---

## 5. Brancher ce projet

`config.yaml` est déjà configuré pour Shadow par défaut. Vérifie juste qu'il pointe bien :

```yaml
hy3d_part_root: "C:/Users/Shadow/Hunyuan3D-Part"
p3sam_ckpt_path: "C:/Users/Shadow/Hunyuan3D-Part/P3-SAM/weights/p3sam.safetensors"
```

> Pas besoin de `PYTHONPATH` ni de `pip install -e .` — ce projet **shell out** sur `auto_mask.py`.

---

## 6. Vérification finale

```cmd
conda activate hy3d-part
python src\check_env.py
```

### ✅ Vérification §6

Attendu :
- `cuda.is_available()` ➜ `True`
- `GPU[0]` ➜ `NVIDIA RTX A4500`
- `GPU[0] VRAM` ➜ ~20 GiB
- `torch` ➜ `2.5.1+cu124`

---

## 7. Premier test

```cmd
REM Deposer un GLB monobloc dans input\, puis :
python src\single.py --glb input\vehicle.glb
python src\batch.py
python src\single.py --glb input\vehicle.glb --export-mode split
scripts\run_batch.bat
```

Pendant que ça tourne, dans un **second** Anaconda Prompt :

```cmd
nvidia-smi
```

Tu dois voir un process Python qui consomme de la VRAM. Si VRAM à 0 et lenteur extrême ➜ fallback CPU, retour §3.

---

## 8. Dépannage

| Symptôme | Cause probable | Action |
|---|---|---|
| `conda env remove` refuse / "env in use" | env actif | `conda deactivate` d'abord, fermer les autres terminaux. |
| Clone hy3d→hy3d-part lent | volume du clone (~Go) | normal, laisser tourner 3-5 min. |
| `fpsample` build fail : `'multiple_interpreters' n'est pas membre de 'pybind11'` | `pybind11 2.13.4` hérité, fpsample exige ≥ 2.14 | `pip install -U "pybind11>=2.14"` puis retry. Déjà géré par `setup_hy3d_part.bat`. |
| `torch.cuda.is_available()==False` après clone | conda a recopié partiellement | `conda env remove -n hy3d-part` puis relancer §3. |
| `cl.exe` ou `nvcc` introuvables au build chamfer3D | shell normal, pas `dev_shell.bat` | relancer `scripts\dev_shell.bat`, refaire le build dedans. |
| Build chamfer3D : `unsupported MSVC` | MSVC 14.44 vs CUDA 12.4 — normalement OK, sinon installer Windows 11 SDK | refaire VS Installer, ajouter Windows 11 SDK. |
| Build chamfer3D : longue liste de warnings puis OK | normal | ignorer warnings, vérifier l'import (§4.3 ✅). |
| `ModuleNotFoundError: spconv` / `torch_scatter` au lancement de `auto_mask.py` | §4.2 pas faite | refaire §4.2 dans `(hy3d-part)`. |
| `AssertionError: Make sure flash_attn is installed.` | patcher §4.2 pas appliqué | `python scripts\patch_hy3d_part.py`, vérifier `[DONE]` ou `[OK]` en sortie. |
| `from models import sonata` ➜ `ModuleNotFoundError` | tu as cloné `facebookresearch/sonata` au lieu d'utiliser le bundle Tencent | le code Tencent utilise `XPart/partgen/models/sonata/`, pas le repo Facebook. Désinstalle `sonata` si tu l'avais : `pip uninstall sonata`. |
| `P3-SAM script not found` au lancement de ce projet | mauvais `hy3d_part_root` | vérifier chemin dans `config.yaml`. |
| `P3-SAM checkpoint not found` | §4.4 pas faite | refaire le `huggingface-cli download`. |
| `OutOfMemoryError` | mesh trop dense | baisser `p3sam_point_num` dans `config.yaml`, décimer en amont. |
| Cache HF sature `C:` | poids gros | rediriger `HF_HOME` (§4.4). |
| Lenteur extrême, GPU à 0% | fallback CPU silencieux | check `nvidia-smi` pendant exécution, vérifier env actif. |
| Veux X-Part | poids upstream `TODO` | non supporté ; attendre publication Tencent. |
