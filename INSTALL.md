# INSTALL — Windows 11 natif / Shadow PC Power (RTX A4500, 20 Go VRAM, 28 Go RAM)

Procédure complète et reproductible sur une machine vierge. Tout le pipeline s'exécute dans un environnement conda **dédié** `hy3d-part`, **distinct** de l'environnement de génération (`hy3d`). P3-SAM/X-Part ont leurs propres poids et dépendances ; ne pas les mélanger avec l'env de génération.

---

## 1. Prérequis système

- **Miniconda** : <https://www.anaconda.com/download/success>
- **Git pour Windows** : <https://git-scm.com/download/win>
- **Visual Studio Build Tools 2022** — workload « *Desktop development with C++* » **complet** (nécessaire si l'un des composants doit être compilé). Cocher tout le workload, puis **redémarrer le terminal après installation** pour que `cl.exe` / `link.exe` soient sur le `PATH`. Téléchargement : <https://visualstudio.microsoft.com/visual-cpp-build-tools/>
- **Pilote NVIDIA** : déjà présent sur Shadow. Vérifier :
  ```cmd
  nvidia-smi
  ```
  Le chiffre « **CUDA Version** » en haut à droite = **version CUDA MAX supportée par le pilote** (pas une install CUDA, pas un wheel). Choisir un wheel PyTorch CUDA **≤ ce chiffre**. Sur Shadow Power le pilote rapporte typiquement CUDA 12.8 → tout wheel `cu124` / `cu126` convient.

---

## 2. Environnement conda DÉDIÉ

```cmd
conda create -n hy3d-part python=3.10 -y
conda activate hy3d-part
```

> **Insistance** : `hy3d-part` est SÉPARÉ de l'env de génération `hy3d`. Avant tout `pip install`, vérifier que le prompt affiche bien `(hy3d-part)`. Mélanger les deux environnements casse les dépendances.

---

## 3. PyTorch

Sélecteur officiel : <https://pytorch.org/get-started/locally/>.

Prendre la version CUDA recommandée par le `README` du repo Tencent **au moment du clone** ; à défaut, un wheel ≤ CUDA pilote (`cu124` ou `cu126`). Exemple :

```cmd
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Vérification :

```cmd
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Attendu : `... True NVIDIA RTX A4500`.

---

## 4. Récupération du pipeline Hunyuan3D-Part

Liens officiels :
- **Dépôt GitHub** : <https://github.com/Tencent-Hunyuan/Hunyuan3D-Part>
- **Composant P3-SAM** : <https://github.com/Tencent-Hunyuan/Hunyuan3D-Part/tree/main/P3-SAM>
- **Poids HuggingFace** : <https://huggingface.co/tencent/Hunyuan3D-Part>
- **Org GitHub** : <https://github.com/Tencent-Hunyuan>

Clés de recherche (si les URLs changent) :
- GitHub : `Tencent-Hunyuan Hunyuan3D-Part`
- HuggingFace : `tencent Hunyuan3D-Part`
- Papiers : `P3-SAM Native 3D Part Segmentation` (arXiv 2509.06784), `X-Part shape decomposition` (arXiv 2509.08643)

### Clone + dépendances

```cmd
cd C:\Users\Shadow
git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-Part.git
cd Hunyuan3D-Part
pip install -r requirements.txt
```

> L'env `(hy3d-part)` doit être actif.

### Build de composants natifs (le cas échéant)

Suivre le `README` du repo Tencent — ces étapes évoluent entre releases, ne pas les figer en dur. Sous Windows, **remplacer tout script `bash *.sh` de compilation par le `setup.py` correspondant** (`pip install -e .` dans le dossier du composant), à condition qu'un `setup.py` soit présent. Si la compilation casse, vérifier que les Build Tools VS 2022 « Desktop C++ » sont installés et que le terminal a été relancé après leur install.

### Poids

Les poids se téléchargent automatiquement via `huggingface_hub` au premier lancement dans `%USERPROFILE%\.cache\huggingface`.

- **Rediriger le cache HF** vers un disque avec de la place (extension stockage Shadow jusqu'à 5 To) :
  ```cmd
  setx HF_HOME "D:\hf_cache"
  ```
  Fermer/rouvrir le terminal pour que la variable s'applique.
- Si l'accès aux poids requiert l'acceptation de conditions sur HuggingFace :
  ```cmd
  huggingface-cli login
  ```

### Note X-Part

La version publique de X-Part est la version **light**, recommandée sur **meshes scannés ou générés par IA** (typiquement sorties Hunyuan V2.5 / V3.0). La version complète est uniquement disponible via Hunyuan3D-Studio et n'est pas couverte ici. X-Part est plus gourmand en VRAM que P3-SAM — sur 20 Go il peut OOM sur les meshes denses ; voir § Dépannage.

---

## 5. Installation de ce projet

```cmd
cd C:\Users\Shadow
git clone https://github.com/zionhigt/hunyuan3d-part-segmenter.git
cd hunyuan3d-part-segmenter
pip install -r requirements.txt
```

### Lier le repo Tencent à ce projet — deux options

**Option A — installation éditable (recommandée si un `setup.py` ou `pyproject.toml` racine est présent) :**

```cmd
pip install -e C:\Users\Shadow\Hunyuan3D-Part
```

**Option B — `PYTHONPATH` (si le repo Tencent n'expose pas de package installable) :**

```cmd
set "PYTHONPATH=C:\Users\Shadow\Hunyuan3D-Part;C:\Users\Shadow\Hunyuan3D-Part\P3-SAM;%PYTHONPATH%"
```

Persistant :

```cmd
setx PYTHONPATH "C:\Users\Shadow\Hunyuan3D-Part;C:\Users\Shadow\Hunyuan3D-Part\P3-SAM"
```

---

## 6. Vérification

```cmd
python src\check_env.py
```

Le diagnostic doit afficher : CUDA disponible, **NVIDIA RTX A4500**, ~20 Go VRAM, versions torch / CUDA, et l'importabilité des modules `P3SAM` / `XPart`.

---

## 7. Premier test

```cmd
REM Déposer un GLB monobloc dans input\, puis :
python src\single.py --glb input\vehicle.glb

REM Batch sur tout input\ :
python src\batch.py

REM Activer X-Part (plus lourd) :
python src\single.py --glb input\vehicle.glb --enable-xpart
```

Lanceur Windows tout-en-un :

```cmd
scripts\run_batch.bat
```

---

## 8. Dépannage

| Symptôme | Cause probable | Action |
|---|---|---|
| `torch.cuda.is_available() == False` | mauvais wheel CUDA / pilote NVIDIA non chargé | réinstaller le wheel PyTorch correspondant à la CUDA pilote (§3) ; vérifier `nvidia-smi`. |
| Erreur compilation `cl.exe` / `link.exe` introuvable | Build Tools « Desktop C++ » incomplets ou terminal non relancé | réinstaller le workload complet, **fermer et rouvrir** le terminal. |
| `OutOfMemoryError` sur X-Part | mesh trop dense pour 20 Go | désactiver X-Part (`--enable-xpart` retiré, P3-SAM seul suffit pour le rigging) ; ou réduire la densité du mesh en amont (paramètre de décimation côté pipeline de génération) ; ou activer un éventuel offload si proposé par le repo upstream. |
| Cache HuggingFace sature le disque système | poids gros + cache par défaut sur `C:` | rediriger `HF_HOME` vers un autre disque (§4 Poids). |
| Lenteur extrême (CPU fallback) | torch ne voit pas le GPU | vérifier `nvidia-smi` **pendant** l'exécution ; un GPU à 0 % indique un fallback CPU silencieux. Revoir l'install PyTorch. |
| `UpstreamNotInstalled: Could not import P3-SAM` | repo Tencent non installé ou pas sur `PYTHONPATH` | refaire §5 option A ou B. |
| L'env actif n'est pas `hy3d-part` | activation oubliée | `conda activate hy3d-part` ; le prompt doit afficher `(hy3d-part)`. |
