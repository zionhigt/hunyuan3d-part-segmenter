@echo off
REM scripts\diag_hy3d.bat
REM Capture l'etat de l'env conda hy3d (Hunyuan3D-2.1) pour servir de reference
REM a l'install de hy3d-part. Le script ne modifie rien.
REM
REM Usage (Anaconda Prompt) :
REM   conda activate hy3d
REM   scripts\diag_hy3d.bat > hy3d_diag.txt 2>&1
REM
REM Puis colle hy3d_diag.txt dans le chat.

setlocal EnableDelayedExpansion

echo ===== 1. ENV ACTIF =====
echo CONDA_DEFAULT_ENV=%CONDA_DEFAULT_ENV%
where python
python --version

echo.
echo ===== 2. PYTORCH / CUDA RUNTIME =====
python -c "import torch; print('torch        :', torch.__version__); print('cuda built   :', torch.version.cuda); print('cudnn        :', torch.backends.cudnn.version()); print('cuda avail   :', torch.cuda.is_available()); print('gpu          :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'); print('arch list    :', torch.cuda.get_arch_list() if torch.cuda.is_available() else 'N/A')"

echo.
echo ===== 3. DRIVER / NVCC =====
nvcc --version
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

echo.
echo ===== 4. C++ BUILD TOOLS =====
echo --- where cl ---
where cl 2>&1
echo --- Recherche cl.exe (Visual Studio / BuildTools) ---
dir /s /b "C:\Program Files\Microsoft Visual Studio\2022\*cl.exe" 2>nul
dir /s /b "C:\Program Files (x86)\Microsoft Visual Studio\2022\*cl.exe" 2>nul
dir /s /b "C:\BuildTools\*cl.exe" 2>nul
echo --- Recherche vcvars64.bat ---
dir /s /b "C:\Program Files\Microsoft Visual Studio\*vcvars64.bat" 2>nul
dir /s /b "C:\Program Files (x86)\Microsoft Visual Studio\*vcvars64.bat" 2>nul

echo.
echo ===== 5. PAQUETS D'INTERET (versions installees) =====
for %%P in (torch torchvision sonata chamfer-3D chamfer_3D flash-attn xformers torch-scatter torch-cluster pointops pytorch3d trimesh huggingface_hub viser fpsample numba gradio einops omegaconf diffusers transformers accelerate) do (
  echo --- %%P ---
  pip show %%P 2>nul | findstr /C:"Name" /C:"Version" /C:"Location"
)

echo.
echo ===== 6. SMOKE TESTS IMPORT (l'import compte plus que pip show) =====
python -c "import chamfer_3D; print('chamfer_3D OK:', chamfer_3D.__file__)" 2>&1
python -c "import sonata; print('sonata OK   :', sonata.__file__)" 2>&1
python -c "import flash_attn; print('flash_attn  :', flash_attn.__version__)" 2>&1
python -c "import xformers; print('xformers    :', xformers.__version__)" 2>&1
python -c "import torch_scatter; print('torch_scat  :', torch_scatter.__version__)" 2>&1
python -c "import torch_cluster; print('torch_clust :', torch_cluster.__version__)" 2>&1
python -c "import pointops; print('pointops    :', pointops.__file__)" 2>&1
python -c "import pytorch3d; print('pytorch3d   :', pytorch3d.__version__)" 2>&1

echo.
echo ===== 7. HUNYUAN3D-2.1 (clone local + module) =====
dir /b "C:\Users\Shadow\Hunyuan3D-2.1" 2>nul
dir /b "C:\Users\Shadow\Hunyuan3D-2" 2>nul
dir /b "C:\Users\Shadow\Hunyuan3D" 2>nul
python -c "import hy3dgen; print('hy3dgen     :', hy3dgen.__file__)" 2>&1
python -c "import hy3dpaint; print('hy3dpaint   :', hy3dpaint.__file__)" 2>&1

echo.
echo ===== 8. SOURCES POTENTIELLES (pour Sonata / chamfer3D installes en editable) =====
pip list --format=freeze 2>nul | findstr /C:"@ file" /C:"@ git"
echo --- editable installs ---
pip list -e 2>nul

echo.
echo ===== 9. PIP FREEZE COMPLET =====
pip freeze

endlocal
