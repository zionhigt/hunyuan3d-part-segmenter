@echo off
REM scripts\setup_hy3d_part.bat
REM Cree l'env conda hy3d-part en clonant hy3d (Hunyuan3D-2.1 working env)
REM puis layer les deps P3-SAM par dessus.
REM
REM N'installe PAS sonata ni chamfer3D : ces deux la sont traites a part
REM dans INSTALL.md sections 4.3 / 4.4 (build C++/CUDA, dev_shell requis).
REM
REM Usage (Anaconda Prompt) :
REM   scripts\setup_hy3d_part.bat
REM Si hy3d-part existe deja, le script s'arrete pour eviter d'ecraser.

setlocal

echo === 1. Verification env source hy3d ===
conda env list | findstr /R /C:"^hy3d " >nul
if errorlevel 1 (
    echo [ERROR] env conda "hy3d" introuvable. Ce script clone hy3d -^> hy3d-part.
    exit /b 1
)

echo === 2. Verification env cible hy3d-part ===
conda env list | findstr /R /C:"^hy3d-part " >nul
if not errorlevel 1 (
    echo [ERROR] env "hy3d-part" existe deja. Supprime-le d'abord :
    echo     conda env remove -n hy3d-part
    exit /b 1
)

echo === 3. Clone hy3d -^> hy3d-part ===
call conda create --clone hy3d -n hy3d-part -y
if errorlevel 1 (
    echo [ERROR] clone conda echoue.
    exit /b 1
)

echo === 4. Activation hy3d-part ===
call conda activate hy3d-part
if errorlevel 1 (
    echo [ERROR] activation hy3d-part echouee.
    exit /b 1
)

echo === 5. Verifications heritage torch / cuda ===
python -c "import torch; print('torch:', torch.__version__, '| cuda built:', torch.version.cuda, '| avail:', torch.cuda.is_available())"
if errorlevel 1 (
    echo [ERROR] torch ne s'importe pas dans hy3d-part. Le clone a foire.
    exit /b 1
)

echo === 6. Layer P3-SAM deps (viser, fpsample) ===
REM huggingface_hub, trimesh, numba, gradio, einops, omegaconf, pyyaml, tqdm
REM sont deja la (herites de hy3d). On ajoute juste les deux specifiques.
REM
REM fpsample (dernieres versions) build depuis source et exige pybind11 >= 2.14
REM (symboles multiple_interpreters / per_interpreter_gil). hy3d herite de
REM pybind11 2.13.4 -> bump prealable. pybind11 etant header-only, ca
REM n'invalide pas les extensions deja compilees dans hy3d-part.
pip install -U "pybind11>=2.14"
if errorlevel 1 (
    echo [ERROR] upgrade pybind11 echoue.
    exit /b 1
)
pip install viser fpsample
if errorlevel 1 (
    echo [ERROR] pip install viser/fpsample echoue.
    exit /b 1
)

echo === 7. Install ce projet (requirements.txt) ===
pip install -r requirements.txt
if errorlevel 1 (
    echo [WARN] requirements.txt a renvoye une erreur (souvent un downgrade refuse).
    echo Continue, mais verifie pip check apres.
)

echo.
echo === Setup base hy3d-part : OK ===
echo Reste a faire (manuel, voir INSTALL.md) :
echo   - clone Tencent/Hunyuan3D-Part
echo   - clone facebookresearch/sonata + pip install -e .
echo   - build chamfer3D (depuis dev_shell.bat)
echo   - download poids p3sam.safetensors
echo.

endlocal
