@echo off
REM scripts\dev_shell.bat
REM Ouvre un shell pret a compiler des extensions C++/CUDA pour hy3d-part :
REM   1. charge vcvars64.bat (cl.exe / link.exe sur PATH)
REM   2. active conda hy3d-part
REM   3. laisse la main
REM
REM Usage : double-clic, ou depuis Anaconda Prompt : scripts\dev_shell.bat
REM
REM Adapte VCVARS si VS Build Tools est installe ailleurs.

set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" (
    echo [ERROR] vcvars64.bat introuvable a : %VCVARS%
    echo Edite scripts\dev_shell.bat pour pointer sur ton install VS Build Tools.
    pause
    exit /b 1
)
call "%VCVARS%"

set "CONDA_HOOK=%USERPROFILE%\miniconda3\Scripts\activate.bat"
if not exist "%CONDA_HOOK%" set "CONDA_HOOK=%USERPROFILE%\anaconda3\Scripts\activate.bat"
if not exist "%CONDA_HOOK%" (
    echo [ERROR] conda activate.bat introuvable. Edite scripts\dev_shell.bat.
    pause
    exit /b 1
)
call "%CONDA_HOOK%" hy3d-part
if errorlevel 1 (
    echo [WARN] env hy3d-part introuvable. Lance d'abord scripts\setup_hy3d_part.bat.
)

echo.
echo === Dev shell pret ===
where cl
where nvcc
python -c "import sys; print('python  :', sys.executable)" 2>nul
echo.
cmd /k
