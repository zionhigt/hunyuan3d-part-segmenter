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
REM
REM NOTE: pas de blocs `if (...)` autour de variables contenant des parentheses :
REM le chemin VS contient (x86), ce qui casse le parser cmd a la parse-time
REM expansion. On passe par des labels goto pour eviter le piege.

set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" goto :vcvars_missing

call "%VCVARS%"

REM PyTorch's cpp_extension verifie que DISTUTILS_USE_SDK=1 quand vcvars est
REM deja charge, sinon il leve "It seems that the VC environment is activated
REM but DISTUTILS_USE_SDK is not set". On le pose ici pour tout le shell.
set DISTUTILS_USE_SDK=1

set "CONDA_HOOK=%USERPROFILE%\miniconda3\Scripts\activate.bat"
if not exist "%CONDA_HOOK%" set "CONDA_HOOK=%USERPROFILE%\anaconda3\Scripts\activate.bat"
if not exist "%CONDA_HOOK%" goto :conda_missing

call "%CONDA_HOOK%" hy3d-part
if errorlevel 1 echo [WARN] env hy3d-part introuvable. Lance d'abord scripts\setup_hy3d_part.bat.

echo.
echo === Dev shell pret ===
where cl
where nvcc
python -c "import sys; print('python  :', sys.executable)" 2>nul
echo.
cmd /k
goto :eof

:vcvars_missing
echo [ERROR] vcvars64.bat introuvable a : "%VCVARS%"
echo Edite scripts\dev_shell.bat pour pointer sur ton install VS Build Tools.
pause
exit /b 1

:conda_missing
echo [ERROR] conda activate.bat introuvable. Edite scripts\dev_shell.bat.
pause
exit /b 1
