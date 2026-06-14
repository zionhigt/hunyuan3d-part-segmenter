@echo off
REM Launch a batch segmentation run inside the dedicated conda env `hy3d-part`.
REM Usage: double-click or run from cmd:  scripts\run_batch.bat

setlocal

REM --- Adjust if your Miniconda is installed elsewhere ---
set "CONDA_HOOK=%USERPROFILE%\miniconda3\Scripts\activate.bat"
if not exist "%CONDA_HOOK%" set "CONDA_HOOK=%USERPROFILE%\anaconda3\Scripts\activate.bat"
if not exist "%CONDA_HOOK%" (
    echo [ERROR] Could not find conda activate.bat. Edit scripts\run_batch.bat to set CONDA_HOOK.
    exit /b 1
)

call "%CONDA_HOOK%" hy3d-part
if errorlevel 1 (
    echo [ERROR] Failed to activate conda env "hy3d-part". See INSTALL.md.
    exit /b 1
)

REM Move to the project root (this script lives in scripts\)
pushd "%~dp0\.."

python src\batch.py %*
set "EC=%ERRORLEVEL%"

popd
endlocal & exit /b %EC%
