@echo off
setlocal
chcp 65001 >nul
pushd "%~dp0"
set "PYTHONUTF8=1"
python -u "%~dp0launcher.py" %*
popd
