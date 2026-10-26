@echo off
cd /d %~dp0
conda run --no-capture-output -n sar_method python run.py --input-dir data\example\demo --output-dir outputs\demo
pause
