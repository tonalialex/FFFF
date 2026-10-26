$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
conda run --no-capture-output -n sar_method python run.py --input-dir data/example/demo --output-dir outputs/demo
