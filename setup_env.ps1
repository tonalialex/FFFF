# Create/repair the Anaconda environment for this project.
# Run from this directory in PowerShell:
#   powershell -ExecutionPolicy Bypass -File .\setup_env.ps1
$ErrorActionPreference = 'Stop'
$envName = 'sar_method'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$conda = (Get-Command conda -ErrorAction SilentlyContinue).Source
if (-not $conda) { throw 'conda was not found on PATH. Open Anaconda Prompt and retry.' }

$envs = & conda env list
if ($envs -match "(^|\s)$envName(\s|$)") {
    Write-Host "Conda environment '$envName' already exists; updating pip packages."
} else {
    Write-Host "Creating conda environment '$envName' (Python 3.12)..."
    & conda create -y -n $envName python=3.12 pip --override-channels -c https://repo.anaconda.com/pkgs/main
    if ($LASTEXITCODE -ne 0) { throw "conda create failed ($LASTEXITCODE)" }
}

Write-Host 'Installing Python packages (PyTorch CUDA 11.8 wheel, about 2.5 GB on first install)...'
& conda run --no-capture-output -n $envName python -m pip install --upgrade pip
& conda run --no-capture-output -n $envName python -m pip install `
    --index-url https://pypi.org/simple `
    --extra-index-url https://download.pytorch.org/whl/cu118 `
    --retries 10 --timeout 60 `
    numpy==2.2.6 scipy==1.13.1 opencv-python-headless==4.11.0.86 Pillow==11.3.0 `
    pandas==2.2.2 matplotlib==3.9.2 contourpy==1.3.3 torch==2.7.1+cu118 torchvision==0.22.1+cu118
if ($LASTEXITCODE -ne 0) { throw "pip install failed ($LASTEXITCODE)" }

Write-Host ''
Write-Host 'Verifying environment...'
& conda run --no-capture-output -n $envName python run.py --check-env
if ($LASTEXITCODE -ne 0) { throw "environment check failed ($LASTEXITCODE)" }

Write-Host ''
Write-Host "Environment is ready: $envName"
Write-Host "Run the demo:"
Write-Host "  conda run --no-capture-output -n $envName python run.py --input-dir data/example --output-dir outputs/example"
Write-Host "Or activate it first:"
Write-Host "  conda activate $envName"
Write-Host "  python run.py --input-dir data/example --output-dir outputs/example"
