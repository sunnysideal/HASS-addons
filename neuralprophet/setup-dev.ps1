# PowerShell setup script for local development
Write-Host "Setting up NeuralProphet local development environment..." -ForegroundColor Green

# Check if .env exists
if (!(Test-Path ".env")) {
    Write-Host "Creating .env file from template..." -ForegroundColor Yellow
    Copy-Item ".env.example" ".env"
    Write-Host "Please edit .env file with your Home Assistant URL and token!" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "To create a long-lived access token:" -ForegroundColor Cyan
    Write-Host "  1. Open Home Assistant" -ForegroundColor Cyan
    Write-Host "  2. Profile -> Security -> Long-Lived Access Tokens" -ForegroundColor Cyan
    Write-Host "  3. Create Token and copy it to .env" -ForegroundColor Cyan
    Write-Host ""
    
    # Open .env in notepad
    notepad .env
    
    Write-Host "Press Enter after you've configured .env..." -ForegroundColor Yellow
    Read-Host
}

# Check if virtual environment exists
if (!(Test-Path "venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"

# Install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Yellow
pip install --upgrade pip
pip install python-dotenv requests pyyaml pandas torch numpy pytorch-lightning neuralprophet

Write-Host ""
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "To run the addon locally:" -ForegroundColor Cyan
Write-Host "  python dev/dev_run.py" -ForegroundColor White
Write-Host ""
Write-Host "Make sure to edit rootfs/neuralprophet.yaml with your sensor entity IDs" -ForegroundColor Yellow
