# RCT Field Flow - Deployment Helper (Windows PowerShell)
# Quick setup script for Docker and deployment

Write-Host "================================" -ForegroundColor Green
Write-Host "RCT Field Flow - Deployment Helper" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Green
Write-Host ""

# Check if Docker is installed
$docker = $null
try {
    $docker = docker --version 2>$null
    Write-Host "✓ Docker found" -ForegroundColor Green
    Write-Host "  $docker"
} catch {
    Write-Host "❌ Docker is not installed" -ForegroundColor Red
    Write-Host "Install from: https://www.docker.com/products/docker-desktop"
    exit 1
}

# Check if Docker Compose is installed
$docker_compose = $null
try {
    $docker_compose = docker-compose --version 2>$null
    Write-Host "✓ Docker Compose found" -ForegroundColor Green
    Write-Host "  $docker_compose"
} catch {
    Write-Host "❌ Docker Compose is not installed" -ForegroundColor Red
    Write-Host "Install from: https://docs.docker.com/compose/install/"
    exit 1
}

# Check for .env file
if (-not (Test-Path ".env")) {
    Write-Host ""
    Write-Host "⚠ .env file not found" -ForegroundColor Yellow
    Write-Host "Creating from .env.example..."
    Copy-Item ".env.example" ".env"
    Write-Host "Please edit .env and add your SurveyCTO credentials" -ForegroundColor Yellow
    Read-Host "Press Enter to continue"
}

# Check for .streamlit/secrets.toml
if (-not (Test-Path ".streamlit\secrets.toml")) {
    Write-Host ""
    Write-Host "⚠ .streamlit\secrets.toml not found" -ForegroundColor Yellow
    Write-Host "Creating from .streamlit\secrets.toml.example..."
    Copy-Item ".streamlit\secrets.toml.example" ".streamlit\secrets.toml"
    Write-Host "Please edit .streamlit\secrets.toml for local development" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "================================" -ForegroundColor Green
Write-Host "Choose deployment option:" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Green
Write-Host ""
Write-Host "1. 🐳 Build & run with Docker Compose locally"
Write-Host "2. ☁️  Prepare for Streamlit Cloud deployment"
Write-Host "3. 🔧 Build Docker image only (no run)"
Write-Host "4. 📋 Show deployment guide"
Write-Host ""

$choice = Read-Host "Select option (1-4)"

switch ($choice) {
    "1" {
        Write-Host ""
        Write-Host "Starting Docker Compose..." -ForegroundColor Green
        Write-Host ""
        docker-compose up
    }
    "2" {
        Write-Host ""
        Write-Host "================================" -ForegroundColor Green
        Write-Host "Streamlit Cloud Deployment" -ForegroundColor Green
        Write-Host "================================" -ForegroundColor Green
        Write-Host ""
        Write-Host "Steps to deploy to Streamlit Cloud:"
        Write-Host ""
        Write-Host "1. Ensure all changes are committed to GitHub:"
        Write-Host "   git add ."
        Write-Host "   git commit -m 'Prepare for Streamlit Cloud deployment'"
        Write-Host "   git push origin master"
        Write-Host ""
        Write-Host "2. Go to: https://streamlit.io/cloud"
        Write-Host ""
        Write-Host "3. Sign in with GitHub and authorize Streamlit"
        Write-Host ""
        Write-Host "4. Click 'New app' and select:"
        Write-Host "   - Repository: ajolex/rct_field_flow"
        Write-Host "   - Branch: master"
        Write-Host "   - Main file path: rct_field_flow/app.py"
        Write-Host ""
        Write-Host "5. After deployment, add secrets in app settings:"
        Write-Host "   Settings → Secrets → Add from .streamlit\secrets.toml"
        Write-Host ""
        Write-Host "Ready? Push to GitHub now:" -ForegroundColor Yellow
        Read-Host "Press Enter to commit and push"
        
        git add .
        git commit -m "Deployment: Add Streamlit Cloud configuration"
        git push origin master
        
        Write-Host "✓ Pushed to GitHub" -ForegroundColor Green
        Write-Host "Now go to https://streamlit.io/cloud to deploy!"
    }
    "3" {
        Write-Host ""
        Write-Host "Building Docker image..." -ForegroundColor Green
        Write-Host ""
        docker build -t rct-field-flow:latest .
        Write-Host ""
        Write-Host "✓ Build complete!" -ForegroundColor Green
        Write-Host "To run: docker run -p 8501:8501 --env-file .env rct-field-flow:latest"
    }
    "4" {
        Write-Host ""
        Write-Host "Deployment guide:" -ForegroundColor Green
        Write-Host ""
        Get-Content DEPLOYMENT.md | Select-Object -First 100
    }
    default {
        Write-Host "Invalid option. Please try again." -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "Done!" -ForegroundColor Green
Write-Host ""
