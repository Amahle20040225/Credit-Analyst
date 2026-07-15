# FNB DataQuest 2026 - Run Script
# This script installs dependencies and starts the Streamlit app

Write-Host "🇿🇦 FNB DataQuest 2026: Credit Risk Modeling Console" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host ""

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Python not found. Please install Python 3.12+" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "📦 Installing dependencies from requirements.txt..." -ForegroundColor Yellow
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

Write-Host ""
Write-Host "🚀 Starting Streamlit application..." -ForegroundColor Green
Write-Host "📱 Open your browser at: http://localhost:8501" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

python -m streamlit run app.py --server.port 8501 --server.address 0.0.0.0
