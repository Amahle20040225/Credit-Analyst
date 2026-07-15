@echo off

echo ============================================
echo FNB DataQuest 2026
echo Interpretable Credit Risk Platform
echo ============================================

echo.
echo Installing dependencies...
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

echo.
echo Launching Streamlit Dashboard...
echo Open browser at http://localhost:8501
echo.

python -m streamlit run app.py --server.port 8501 --server.address 0.0.0.0

pause