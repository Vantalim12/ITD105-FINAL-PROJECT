@echo off
echo ============================================================
echo   SDG 14: Fish Conservation Status Prediction System
echo   Setup and Launch Script
echo ============================================================
echo.

echo Step 1: Installing required packages...
pip install -r requirements.txt
if errorlevel 1 (
    echo Error installing packages!
    pause
    exit /b 1
)
echo.

echo Step 2: Generating dataset...
python generate_dataset.py
if errorlevel 1 (
    echo Error generating dataset!
    pause
    exit /b 1
)
echo.

echo Step 3: Training ML models...
echo This may take a few minutes...
python train_models.py
if errorlevel 1 (
    echo Error training models!
    pause
    exit /b 1
)
echo.

echo Step 4: Launching web application...
echo The app will open in your browser at http://localhost:8501
echo Press Ctrl+C to stop the server.
echo.
streamlit run app.py

pause

