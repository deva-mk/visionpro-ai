@echo off
echo ============================================
echo   VisionPro - Professional Vision AI
echo ============================================
echo.
echo Installing packages...
python -m pip install flask flask-cors numpy Pillow opencv-python --timeout 300 -q
echo.
echo Starting VisionPro...
echo.
echo ============================================
echo   Open your browser:  http://127.0.0.1:5000
echo   Login: admin / admin123
echo ============================================
echo.
python app.py
pause
