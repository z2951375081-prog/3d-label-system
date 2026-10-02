@echo off
cd /d "%~dp0"
start "OBJ 3D Label Studio" cmd /k "npm start"
timeout /t 2 /nobreak >nul
start "OBJ 3D Label Studio Web" "http://localhost:5173/"
