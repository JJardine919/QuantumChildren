@echo off
set VPS_IP=72.62.170.153
set VPS_USER=root

echo ========================================================
echo  DEPLOYING QUANTUM COMPRESSOR (WEB DASHBOARD)
echo ========================================================

echo.
echo [1/3] Uploading Web Server Code...
scp 01_Systems/QuantumCompression/quantum_server.py %VPS_USER%@%VPS_IP%:~/QuantumTradingLibrary/01_Systems/QuantumCompression/
scp quantum_monitor.html %VPS_USER%@%VPS_IP%:~/QuantumTradingLibrary/01_Systems/QuantumCompression/

echo.
echo [2/3] Installing Web Dependencies...
ssh %VPS_USER%@%VPS_IP% "apt-get install -y python3-pip; pip3 install --break-system-packages fastapi uvicorn websockets qutip scipy"

echo.
echo [3/3] Launching Background Service...
ssh %VPS_USER%@%VPS_IP% "killall uvicorn 2>/dev/null; nohup python3 ~/QuantumTradingLibrary/01_Systems/QuantumCompression/quantum_server.py > compressor.log 2>&1 &"

echo.
echo ========================================================
echo  SUCCESS!
echo  Your Quantum Compressor is live at:
echo  http://%VPS_IP%:8000
echo ========================================================
pause
