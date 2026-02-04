@echo off
set VPS_IP=72.62.170.153
set VPS_USER=root

echo ========================================================
echo  QUANTUM TRADING LIBRARY - VPS DEPLOYMENT (FORCE REFRESH)
echo ========================================================
echo.
echo [1/5] Cleaning old files on VPS...
ssh %VPS_USER%@%VPS_IP% "rm -rf ~/QuantumTradingLibrary/06_Integration ~/QuantumTradingLibrary/01_Systems"

echo.
echo [2/5] Creating folders...
ssh %VPS_USER%@%VPS_IP% "mkdir -p ~/QuantumTradingLibrary/01_Systems ~/QuantumTradingLibrary/06_Integration"

echo.
echo [3/5] Uploading Fresh Code...
scp -r 01_Systems %VPS_USER%@%VPS_IP%:~/QuantumTradingLibrary/
scp -r 06_Integration %VPS_USER%@%VPS_IP%:~/QuantumTradingLibrary/

echo.
echo [4/5] Installing Dependencies (Linux Compatible)...
ssh %VPS_USER%@%VPS_IP% "apt-get install -y python3-pip python3-numpy python3-pandas python3-torch"

echo.
echo [5/5] STARTING THE BRAIN (Linux Mode)...
ssh -t %VPS_USER%@%VPS_IP% "export PYTHONPATH=$PYTHONPATH:~/QuantumTradingLibrary; python3 ~/QuantumTradingLibrary/06_Integration/HybridBridge/etare_signal_generator.py"

pause
