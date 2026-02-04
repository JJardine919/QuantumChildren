@echo off
set VPS_IP=72.62.170.153
set VPS_USER=root

echo ========================================================
echo  QUANTUM DOCKER DEPLOYMENT (FORCED CLEAN INSTALL)
echo ========================================================

echo.
echo [1/4] Reinstalling Docker Service...
ssh %VPS_USER%@%VPS_IP% "apt-get remove -y docker docker-engine docker.io containerd runc; curl -fsSL https://get.docker.com -o get-docker.sh; sh get-docker.sh; service docker start"

echo.
echo [2/4] Uploading Brain code...
ssh %VPS_USER%@%VPS_IP% "mkdir -p ~/QuantumDocker"
scp Dockerfile requirements.txt %VPS_USER%@%VPS_IP%:~/QuantumDocker/
scp -r 01_Systems 06_Integration %VPS_USER%@%VPS_IP%:~/QuantumDocker/

echo.
echo [3/4] Building the Brain Container...
ssh %VPS_USER%@%VPS_IP% "cd ~/QuantumDocker && docker build -t quantum-brain ."

echo.
echo [4/4] STARTING THE BRAIN...
ssh -t %VPS_USER%@%VPS_IP% "docker stop quantum-brain 2>/dev/null; docker rm quantum-brain 2>/dev/null; docker run --name quantum-brain --restart unless-stopped -v /root/.wine/drive_c/Program\ Files/MetaTrader\ 5/MQL5/Files:/root/.wine/drive_c/Program\ Files/MetaTrader\ 5/MQL5/Files quantum-brain"

pause
