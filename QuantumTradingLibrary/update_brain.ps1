# Update Brain Script
$VPS_HOST = "72.62.170.153"
$VPS_USER = "root"

Write-Host "Uploading fixed signal generator..."
scp 06_Integration/HybridBridge/etare_signal_generator_redux.py ${VPS_USER}@${VPS_HOST}:/root/quantum-brain/06_Integration/HybridBridge/

Write-Host "Rebuilding and restarting Brain..."
ssh ${VPS_USER}@${VPS_HOST} "cd /root/quantum-brain && docker build -f Dockerfile.redux -t quantum-brain-redux . && docker restart quantum-brain"

Write-Host "Done! Brain should be active."