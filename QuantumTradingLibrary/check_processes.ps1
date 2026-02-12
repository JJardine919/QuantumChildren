Get-WmiObject Win32_Process -Filter "name='python.exe'" | ForEach-Object {
    $cmdLine = $_.CommandLine
    if ($cmdLine -match 'BRAIN|WATCH|quantum_server|MASTER_LAUNCH') {
        Write-Host "PID: $($_.ProcessId) | $cmdLine"
    }
}
