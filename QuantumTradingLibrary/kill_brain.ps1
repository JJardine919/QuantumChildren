Get-CimInstance Win32_Process -Filter "name='python.exe'" | ForEach-Object {
    if ($_.CommandLine -like '*BRAIN_GETLEVERAGED*') {
        Stop-Process -Id $_.ProcessId -Force
        Write-Output "Killed PID $($_.ProcessId)"
    }
}
