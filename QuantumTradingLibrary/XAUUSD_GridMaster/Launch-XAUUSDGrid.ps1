#Requires -Version 5.1
<#
.SYNOPSIS
    XAUUSD Grid Trading System - PowerShell Launcher
    GetLeveraged Multi-Account Edition

.DESCRIPTION
    Deploys and manages the XAUUSD Grid Trading EA across all 3 GetLeveraged accounts.

    Features:
    - Automatic deployment to all detected MT5 terminals
    - LLM companion management
    - Status monitoring
    - Process management

.PARAMETER Deploy
    Deploy MQL5 files to all terminals

.PARAMETER LaunchLLM
    Launch the LLM companion script

.PARAMETER Monitor
    Monitor running processes and signal files

.PARAMETER StopLLM
    Stop the running LLM companion

.EXAMPLE
    .\Launch-XAUUSDGrid.ps1 -Deploy -LaunchLLM

.EXAMPLE
    .\Launch-XAUUSDGrid.ps1 -Monitor

.NOTES
    Author: DooDoo - Quantum Trading Library
    Accounts: 113328, 113326, 107245 @ GetLeveraged-Trade
#>

[CmdletBinding()]
param(
    [switch]$Deploy,
    [switch]$LaunchLLM,
    [switch]$Monitor,
    [switch]$StopLLM,
    [switch]$All
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Colors
function Write-Header($text) {
    Write-Host "`n$("=" * 60)" -ForegroundColor Cyan
    Write-Host "  $text" -ForegroundColor Yellow
    Write-Host "$("=" * 60)" -ForegroundColor Cyan
}

function Write-Success($text) {
    Write-Host "[OK] $text" -ForegroundColor Green
}

function Write-Info($text) {
    Write-Host "[INFO] $text" -ForegroundColor White
}

function Write-Warn($text) {
    Write-Host "[WARN] $text" -ForegroundColor Yellow
}

function Write-Err($text) {
    Write-Host "[ERROR] $text" -ForegroundColor Red
}

# Banner
Write-Host @"

  __  __    _   _   _ _   _ ___  ___    ____      _     _
  \ \/ /   / \ | | | | | | / __||   \  / ___|_ __(_) __| |
   \  /   / _ \| | | | | | \__ \| |) || |  _| '__| |/ _` |
   /  \  / ___ \ |_| | |_| |___/|___/ | |_| | |  | | (_| |
  /_/\_\/_/   \_\___/ \___/|____/      \____|_|  |_|\__,_|

  GetLeveraged Multi-Account Edition
  Accounts: 113328, 113326, 107245

"@ -ForegroundColor Cyan

# If no parameters, show help
if (-not ($Deploy -or $LaunchLLM -or $Monitor -or $StopLLM -or $All)) {
    Write-Host "Usage: .\Launch-XAUUSDGrid.ps1 [-Deploy] [-LaunchLLM] [-Monitor] [-StopLLM] [-All]" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Parameters:"
    Write-Host "  -Deploy     Deploy MQL5 files to all terminals"
    Write-Host "  -LaunchLLM  Launch the LLM companion script"
    Write-Host "  -Monitor    Monitor running processes and signal files"
    Write-Host "  -StopLLM    Stop the running LLM companion"
    Write-Host "  -All        Deploy and launch everything"
    Write-Host ""
    exit
}

if ($All) {
    $Deploy = $true
    $LaunchLLM = $true
}

# Deploy MQL5 files
if ($Deploy) {
    Write-Header "DEPLOYING XAUUSD GRID SYSTEM"

    $deployScript = Join-Path $ScriptDir "deploy_xauusd_grid.py"

    if (-not (Test-Path $deployScript)) {
        Write-Err "Deployment script not found: $deployScript"
        exit 1
    }

    Write-Info "Running deployment script..."
    & python $deployScript

    if ($LASTEXITCODE -eq 0) {
        Write-Success "Deployment completed"
    } else {
        Write-Warn "Deployment completed with warnings"
    }
}

# Launch LLM Companion
if ($LaunchLLM) {
    Write-Header "LAUNCHING LLM COMPANION"

    $llmScript = Join-Path $ScriptDir "xauusd_llm_companion.py"

    if (-not (Test-Path $llmScript)) {
        Write-Err "LLM companion script not found: $llmScript"
        exit 1
    }

    # Check if already running
    $existing = Get-Process -Name "python*" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like "*xauusd_llm_companion*" }

    if ($existing) {
        Write-Warn "LLM Companion is already running (PID: $($existing.Id))"
        $response = Read-Host "Restart? (y/n)"
        if ($response -eq 'y') {
            Write-Info "Stopping existing process..."
            $existing | Stop-Process -Force
            Start-Sleep -Seconds 2
        } else {
            Write-Info "Keeping existing process"
            $LaunchLLM = $false
        }
    }

    if ($LaunchLLM) {
        Write-Info "Launching LLM Companion..."

        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName = "python"
        $psi.Arguments = "`"$llmScript`" --account 113328"
        $psi.WorkingDirectory = $ScriptDir
        $psi.UseShellExecute = $true
        $psi.WindowStyle = "Normal"

        $process = [System.Diagnostics.Process]::Start($psi)

        Write-Success "LLM Companion launched (PID: $($process.Id))"
    }
}

# Stop LLM Companion
if ($StopLLM) {
    Write-Header "STOPPING LLM COMPANION"

    $processes = Get-Process -Name "python*" -ErrorAction SilentlyContinue |
        Where-Object {
            try { $_.CommandLine -like "*xauusd_llm_companion*" } catch { $false }
        }

    if ($processes) {
        foreach ($proc in $processes) {
            Write-Info "Stopping process $($proc.Id)..."
            Stop-Process -Id $proc.Id -Force
        }
        Write-Success "LLM Companion stopped"
    } else {
        Write-Info "No LLM Companion processes found"
    }
}

# Monitor
if ($Monitor) {
    Write-Header "SYSTEM MONITOR"

    # Check Python processes
    Write-Info "Checking Python processes..."
    $pythonProcs = Get-Process -Name "python*" -ErrorAction SilentlyContinue
    if ($pythonProcs) {
        foreach ($proc in $pythonProcs) {
            try {
                $cmdLine = (Get-CimInstance Win32_Process -Filter "ProcessId = $($proc.Id)").CommandLine
                if ($cmdLine -like "*xauusd*") {
                    Write-Success "LLM Companion running: PID $($proc.Id)"
                }
            } catch { }
        }
    } else {
        Write-Warn "No Python processes found"
    }

    # Check signal file
    $signalFile = Join-Path $env:APPDATA "MetaQuotes\Terminal\Common\Files\xauusd_llm_signal.txt"
    Write-Info "Checking signal file..."

    if (Test-Path $signalFile) {
        $fileInfo = Get-Item $signalFile
        $age = (Get-Date) - $fileInfo.LastWriteTime

        if ($age.TotalMinutes -lt 2) {
            Write-Success "Signal file is fresh (updated $([int]$age.TotalSeconds)s ago)"
        } else {
            Write-Warn "Signal file is stale (last update: $($fileInfo.LastWriteTime))"
        }

        Write-Host "`nSignal file contents:" -ForegroundColor Cyan
        Get-Content $signalFile | Select-Object -First 15 | ForEach-Object {
            Write-Host "  $_" -ForegroundColor Gray
        }
    } else {
        Write-Warn "Signal file not found: $signalFile"
    }

    # Check MT5 processes
    Write-Info "`nChecking MT5 terminals..."
    $mt5Procs = Get-Process -Name "terminal64" -ErrorAction SilentlyContinue
    if ($mt5Procs) {
        Write-Success "Found $($mt5Procs.Count) MT5 terminal(s) running"
    } else {
        Write-Warn "No MT5 terminals running"
    }
}

Write-Host "`n" -NoNewline
