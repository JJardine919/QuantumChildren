@echo off
title Stop Quantum Network
cd /d "%~dp0"
powershell -ExecutionPolicy Bypass -NoProfile -File "%~dp0automation\Stop-QuantumNetwork.ps1"
