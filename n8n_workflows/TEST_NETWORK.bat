@echo off
title Quantum Network Test
cd /d "%~dp0"
powershell -ExecutionPolicy Bypass -NoProfile -File "%~dp0automation\Test-QuantumNetwork.ps1"
