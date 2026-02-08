@echo off
:: Simple starter - just opens n8n with instructions
title Quantum Network Simple Start
cd /d "%~dp0"
powershell -ExecutionPolicy Bypass -NoProfile -File "%~dp0automation\Start-N8N-Simple.ps1"
