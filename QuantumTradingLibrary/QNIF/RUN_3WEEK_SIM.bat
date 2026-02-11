@echo off
REM QNIF 3-Week Simulation Launcher
REM Built by Biskits
REM
REM Usage:
REM   RUN_3WEEK_SIM.bat           - Default run (MT5 data, ATLAS account)
REM   RUN_3WEEK_SIM.bat csv       - CSV mode (offline)
REM   RUN_3WEEK_SIM.bat quick     - Quick test (7 days, 3 accounts)
REM   RUN_3WEEK_SIM.bat full      - Full run (55 accounts)

echo ================================================================================
echo   QNIF 3-WEEK BACKTEST SIMULATION
echo   Built by Biskits - 20 years of building complicated systems
echo ================================================================================
echo.

set PYTHON_GPU=..\..\.venv312_gpu\Scripts\python.exe
set PYTHON_SYS=python

REM Check if GPU venv exists
if exist "%PYTHON_GPU%" (
    set PYTHON_EXE=%PYTHON_GPU%
    echo Using GPU Python: %PYTHON_GPU%
) else (
    set PYTHON_EXE=%PYTHON_SYS%
    echo Using System Python: %PYTHON_SYS%
)

echo.

REM Parse command line arguments
if "%1"=="csv" (
    echo Mode: CSV Offline
    %PYTHON_EXE% qnif_3week_sim.py --csv
    goto :end
)

if "%1"=="quick" (
    echo Mode: Quick Test (7 days, 3 accounts)
    %PYTHON_EXE% qnif_3week_sim.py --days 7 --accounts FARM_01 FARM_02 FARM_15
    goto :end
)

if "%1"=="full" (
    echo Mode: Full Run (55 accounts)
    %PYTHON_EXE% qnif_3week_sim.py --accounts FARM_01 FARM_02 FARM_03 FARM_04 FARM_05 FARM_06 FARM_07 FARM_08 FARM_09 FARM_10 FARM_11 FARM_12 FARM_13 FARM_14 FARM_15 FARM_16 FARM_17 FARM_18 FARM_19 FARM_20 FARM_21 FARM_22 FARM_23 FARM_24 FARM_25 FARM_26 FARM_27 FARM_28 FARM_29 FARM_30 FARM_31 FARM_32 FARM_33 FARM_34 FARM_35 FARM_36 FARM_37 FARM_38 FARM_39 FARM_40 FARM_41 FARM_42 FARM_43 FARM_44 FARM_45 FARM_46 FARM_47 FARM_48 FARM_49 FARM_50 FARM_51 FARM_52 FARM_53 FARM_54 FARM_55
    goto :end
)

REM Default run
echo Mode: Default (MT5 data, ATLAS account, 10 accounts)
%PYTHON_EXE% qnif_3week_sim.py --account ATLAS

:end
echo.
echo ================================================================================
echo   Simulation Complete
echo   Results saved to sim_results/
echo ================================================================================
pause
