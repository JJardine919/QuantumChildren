"""
XAUUSD Grid Trading System - Deployment Script
================================================
Deploys the XAUUSD Grid Trading EA to all 3 GetLeveraged accounts.

Features:
- Copies MQL5 files to correct terminal folders
- Generates account-specific preset files
- Launches the LLM companion script
- Provides status monitoring

Usage:
    python deploy_xauusd_grid.py [--compile] [--launch-llm]

Author: DooDoo - Quantum Trading Library
"""

import os
import sys
import json
import shutil
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

# Configuration
SCRIPT_DIR = Path(__file__).parent
CONFIG_FILE = SCRIPT_DIR / "getleveraged_accounts.json"

# MQL5 source files
MQL5_FILES = [
    "XAUUSD_GridMaster.mq5",
    "XAUUSD_GridCore.mqh"
]

# Python files
PYTHON_FILES = [
    "xauusd_llm_companion.py"
]

# Common MT5 data paths (Windows)
MT5_COMMON_PATH = Path(os.path.expandvars(r"%APPDATA%\MetaQuotes\Terminal\Common\Files"))


def load_config() -> dict:
    """Load configuration from JSON file"""
    if not CONFIG_FILE.exists():
        print(f"ERROR: Config file not found: {CONFIG_FILE}")
        sys.exit(1)

    with open(CONFIG_FILE, 'r') as f:
        return json.load(f)


def find_mt5_terminals() -> list:
    """Find all MT5 terminal data folders"""
    appdata = Path(os.path.expandvars(r"%APPDATA%\MetaQuotes\Terminal"))

    if not appdata.exists():
        print("WARNING: MetaQuotes Terminal folder not found")
        return []

    terminals = []
    for folder in appdata.iterdir():
        if folder.is_dir() and len(folder.name) == 32:  # Terminal ID is 32 chars
            mql5_path = folder / "MQL5"
            if mql5_path.exists():
                # Try to identify the terminal
                terminal_info = {
                    'id': folder.name,
                    'path': folder,
                    'mql5_path': mql5_path,
                    'experts_path': mql5_path / "Experts",
                    'include_path': mql5_path / "Include",
                }

                # Check origin.txt for terminal identification
                origin_file = folder / "origin.txt"
                if origin_file.exists():
                    terminal_info['origin'] = origin_file.read_text().strip()

                terminals.append(terminal_info)

    return terminals


def copy_mql5_files(terminals: list, config: dict):
    """Copy MQL5 files to all terminal folders"""
    print("\n" + "="*60)
    print("COPYING MQL5 FILES")
    print("="*60)

    deploy_config = config.get('deployment', {})
    experts_subfolder = deploy_config.get('copy_to_folders', {}).get('mql5', 'Experts/XAUUSD_GridMaster')
    include_subfolder = deploy_config.get('copy_to_folders', {}).get('include', 'Include')

    for terminal in terminals:
        print(f"\nTerminal: {terminal['id'][:16]}...")

        # Create experts folder
        experts_dest = terminal['mql5_path'] / experts_subfolder.replace('MQL5/', '')
        experts_dest.mkdir(parents=True, exist_ok=True)

        # Create include folder
        include_dest = terminal['mql5_path'] / include_subfolder.replace('MQL5/', '')
        include_dest.mkdir(parents=True, exist_ok=True)

        # Copy EA file
        for filename in MQL5_FILES:
            src = SCRIPT_DIR / filename
            if src.exists():
                if filename.endswith('.mq5'):
                    dest = experts_dest / filename
                else:  # .mqh files go to same folder as EA
                    dest = experts_dest / filename

                shutil.copy2(src, dest)
                print(f"  Copied: {filename} -> {dest}")
            else:
                print(f"  WARNING: Source file not found: {src}")


def generate_preset_files(terminals: list, config: dict):
    """Generate account-specific preset (.set) files"""
    print("\n" + "="*60)
    print("GENERATING PRESET FILES")
    print("="*60)

    accounts = config.get('accounts', [])
    system = config.get('system_settings', {})

    deploy_config = config.get('deployment', {})
    experts_subfolder = deploy_config.get('copy_to_folders', {}).get('mql5', 'Experts/XAUUSD_GridMaster')

    for terminal in terminals:
        presets_folder = terminal['mql5_path'] / "Presets" / "XAUUSD_GridMaster"
        presets_folder.mkdir(parents=True, exist_ok=True)

        for account in accounts:
            acc_id = account['id']
            settings = account.get('ea_settings', {})

            preset_content = f"""; XAUUSD GridMaster Preset for {acc_id}
; Generated: {datetime.now().isoformat()}
; Account: {account['account_number']} @ {account['server']}

; === ACCOUNT CONFIGURATION ===
InpAccountName={acc_id}
InpMagicBullish={settings.get('magic_bullish', 100001)}
InpMagicBearish={settings.get('magic_bearish', 100002)}
InpMagicNeutral={settings.get('magic_neutral', 100003)}
InpAccountBalance={settings.get('account_balance', 10000)}
InpDailyDDLimit={settings.get('daily_dd_limit', 5.0)}
InpMaxDDLimit={settings.get('max_dd_limit', 10.0)}

; === RISK MANAGEMENT ===
InpRiskPerGrid={settings.get('risk_per_grid', 0.25)}
InpDynamicHiddenSLTP=true
InpTrailingStop=true
InpBreakEven=true

; === GRID SETTINGS ===
InpMaxOrdersPerExpert={settings.get('max_orders_per_expert', 5)}
InpMaxTotalOrders={settings.get('max_total_orders', 15)}
InpGridSpacingAtr={system.get('grid_spacing_atr', 0.75)}

; === ENTROPY FILTER ===
InpConfidenceThreshold={system.get('base_confidence_threshold', 0.65)}
InpUseEntropyFilter={str(system.get('entropy_filter_enabled', True)).lower()}
InpEntropyThreshold={system.get('max_entropy_threshold', 2.0)}

; === LLM INTEGRATION ===
InpUseLLM={str(config.get('llm_settings', {}).get('enabled', True)).lower()}
InpLLMSignalFile={config.get('llm_settings', {}).get('signal_file', 'xauusd_llm_signal.txt')}

; === TRADING ===
InpTradeEnabled=true
InpCheckInterval={system.get('check_interval_seconds', 30)}
"""

            preset_file = presets_folder / f"{acc_id}.set"
            with open(preset_file, 'w') as f:
                f.write(preset_content)

            print(f"  Created preset: {preset_file.name}")


def compile_mql5(terminals: list):
    """Attempt to compile MQL5 files using MetaEditor"""
    print("\n" + "="*60)
    print("COMPILING MQL5 FILES")
    print("="*60)

    # Find MetaEditor
    metaeditor_paths = [
        Path(r"C:\Program Files\GetLeveraged MT5 Terminal\metaeditor64.exe"),
        Path(r"C:\Program Files (x86)\GetLeveraged MT5 Terminal\metaeditor64.exe"),
    ]

    # Also check in terminal folders
    for terminal in terminals:
        possible = terminal['path'].parent.parent / "metaeditor64.exe"
        if possible.exists():
            metaeditor_paths.insert(0, possible)

    metaeditor = None
    for path in metaeditor_paths:
        if path.exists():
            metaeditor = path
            break

    if not metaeditor:
        print("WARNING: MetaEditor not found - manual compilation required")
        print("Please compile XAUUSD_GridMaster.mq5 in MT5 terminal")
        return

    # Compile in first terminal
    if terminals:
        terminal = terminals[0]
        ea_file = terminal['mql5_path'] / "Experts" / "QuantumChildren" / "XAUUSD_GridMaster" / "XAUUSD_GridMaster.mq5"

        if ea_file.exists():
            print(f"Compiling: {ea_file}")
            cmd = [str(metaeditor), "/compile:" + str(ea_file)]

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                if result.returncode == 0:
                    print("Compilation successful!")
                else:
                    print(f"Compilation may have issues - check MetaEditor")
            except Exception as e:
                print(f"Compilation error: {e}")
        else:
            print(f"EA file not found for compilation: {ea_file}")


def launch_llm_companion(config: dict):
    """Launch the LLM companion script"""
    print("\n" + "="*60)
    print("LAUNCHING LLM COMPANION")
    print("="*60)

    llm_script = SCRIPT_DIR / "xauusd_llm_companion.py"
    if not llm_script.exists():
        print(f"ERROR: LLM companion script not found: {llm_script}")
        return

    llm_settings = config.get('llm_settings', {})
    accounts = config.get('accounts', [])

    # Launch for primary account
    if accounts:
        primary = accounts[0]
        acc_id = str(primary['account_number'])

        cmd = [
            sys.executable,
            str(llm_script),
            "--account", acc_id,
            "--signal-file", llm_settings.get('signal_file', 'xauusd_llm_signal.txt'),
            "--interval", str(llm_settings.get('analysis_interval_seconds', 30))
        ]

        model = llm_settings.get('model')
        if model:
            cmd.extend(["--model", model])

        print(f"Launching: {' '.join(cmd)}")

        # Launch in new process
        try:
            subprocess.Popen(
                cmd,
                creationflags=subprocess.CREATE_NEW_CONSOLE,
                cwd=str(SCRIPT_DIR)
            )
            print("LLM Companion launched in new console")
        except Exception as e:
            print(f"Failed to launch LLM Companion: {e}")


def print_deployment_summary(terminals: list, config: dict):
    """Print deployment summary and next steps"""
    print("\n" + "="*60)
    print("DEPLOYMENT SUMMARY")
    print("="*60)

    print("\nFiles deployed to", len(terminals), "terminal(s)")

    print("\n--- ACCOUNTS ---")
    for account in config.get('accounts', []):
        print(f"  {account['id']}: Account #{account['account_number']} @ {account['server']}")

    print("\n--- SYSTEM SETTINGS (HARD-CODED) ---")
    system = config.get('system_settings', {})
    print(f"  Symbol: {system.get('symbol', 'XAUUSD')}")
    print(f"  ATR SL Multiplier: {system.get('atr_sl_multiplier', 1.5)}x")
    print(f"  ATR TP Multiplier: {system.get('atr_tp_multiplier', 3.0)}x")
    print(f"  Partial TP: {system.get('partial_tp_percent', 50)}%")
    print(f"  Compression Boost: +{system.get('compression_boost', 12)}")

    print("\n--- NEXT STEPS ---")
    print("""
1. Open MT5 Terminal (GetLeveraged)

2. Log into each account:
   - Account 113328
   - Account 113326
   - Account 107245

3. For each account:
   a. Open XAUUSD chart (M5 timeframe)
   b. Attach XAUUSD_GridMaster EA
   c. Load the corresponding preset file
   d. Enable auto-trading

4. Verify LLM Companion is running:
   - Check for xauusd_llm_companion.py process
   - Signal file should update every 30 seconds

5. Monitor:
   - Watch EA logs in Experts tab
   - Check signal file: %APPDATA%\\MetaQuotes\\Terminal\\Common\\Files\\xauusd_llm_signal.txt
""")


def main():
    """Main deployment function"""
    parser = argparse.ArgumentParser(
        description="Deploy XAUUSD Grid Trading System to GetLeveraged accounts"
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Attempt to compile MQL5 files"
    )
    parser.add_argument(
        "--launch-llm",
        action="store_true",
        help="Launch LLM companion script"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )

    args = parser.parse_args()

    print("="*60)
    print("XAUUSD GRID TRADING SYSTEM - DEPLOYMENT")
    print("GetLeveraged Multi-Account Edition")
    print("="*60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Script Directory: {SCRIPT_DIR}")

    # Load configuration
    config = load_config()
    print(f"\nLoaded configuration: {len(config.get('accounts', []))} accounts")

    # Find MT5 terminals
    terminals = find_mt5_terminals()
    print(f"Found {len(terminals)} MT5 terminal(s)")

    if args.dry_run:
        print("\n[DRY RUN MODE - No changes will be made]")
        print_deployment_summary(terminals, config)
        return

    if not terminals:
        print("\nWARNING: No MT5 terminals found!")
        print("Please ensure MetaTrader 5 is installed")

        # Still create files in script directory for manual copy
        print("\nCreating files in script directory for manual deployment...")

    # Copy MQL5 files
    if terminals:
        copy_mql5_files(terminals, config)

    # Generate preset files
    if terminals:
        generate_preset_files(terminals, config)

    # Compile if requested
    if args.compile and terminals:
        compile_mql5(terminals)

    # Launch LLM companion if requested
    if args.launch_llm:
        launch_llm_companion(config)

    # Print summary
    print_deployment_summary(terminals, config)


if __name__ == "__main__":
    main()
