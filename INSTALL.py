#!/usr/bin/env python3
"""
QUANTUM CHILDREN - INSTALLER
============================
Automated setup script for the Quantum Children trading system.

Run: python INSTALL.py
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

REQUIRED_PYTHON = (3, 9)  # Minimum Python version
PROJECT_ROOT = Path(__file__).parent
QTL_PATH = PROJECT_ROOT / "QuantumTradingLibrary"

REQUIRED_PACKAGES = [
    "numpy",
    "pandas",
    "MetaTrader5",
    "python-dotenv",
    "requests",
]

OPTIONAL_PACKAGES = [
    ("torch", "PyTorch for ML features"),
    ("catboost", "CatBoost for expert training"),
    ("scikit-learn", "Scikit-learn for ML utilities"),
    ("qiskit", "Qiskit for quantum computing features"),
]

# ============================================================
# UTILITIES
# ============================================================

def print_header(text):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)

def print_step(text):
    print(f"\n>> {text}")

def print_ok(text):
    print(f"   [OK] {text}")

def print_warn(text):
    print(f"   [!!] {text}")

def print_fail(text):
    print(f"   [XX] {text}")

def run_command(cmd, check=True):
    """Run a shell command and return success status."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            check=check
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr

# ============================================================
# CHECKS
# ============================================================

def check_python_version():
    """Verify Python version meets requirements."""
    print_step("Checking Python version...")

    major, minor = sys.version_info[:2]
    required = f"{REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}"

    if (major, minor) >= REQUIRED_PYTHON:
        print_ok(f"Python {major}.{minor} (required: {required}+)")
        return True
    else:
        print_fail(f"Python {major}.{minor} is too old (required: {required}+)")
        print("   Please install Python 3.9 or newer from python.org")
        return False

def check_pip():
    """Verify pip is available."""
    print_step("Checking pip...")

    success, _ = run_command(f"{sys.executable} -m pip --version", check=False)
    if success:
        print_ok("pip is available")
        return True
    else:
        print_fail("pip is not available")
        return False

def check_mt5_installation():
    """Check if MetaTrader 5 terminal is installed."""
    print_step("Checking MetaTrader 5 installation...")

    # Common MT5 installation paths on Windows
    mt5_paths = [
        Path(r"C:\Program Files\Blue Guardian MT5 Terminal"),
        Path(r"C:\Program Files\Atlas Funded MT5 Terminal"),
        Path(r"C:\Program Files\MetaTrader 5"),
        Path(os.environ.get("LOCALAPPDATA", "") + r"\Programs\MetaTrader 5"),
    ]

    found = []
    for path in mt5_paths:
        if path.exists():
            terminal = path / "terminal64.exe"
            if terminal.exists():
                found.append(path)

    if found:
        for p in found:
            print_ok(f"Found: {p}")
        return True
    else:
        print_warn("No MetaTrader 5 installation found in common locations")
        print("   You can still proceed - just ensure MT5 is installed")
        return True  # Warning only, not blocking

# ============================================================
# INSTALLATION
# ============================================================

def install_required_packages():
    """Install required Python packages."""
    print_step("Installing required packages...")

    for package in REQUIRED_PACKAGES:
        print(f"   Installing {package}...", end=" ", flush=True)
        success, _ = run_command(
            f"{sys.executable} -m pip install {package} --quiet",
            check=False
        )
        if success:
            print("[OK]")
        else:
            print("[FAILED]")
            print_fail(f"Failed to install {package}")
            return False

    return True

def install_optional_packages():
    """Offer to install optional packages."""
    print_step("Optional packages:")

    for package, description in OPTIONAL_PACKAGES:
        print(f"   - {package}: {description}")

    print()
    response = input("   Install optional packages? [y/N]: ").strip().lower()

    if response == 'y':
        for package, _ in OPTIONAL_PACKAGES:
            print(f"   Installing {package}...", end=" ", flush=True)
            success, _ = run_command(
                f"{sys.executable} -m pip install {package} --quiet",
                check=False
            )
            if success:
                print("[OK]")
            else:
                print("[SKIP]")
    else:
        print_ok("Skipping optional packages")

    return True

# ============================================================
# CONFIGURATION
# ============================================================

def setup_credentials():
    """Guide user through credential setup."""
    print_step("Setting up credentials...")

    env_file = QTL_PATH / ".env"
    env_example = QTL_PATH / ".env.example"

    if env_file.exists():
        print_ok(".env file already exists")
        response = input("   Overwrite with new credentials? [y/N]: ").strip().lower()
        if response != 'y':
            return True

    if not env_example.exists():
        print_fail(".env.example template not found")
        return False

    print()
    print("   Enter your MT5 account passwords (or press Enter to skip):")
    print()

    accounts = [
        ("BG_INSTANT_PASSWORD", "BlueGuardian Instant (366604)"),
        ("BG_CHALLENGE_PASSWORD", "BlueGuardian Challenge (365060)"),
        ("ATLAS_PASSWORD", "Atlas Funded (212000584)"),
        ("GL_1_PASSWORD", "GetLeveraged #1 (113326)"),
        ("GL_2_PASSWORD", "GetLeveraged #2 (113328)"),
        ("GL_3_PASSWORD", "GetLeveraged #3 (107245)"),
    ]

    credentials = {}
    for env_key, description in accounts:
        password = input(f"   {description}: ").strip()
        if password:
            credentials[env_key] = password

    if not credentials:
        print_warn("No credentials entered - you can add them later to .env")
        # Copy template
        shutil.copy(env_example, env_file)
        return True

    # Write .env file
    lines = [
        "# QUANTUM CHILDREN - CREDENTIALS (PRIVATE)",
        "# Generated by INSTALL.py",
        ""
    ]
    for env_key, _ in accounts:
        if env_key in credentials:
            lines.append(f"{env_key}={credentials[env_key]}")
        else:
            lines.append(f"# {env_key}=")

    with open(env_file, 'w') as f:
        f.write("\n".join(lines))

    print_ok(f"Credentials saved to {env_file}")
    return True

def verify_installation():
    """Verify the installation is working."""
    print_step("Verifying installation...")

    # Test imports
    tests = [
        ("numpy", "import numpy"),
        ("pandas", "import pandas"),
        ("MetaTrader5", "import MetaTrader5"),
        ("dotenv", "import dotenv"),
        ("config_loader", f"import sys; sys.path.insert(0, r'{QTL_PATH}'); import config_loader"),
        ("credential_manager", f"import sys; sys.path.insert(0, r'{QTL_PATH}'); import credential_manager"),
    ]

    all_passed = True
    for name, test_code in tests:
        success, _ = run_command(
            f'{sys.executable} -c "{test_code}"',
            check=False
        )
        if success:
            print_ok(f"{name}")
        else:
            print_fail(f"{name}")
            all_passed = False

    return all_passed

# ============================================================
# MAIN
# ============================================================

def show_legal():
    """Display legal disclaimers."""
    print_header("LEGAL NOTICES")

    disclaimer = PROJECT_ROOT / "RISK_DISCLAIMER.md"
    if disclaimer.exists():
        print()
        print("   IMPORTANT: By using Quantum Children, you agree to:")
        print("   - RISK_DISCLAIMER.md - Trading risk warnings")
        print("   - TERMS_OF_SERVICE.md - Usage terms")
        print("   - PRIVACY_POLICY.md - Data collection policy")
        print()
        print("   Please read these documents before trading live.")
        print()

    response = input("   I have read and accept the terms [y/N]: ").strip().lower()
    return response == 'y'

def show_next_steps():
    """Display post-installation instructions."""
    print_header("INSTALLATION COMPLETE")

    print("""
   Next steps:

   1. CONFIGURE CREDENTIALS (if not done):
      Edit: QuantumTradingLibrary/.env

   2. VERIFY MT5 CONNECTION:
      python -c "import MetaTrader5 as mt5; print(mt5.initialize())"

   3. TEST CREDENTIAL LOADING:
      cd QuantumTradingLibrary
      python credential_manager.py

   4. READ THE DOCUMENTATION:
      - DISTRIBUTION/README.md (Quick Start)
      - RISK_DISCLAIMER.md (Important!)

   5. START TRADING (Paper first!):
      cd QuantumTradingLibrary
      python BRAIN_ATLAS.py

   For help, see: quantum-children.com
""")

def main():
    print_header("QUANTUM CHILDREN INSTALLER")
    print("   AI-Driven Trading System")
    print("   Version 2.0")

    # Pre-flight checks
    if not check_python_version():
        return 1

    if not check_pip():
        return 1

    check_mt5_installation()

    # Legal acceptance
    if not show_legal():
        print_fail("You must accept the terms to continue")
        return 1

    # Installation
    print_header("INSTALLING DEPENDENCIES")

    if not install_required_packages():
        return 1

    install_optional_packages()

    # Configuration
    print_header("CONFIGURATION")

    setup_credentials()

    # Verification
    print_header("VERIFICATION")

    if not verify_installation():
        print_warn("Some components failed verification")
        print("   The system may still work - check the errors above")

    # Done
    show_next_steps()

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n   Installation cancelled.")
        sys.exit(1)
    except Exception as e:
        print(f"\n   Error: {e}")
        sys.exit(1)
