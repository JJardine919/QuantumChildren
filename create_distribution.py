#!/usr/bin/env python3
"""
QUANTUM CHILDREN - Distribution Package Creator
================================================
Creates a .zip file for distribution at quantum-children.com

Excludes:
- .env files (real credentials)
- __pycache__ directories
- .git directory
- *.pyc files
- Log files
- Compiled MQL5 files (.ex5)
"""

import os
import zipfile
import shutil
from pathlib import Path
from datetime import datetime

# Configuration
PROJECT_ROOT = Path(__file__).parent
DIST_NAME = f"QuantumChildren_v2.0_{datetime.now().strftime('%Y%m%d')}"
OUTPUT_DIR = PROJECT_ROOT / "dist"
OUTPUT_ZIP = OUTPUT_DIR / f"{DIST_NAME}.zip"

# Files/directories to exclude
EXCLUDE_PATTERNS = [
    '.env',              # Real credentials (but include .env.example)
    '__pycache__',
    '.git',
    '.gitignore',        # Users don't need this
    '*.pyc',
    '*.pyo',
    '*.log',
    '*.ex5',             # Compiled MQL5
    '*.ex4',
    'dist/',             # Don't include dist folder itself
    'create_distribution.py',  # This script
    '.claude/',
    'node_modules/',
    '*.tmp',
    '*.bak',
    'Thumbs.db',
    '.DS_Store',
    'brain_*.log',       # Trading logs
    '*.output',          # Task output files
]

# Files to always include even if they match exclude patterns
ALWAYS_INCLUDE = [
    '.env.example',
    '.gitignore',  # Actually include this for users
]

def should_exclude(path: Path, root: Path) -> bool:
    """Check if a path should be excluded from distribution."""
    rel_path = str(path.relative_to(root))
    name = path.name

    # Always include these
    if name in ALWAYS_INCLUDE:
        return False

    # Check exclusion patterns
    for pattern in EXCLUDE_PATTERNS:
        if pattern.startswith('*'):
            # Wildcard pattern
            if name.endswith(pattern[1:]):
                return True
        elif pattern.endswith('/'):
            # Directory pattern
            if pattern[:-1] in rel_path.split(os.sep):
                return True
        else:
            # Exact match
            if name == pattern or pattern in rel_path:
                return True

    # Exclude .env but not .env.example
    if name == '.env' and not name.endswith('.example'):
        return True

    return False

def create_distribution():
    """Create the distribution zip file."""
    print("=" * 60)
    print("  QUANTUM CHILDREN - Distribution Package Creator")
    print("=" * 60)
    print()

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Remove old zip if exists
    if OUTPUT_ZIP.exists():
        OUTPUT_ZIP.unlink()
        print(f"  Removed old: {OUTPUT_ZIP.name}")

    print(f"  Creating: {OUTPUT_ZIP.name}")
    print()

    file_count = 0
    total_size = 0

    with zipfile.ZipFile(OUTPUT_ZIP, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(PROJECT_ROOT):
            root_path = Path(root)

            # Filter directories in-place to skip excluded ones
            dirs[:] = [d for d in dirs if not should_exclude(root_path / d, PROJECT_ROOT)]

            for file in files:
                file_path = root_path / file

                if should_exclude(file_path, PROJECT_ROOT):
                    continue

                # Calculate archive path (relative to project root)
                arc_path = file_path.relative_to(PROJECT_ROOT)

                # Add to zip with QuantumChildren prefix
                arc_name = f"QuantumChildren/{arc_path}"
                zf.write(file_path, arc_name)

                file_count += 1
                total_size += file_path.stat().st_size

    # Get final zip size
    zip_size = OUTPUT_ZIP.stat().st_size

    print(f"  Files included: {file_count}")
    print(f"  Original size:  {total_size / 1024 / 1024:.1f} MB")
    print(f"  Compressed:     {zip_size / 1024 / 1024:.1f} MB")
    print(f"  Compression:    {(1 - zip_size/total_size) * 100:.0f}%")
    print()
    print(f"  Output: {OUTPUT_ZIP}")
    print()
    print("=" * 60)
    print("  Distribution package created successfully!")
    print("=" * 60)

    return OUTPUT_ZIP

def verify_no_credentials(zip_path: Path):
    """Verify no real credentials are in the zip.

    Checks for .env files and common credential patterns.
    Actual passwords are loaded from .env at runtime - never stored in code.
    """
    print()
    print("  Verifying no credentials leaked...")

    found_issues = []

    with zipfile.ZipFile(zip_path, 'r') as zf:
        for name in zf.namelist():
            if name.endswith('.example'):
                continue

            # Check for .env files that slipped through
            basename = name.split('/')[-1]
            if basename == '.env':
                found_issues.append((name, '.env file included'))
                continue

            # Check for common credential patterns in file content
            try:
                content = zf.read(name)
                # Look for password assignment patterns (not imports)
                if b'password' in content.lower() if isinstance(content, bytes) else False:
                    lines = content.decode('utf-8', errors='ignore').split('\n')
                    for line_num, line in enumerate(lines, 1):
                        stripped = line.strip()
                        # Skip comments, imports, and dict key access
                        if stripped.startswith('#') or stripped.startswith('//'):
                            continue
                        if 'import' in stripped or "['password']" in stripped:
                            continue
                        if 'password=' in stripped.replace(' ', '') and not stripped.startswith('password='):
                            # Looks like hardcoded password assignment
                            if 'get_credentials' not in stripped and 'creds' not in stripped:
                                found_issues.append((name, f"line {line_num}: possible hardcoded password"))
            except:
                pass

    if found_issues:
        print("  [WARNING] Potential credential leaks found:")
        for name, issue in found_issues:
            print(f"    - {name}: {issue}")
        return False
    else:
        print("  [OK] No credentials found in distribution")
        return True

if __name__ == "__main__":
    zip_path = create_distribution()
    verify_no_credentials(zip_path)
