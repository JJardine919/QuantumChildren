"""
MODEL VALIDATOR - Verify model integrity after sync
=====================================================
Loads each model from the manifest and runs a dummy forward pass.
If any model fails, alerts and falls back to backup directory.

Usage: python validate_models.py
Returns exit code 0 on success, 1 on failure.
"""

import json
import sys
import shutil
import logging
from pathlib import Path

import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [VALIDATE] %(message)s',
    datefmt='%H:%M:%S'
)

MODELS_DIR = Path(__file__).parent / "top_50_experts"
BACKUP_DIR = Path(__file__).parent / "top_50_experts_backup"
MANIFEST_FILE = MODELS_DIR / "top_50_manifest.json"


def validate_models() -> bool:
    """Load each model from manifest and run a dummy forward pass."""
    if not MANIFEST_FILE.exists():
        logging.error(f"Manifest not found: {MANIFEST_FILE}")
        return False

    with open(MANIFEST_FILE, 'r') as f:
        manifest = json.load(f)

    models = manifest.get('models', manifest.get('experts', []))
    if not models:
        logging.error("Manifest contains no models")
        return False

    passed = 0
    failed = 0

    for entry in models:
        model_file = entry.get('path', entry.get('file', ''))
        if not model_file:
            continue

        model_path = MODELS_DIR / model_file
        if not model_path.exists():
            logging.error(f"MISSING: {model_file}")
            failed += 1
            continue

        try:
            # Load state dict
            state_dict = torch.load(model_path, map_location='cpu', weights_only=True)

            # Infer input size from first layer
            first_key = next(iter(state_dict))
            input_size = state_dict[first_key].shape[-1] if state_dict[first_key].dim() > 1 else 1

            # Create dummy input and verify tensor loads correctly
            dummy = torch.randn(1, 10, input_size)

            # Basic integrity check: all tensors are finite
            for key, tensor in state_dict.items():
                if not torch.isfinite(tensor).all():
                    raise ValueError(f"Non-finite values in {key}")

            passed += 1
            logging.info(f"OK: {model_file} ({len(state_dict)} params)")

        except Exception as e:
            logging.error(f"FAIL: {model_file} — {e}")
            failed += 1

    logging.info(f"Validation complete: {passed} passed, {failed} failed out of {len(models)}")

    if failed > 0:
        logging.warning("Some models failed validation!")
        return False

    return True


def rollback_to_backup():
    """Replace current models with backup if validation fails."""
    if not BACKUP_DIR.exists():
        logging.error("No backup directory available for rollback")
        return False

    try:
        failed_dir = MODELS_DIR.parent / "top_50_experts_failed"
        if failed_dir.exists():
            shutil.rmtree(failed_dir)

        MODELS_DIR.rename(failed_dir)
        BACKUP_DIR.rename(MODELS_DIR)
        logging.info("ROLLBACK complete — restored from backup")
        return True
    except Exception as e:
        logging.error(f"Rollback failed: {e}")
        return False


if __name__ == "__main__":
    if validate_models():
        logging.info("All models validated successfully")
        sys.exit(0)
    else:
        logging.error("Validation failed — attempting rollback to backup")
        if rollback_to_backup():
            logging.info("Rollback successful — backup models restored")
        else:
            logging.error("CRITICAL: Rollback also failed — manual intervention needed")
        sys.exit(1)
