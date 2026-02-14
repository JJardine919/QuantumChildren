"""
STATE PERSISTENCE - Save/Restore ephemeral BRAIN state
=======================================================
Saves runtime state that would otherwise be lost on restart:
  - Loss cooldown timers (per-symbol)
  - VDJ immune memory cells
  - CRISPR-Cas9 spacers
  - Toxoplasma DD accumulator state
  - Seen ticket sets (DD tracking)

Usage:
    from state_persistence import StatePersistence

    sp = StatePersistence(account_key="ATLAS")
    sp.save_state(trader)     # Save before rest or periodically
    sp.restore_state(trader)  # Restore on wake or restart
"""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path

STATE_DIR = Path(__file__).parent / "state_snapshots"
STATE_DIR.mkdir(exist_ok=True)

PERIODIC_SAVE_MINUTES = 30
_STATE_VERSION = 1


class StatePersistence:
    def __init__(self, account_key: str):
        self.account_key = account_key
        self.state_file = STATE_DIR / f"state_{account_key}.json"
        self._last_save = datetime.min

    def save_state(self, trader) -> bool:
        """Save all ephemeral state from an AccountTrader instance."""
        try:
            state = {
                "account_key": self.account_key,
                "saved_at": datetime.utcnow().isoformat(),
                "loss_cooldown": {},
                "dd_seen_tickets": [],
                "starting_balance": getattr(trader, 'starting_balance', 0.0),
            }

            # Loss cooldowns (symbol -> expiry datetime)
            for symbol, expiry in getattr(trader, '_loss_cooldown', {}).items():
                state["loss_cooldown"][symbol] = expiry.isoformat()

            # DD seen tickets
            state["dd_seen_tickets"] = list(getattr(trader, '_dd_seen_tickets', set()))

            # VDJ immune memory cells
            vdj = getattr(trader, 'vdj_engine', None)
            if vdj and hasattr(vdj, 'memory_cells'):
                cells = []
                for cell in vdj.memory_cells:
                    cells.append({
                        "symbol": getattr(cell, 'symbol', ''),
                        "te_combo": getattr(cell, 'te_combo', ''),
                        "weight": getattr(cell, 'weight', 0.0),
                        "wins": getattr(cell, 'wins', 0),
                        "losses": getattr(cell, 'losses', 0),
                        "last_seen": getattr(cell, 'last_seen', ''),
                    })
                state["vdj_memory_cells"] = cells

            # CRISPR spacers
            crispr = getattr(trader, 'crispr_bridge', None)
            if crispr and hasattr(crispr, 'spacers'):
                spacers = []
                for spacer in crispr.spacers:
                    spacers.append({
                        "symbol": getattr(spacer, 'symbol', ''),
                        "te_combo": getattr(spacer, 'te_combo', ''),
                        "direction": getattr(spacer, 'direction', 0),
                        "loss_count": getattr(spacer, 'loss_count', 0),
                        "acquired_at": getattr(spacer, 'acquired_at', ''),
                    })
                state["crispr_spacers"] = spacers

            # Toxoplasma DD state
            toxo = getattr(trader, 'toxoplasma_engine', None)
            if toxo:
                dd_tracker = getattr(toxo, 'dd_tracker', None)
                if dd_tracker:
                    state["toxoplasma_dd"] = {
                        "daily_realized": getattr(dd_tracker, 'daily_realized', 0.0),
                        "daily_peak_equity": getattr(dd_tracker, 'daily_peak_equity', 0.0),
                        "dopamine_multiplier": getattr(dd_tracker, 'dopamine_multiplier', 1.0),
                        "last_reset_date": str(getattr(dd_tracker, 'last_reset_date', '')),
                    }

            # Add version and checksum for integrity verification
            state["_version"] = _STATE_VERSION
            json_bytes = json.dumps(state, indent=2, sort_keys=True).encode('utf-8')
            state["_checksum"] = hashlib.md5(json_bytes).hexdigest()

            # Write atomically (write to temp, then rename)
            temp_file = self.state_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(state, f, indent=2)
            temp_file.replace(self.state_file)

            self._last_save = datetime.utcnow()
            logging.info(f"[{self.account_key}] State saved to {self.state_file.name}")
            return True

        except Exception as e:
            logging.error(f"[{self.account_key}] State save failed: {e}")
            return False

    def restore_state(self, trader) -> bool:
        """Restore ephemeral state into an AccountTrader instance."""
        if not self.state_file.exists():
            logging.info(f"[{self.account_key}] No saved state found — starting fresh")
            return False

        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)

            # Verify version compatibility
            file_version = state.get("_version", 0)
            if file_version != _STATE_VERSION:
                logging.warning(
                    f"[{self.account_key}] State version mismatch: "
                    f"file={file_version}, expected={_STATE_VERSION} -- starting fresh"
                )
                return False

            # Verify checksum integrity
            saved_checksum = state.pop("_checksum", None)
            if saved_checksum is not None:
                verify_bytes = json.dumps(state, indent=2, sort_keys=True).encode('utf-8')
                computed_checksum = hashlib.md5(verify_bytes).hexdigest()
                if computed_checksum != saved_checksum:
                    logging.warning(
                        f"[{self.account_key}] State checksum mismatch: "
                        f"file may be corrupt -- starting fresh"
                    )
                    return False

            restored_items = []

            # Loss cooldowns
            cooldowns = state.get("loss_cooldown", {})
            for symbol, expiry_str in cooldowns.items():
                expiry = datetime.fromisoformat(expiry_str)
                if expiry > datetime.utcnow():
                    trader._loss_cooldown[symbol] = expiry
                    restored_items.append(f"cooldown:{symbol}")

            # DD seen tickets
            tickets = state.get("dd_seen_tickets", [])
            if tickets:
                trader._dd_seen_tickets = set(tickets)
                restored_items.append(f"tickets:{len(tickets)}")

            # VDJ memory cells
            vdj_cells = state.get("vdj_memory_cells", [])
            vdj = getattr(trader, 'vdj_engine', None)
            if vdj and vdj_cells and hasattr(vdj, 'restore_cells'):
                vdj.restore_cells(vdj_cells)
                restored_items.append(f"vdj:{len(vdj_cells)}cells")

            # CRISPR spacers
            spacers = state.get("crispr_spacers", [])
            crispr = getattr(trader, 'crispr_bridge', None)
            if crispr and spacers and hasattr(crispr, 'restore_spacers'):
                crispr.restore_spacers(spacers)
                restored_items.append(f"crispr:{len(spacers)}spacers")

            # Toxoplasma DD state
            toxo_dd = state.get("toxoplasma_dd")
            toxo = getattr(trader, 'toxoplasma_engine', None)
            if toxo and toxo_dd:
                dd_tracker = getattr(toxo, 'dd_tracker', None)
                if dd_tracker and hasattr(dd_tracker, 'restore_state'):
                    dd_tracker.restore_state(toxo_dd)
                    restored_items.append("toxo_dd")

            saved_at = state.get("saved_at", "unknown")
            logging.info(
                f"[{self.account_key}] State restored from {saved_at}: "
                f"{', '.join(restored_items) if restored_items else 'empty'}"
            )
            return True

        except Exception as e:
            logging.error(f"[{self.account_key}] State restore failed: {e}")
            return False

    def should_periodic_save(self) -> bool:
        """Check if it's time for a periodic save (every 30 min)."""
        elapsed = (datetime.utcnow() - self._last_save).total_seconds() / 60
        return elapsed >= PERIODIC_SAVE_MINUTES

    def get_state_age_minutes(self) -> float:
        """Get age of saved state in minutes, or -1 if no state file."""
        if not self.state_file.exists():
            return -1.0
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            saved_at = datetime.fromisoformat(state.get("saved_at", ""))
            return (datetime.utcnow() - saved_at).total_seconds() / 60
        except Exception:
            return -1.0
