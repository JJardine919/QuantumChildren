"""
VISUAL VETO SYSTEM
==================
Acts as a Gatekeeper between the Brain (Signals) and the Hands (Execution).

Workflow:
1. Watch 'etare_signals.json' for new signals.
2. If signal found -> Take Screenshot of Chart.
3. Send to Visual Agent (Claude/GPT-4V).
4. If Agent says "VETO", delete the signal.
5. If Agent says "APPROVED", let it pass.

The Robot now has EYES.
"""

import json
import time
import os
import logging
import base64
import re
from datetime import datetime
from pathlib import Path

# Screenshot libraries
try:
    import pyautogui
    HAS_PYAUTOGUI = True
except ImportError:
    HAS_PYAUTOGUI = False

try:
    import win32gui
    import win32ui
    import win32con
    from PIL import Image
    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

# Configuration
SIGNAL_FILE = "C:\\Users\\jjj10\\QuantumTradingLibrary\\etare_signals.json"
VETO_LOG = "veto_decision_log.txt"
SCREENSHOT_DIR = "C:\\Users\\jjj10\\QuantumTradingLibrary\\veto_screenshots"
MT5_WINDOW_TITLE = "MetaTrader 5"  # Partial match for window title

# Ensure screenshot directory exists
Path(SCREENSHOT_DIR).mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [VETO] - %(message)s")

# ============================================================================
# SCREENSHOT CAPTURE FUNCTIONS
# ============================================================================

def find_mt5_window():
    """Find the MetaTrader 5 window handle."""
    if not HAS_WIN32:
        return None

    def callback(hwnd, windows):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if MT5_WINDOW_TITLE.lower() in title.lower():
                windows.append((hwnd, title))
        return True

    windows = []
    win32gui.EnumWindows(callback, windows)

    if windows:
        return windows[0][0]  # Return first match
    return None


def capture_mt5_screenshot(symbol=None):
    """
    Capture a screenshot of the MT5 window.
    Returns the filepath to the saved screenshot.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"veto_{symbol}_{timestamp}.png" if symbol else f"veto_{timestamp}.png"
    filepath = os.path.join(SCREENSHOT_DIR, filename)

    # Method 1: Try to capture specific MT5 window (Windows only)
    if HAS_WIN32:
        hwnd = find_mt5_window()
        if hwnd:
            try:
                # Get window dimensions
                left, top, right, bottom = win32gui.GetWindowRect(hwnd)
                width = right - left
                height = bottom - top

                # Bring window to foreground
                win32gui.SetForegroundWindow(hwnd)
                time.sleep(0.3)  # Wait for window to come to front

                # Capture using pyautogui for the region
                if HAS_PYAUTOGUI:
                    screenshot = pyautogui.screenshot(region=(left, top, width, height))
                    screenshot.save(filepath)
                    logging.info(f"[EYES] MT5 window captured: {filepath}")
                    return filepath
            except Exception as e:
                logging.warning(f"Win32 capture failed: {e}, falling back to full screen")

    # Method 2: Fallback to full screen capture
    if HAS_PYAUTOGUI:
        try:
            screenshot = pyautogui.screenshot()
            screenshot.save(filepath)
            logging.info(f"[EYES] Full screen captured: {filepath}")
            return filepath
        except Exception as e:
            logging.error(f"Screenshot capture failed: {e}")
            return None

    logging.error("[EYES] No screenshot library available. Install pyautogui: pip install pyautogui")
    return None


def encode_image_to_base64(filepath):
    """Encode an image file to base64 for API submission."""
    with open(filepath, "rb") as image_file:
        return base64.standard_b64encode(image_file.read()).decode("utf-8")


# ============================================================================
# VISUAL ANALYSIS FUNCTIONS
# ============================================================================

TRAP_PATTERNS = """
You are a professional technical analyst reviewing a trading chart.
Analyze this chart for the proposed {action} trade on {symbol}.

LOOK FOR THESE TRAP PATTERNS (reasons to VETO):
1. Bull Trap: Price breaks above resistance but immediately reverses down
2. Bear Trap: Price breaks below support but immediately reverses up
3. False Breakout: Price pierces a level but closes back inside the range
4. Exhaustion Gap: Gap after extended move, often signals reversal
5. Divergence: Price making new highs/lows but RSI/MACD not confirming
6. Overextension: Price far from moving averages, due for mean reversion
7. Low Volume Breakout: Breakout without volume confirmation
8. Wedge Breakdown: Rising wedge (bearish) or falling wedge (bullish) pattern
9. Head & Shoulders: Classic reversal pattern near completion
10. Double Top/Bottom: Price hitting same level twice, reversal likely

LOOK FOR THESE CONFIRMATION PATTERNS (reasons to APPROVE):
1. Clean breakout with volume
2. Trend continuation after pullback
3. Support/Resistance bounce with confirmation
4. Moving average alignment (trending)
5. Healthy consolidation before continuation
6. Clear market structure (higher highs/lows or lower highs/lows)

YOUR TASK:
1. Describe what you see on the chart
2. Identify the current market structure
3. List any trap patterns you detect
4. Make a FINAL DECISION: Reply with exactly "DECISION: APPROVED" or "DECISION: VETO"
5. Provide a brief reason for your decision

Be conservative - when in doubt, VETO. It's better to miss a trade than take a bad one.
"""


def analyze_chart_with_claude(filepath, symbol, action):
    """
    Send the screenshot to Claude for visual analysis.
    Returns (is_approved: bool, reason: str)
    """
    if not HAS_ANTHROPIC:
        logging.warning("[EYES] Anthropic library not installed. pip install anthropic")
        return True, "Anthropic not available - defaulting to approve"

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logging.warning("[EYES] ANTHROPIC_API_KEY not set in environment")
        return True, "API key not configured - defaulting to approve"

    try:
        client = anthropic.Anthropic(api_key=api_key)

        # Encode the image
        image_data = encode_image_to_base64(filepath)

        # Prepare the prompt
        prompt = TRAP_PATTERNS.format(action=action, symbol=symbol)

        # Call Claude with vision
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ],
                }
            ],
        )

        # Parse the response
        response_text = message.content[0].text
        logging.info(f"[EYES] Claude Analysis:\n{response_text}")

        # Extract decision
        if "DECISION: VETO" in response_text.upper():
            # Extract reason (text after DECISION line)
            reason_match = re.search(r'DECISION:\s*VETO[.\s]*(.+?)(?:$|\n\n)', response_text, re.IGNORECASE | re.DOTALL)
            reason = reason_match.group(1).strip() if reason_match else "Trap pattern detected"
            return False, reason
        elif "DECISION: APPROVED" in response_text.upper():
            reason_match = re.search(r'DECISION:\s*APPROVED[.\s]*(.+?)(?:$|\n\n)', response_text, re.IGNORECASE | re.DOTALL)
            reason = reason_match.group(1).strip() if reason_match else "Chart looks clean"
            return True, reason
        else:
            # If unclear, be conservative and veto
            return False, "Unclear analysis - defaulting to veto for safety"

    except Exception as e:
        logging.error(f"[EYES] Claude analysis failed: {e}")
        return True, f"Analysis error: {e} - defaulting to approve"


# ============================================================================
# MAIN VISUAL CONFIRMATION FUNCTION
# ============================================================================

def check_visual_confirmation(symbol, action):
    """
    THE ROBOT'S EYES - Visual confirmation before trade execution.

    1. Capture screenshot of MT5 chart
    2. Send to Claude for visual analysis
    3. Return True (Approve) or False (Veto)

    Args:
        symbol: Trading pair (e.g., "BTCUSD", "EURUSD")
        action: Trade action ("BUY" or "SELL")

    Returns:
        tuple: (is_approved: bool, reason: str)
    """
    print(f"\n{'='*60}")
    print(f"[EYES] 👁️  VISUAL VETO SYSTEM ACTIVATED")
    print(f"[EYES] Analyzing: {symbol} | Action: {action}")
    print(f"{'='*60}")

    # Step 1: Capture Screenshot
    print("[EYES] Step 1: Capturing MT5 chart...")
    screenshot_path = capture_mt5_screenshot(symbol)

    if not screenshot_path:
        print("[EYES] ⚠️  Screenshot capture failed - defaulting to APPROVE")
        return True, "Screenshot capture failed - cannot analyze"

    print(f"[EYES] ✓ Screenshot saved: {screenshot_path}")

    # Step 2: Send to Claude for Analysis
    print("[EYES] Step 2: Sending to Visual Agent (Claude)...")
    is_approved, reason = analyze_chart_with_claude(screenshot_path, symbol, action)

    # Step 3: Report Decision
    print(f"\n[EYES] {'='*40}")
    if is_approved:
        print(f"[EYES] ✅ VERDICT: APPROVED")
    else:
        print(f"[EYES] ❌ VERDICT: VETO")
    print(f"[EYES] Reason: {reason}")
    print(f"[EYES] {'='*40}\n")

    return is_approved, reason


# ============================================================================
# QUICK TEST MODE - Analyze a provided screenshot
# ============================================================================

def test_with_screenshot(filepath, symbol="TEST", action="BUY"):
    """Test the visual analysis with a specific screenshot file."""
    if not os.path.exists(filepath):
        print(f"[TEST] File not found: {filepath}")
        return

    print(f"[TEST] Analyzing: {filepath}")
    is_approved, reason = analyze_chart_with_claude(filepath, symbol, action)

    print(f"\n[TEST] Result: {'APPROVED' if is_approved else 'VETO'}")
    print(f"[TEST] Reason: {reason}")

def monitor_signals():
    last_modified = 0
    
    print("Visual Veto System Active...")
    print(f"Watching: {SIGNAL_FILE}")

    while True:
        try:
            if not os.path.exists(SIGNAL_FILE):
                time.sleep(1)
                continue

            current_modified = os.path.getmtime(SIGNAL_FILE)
            
            if current_modified > last_modified:
                last_modified = current_modified
                
                # 1. Read Signals
                with open(SIGNAL_FILE, 'r') as f:
                    signals = json.load(f)
                
                modified = False
                approved_signals = {}
                
                # 2. Check each signal
                for symbol, data in signals.items():
                    if symbol.startswith("_"): 
                        approved_signals[symbol] = data # Keep metadata
                        continue
                        
                    action = data.get("action")
                    
                    # Call the Eyes
                    is_approved, reason = check_visual_confirmation(symbol, action)
                    
                    if is_approved:
                        approved_signals[symbol] = data
                    else:
                        logging.warning(f"VETOED {action} on {symbol}: {reason}")
                        modified = True
                        
                        # Log the Veto
                        with open(VETO_LOG, "a") as log:
                            log.write(f"{datetime.now()} - VETO - {symbol} - {reason}\n")

                # 3. If any vetoed, overwrite the file with only approved signals
                if modified:
                    # Wait a split second to ensure Brain is done writing
                    time.sleep(0.1)
                    with open(SIGNAL_FILE, 'w') as f:
                        json.dump(approved_signals, f, indent=2)
                    logging.info("Signal file updated (Vetoes removed).")
                    
            time.sleep(0.5)
            
        except Exception as e:
            logging.error(f"Error in monitoring loop: {e}")
            time.sleep(1)

if __name__ == "__main__":
    monitor_signals()
