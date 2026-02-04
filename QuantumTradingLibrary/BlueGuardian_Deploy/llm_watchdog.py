# ================================================
# llm_watchdog.py
# Blue Guardian LLM Emergency Shutoff Watchdog
# Uses Ollama Gemma 3 12B for intelligent risk monitoring
# ================================================
import os
import json
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("WARNING: ollama not installed - pip install ollama")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | LLM_WATCHDOG | %(message)s",
    handlers=[
        logging.FileHandler("llm_watchdog.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert levels for the watchdog"""
    NORMAL = "NORMAL"
    CAUTION = "CAUTION"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"


@dataclass
class WatchdogDecision:
    """Decision from the LLM watchdog"""
    timestamp: str
    alert_level: AlertLevel
    allow_trading: bool
    close_positions: bool
    reason: str
    recommendations: List[str]
    analysis: str
    confidence: float


class LLMWatchdog:
    """
    LLM-based emergency shutoff watchdog for Blue Guardian.

    Features:
    - Monitors account status, market conditions, and quantum signals
    - Uses Gemma 3 12B for intelligent risk assessment
    - Can halt trading or close positions in emergencies
    - Logs all decisions for audit trail
    """

    # Default model - Gemma 3 12B for production, smaller for testing
    DEFAULT_MODEL = "gemma3:12b"
    FALLBACK_MODEL = "gemma2:2b"

    # Alert thresholds
    THRESHOLDS = {
        'daily_drawdown_warning': 3.0,     # % - start warning
        'daily_drawdown_critical': 4.0,    # % - critical, reduce trading
        'daily_drawdown_emergency': 4.5,   # % - emergency shutoff
        'max_drawdown_warning': 6.0,       # %
        'max_drawdown_critical': 7.0,      # %
        'max_drawdown_emergency': 8.0,     # %
        'consecutive_losses_warning': 3,   # trades
        'consecutive_losses_critical': 5,  # trades
        'quantum_entropy_high': 2.8,       # high uncertainty
        'win_rate_minimum': 0.40,          # below this = concern
    }

    def __init__(self, model_name: str = None,
                 check_interval_seconds: int = 60):
        self.model_name = model_name or self.DEFAULT_MODEL
        self.check_interval = check_interval_seconds
        self.running = False
        self.thread = None
        self.last_decision: Optional[WatchdogDecision] = None
        self.decision_history: List[WatchdogDecision] = []
        self.emergency_callback = None
        self.lock = threading.Lock()

        # State tracking
        self.account_states: Dict[str, Dict] = {}
        self.recent_trades: Dict[str, List[Dict]] = {}

        self._verify_model()
        log.info(f"LLM Watchdog initialized with model: {self.model_name}")

    def _verify_model(self):
        """Verify the LLM model is available"""
        if not OLLAMA_AVAILABLE:
            log.warning("Ollama not available - watchdog will use rule-based fallback")
            return

        try:
            models = ollama.list()
            model_names = [m.get('name', '') for m in models.get('models', [])]

            if self.model_name not in str(model_names):
                log.warning(f"Model {self.model_name} not found, will try to pull")
                # Try to use fallback first
                if self.FALLBACK_MODEL in str(model_names):
                    log.info(f"Using fallback model: {self.FALLBACK_MODEL}")
                    self.model_name = self.FALLBACK_MODEL
                else:
                    log.info(f"Attempting to pull {self.model_name}...")
                    try:
                        ollama.pull(self.model_name)
                    except Exception as e:
                        log.error(f"Failed to pull model: {e}")
            else:
                log.info(f"Model {self.model_name} verified")

        except Exception as e:
            log.error(f"Error verifying model: {e}")

    def set_emergency_callback(self, callback):
        """Set callback function for emergency shutoffs"""
        self.emergency_callback = callback

    def update_account_state(self, account_name: str, state: Dict):
        """
        Update the state of an account for monitoring.

        Args:
            account_name: Account identifier
            state: Dict with balance, equity, drawdown, etc.
        """
        with self.lock:
            self.account_states[account_name] = {
                **state,
                'last_update': datetime.now().isoformat()
            }

    def add_trade_result(self, account_name: str, trade: Dict):
        """
        Add a trade result for tracking.

        Args:
            account_name: Account identifier
            trade: Dict with trade details and result
        """
        with self.lock:
            if account_name not in self.recent_trades:
                self.recent_trades[account_name] = []

            self.recent_trades[account_name].append({
                **trade,
                'timestamp': datetime.now().isoformat()
            })

            # Keep only last 50 trades per account
            self.recent_trades[account_name] = self.recent_trades[account_name][-50:]

    def _calculate_metrics(self, account_name: str) -> Dict:
        """Calculate risk metrics for an account"""
        state = self.account_states.get(account_name, {})
        trades = self.recent_trades.get(account_name, [])

        metrics = {
            'daily_drawdown': state.get('daily_drawdown', 0),
            'max_drawdown': state.get('max_drawdown', 0),
            'current_balance': state.get('balance', 0),
            'current_equity': state.get('equity', 0),
            'open_positions': state.get('open_positions', 0),
            'unrealized_pnl': state.get('unrealized_pnl', 0),
        }

        # Calculate win rate from recent trades
        if trades:
            wins = sum(1 for t in trades if t.get('profit_usd', 0) > 0)
            metrics['win_rate'] = wins / len(trades)
            metrics['total_trades'] = len(trades)

            # Consecutive losses
            consecutive_losses = 0
            for t in reversed(trades):
                if t.get('profit_usd', 0) < 0:
                    consecutive_losses += 1
                else:
                    break
            metrics['consecutive_losses'] = consecutive_losses

            # Average profit/loss
            profits = [t.get('profit_usd', 0) for t in trades if t.get('profit_usd', 0) > 0]
            losses = [t.get('profit_usd', 0) for t in trades if t.get('profit_usd', 0) < 0]

            metrics['avg_win'] = sum(profits) / len(profits) if profits else 0
            metrics['avg_loss'] = sum(losses) / len(losses) if losses else 0

        else:
            metrics['win_rate'] = 0.5
            metrics['total_trades'] = 0
            metrics['consecutive_losses'] = 0
            metrics['avg_win'] = 0
            metrics['avg_loss'] = 0

        return metrics

    def _rule_based_check(self, metrics: Dict) -> Tuple[AlertLevel, bool, str]:
        """
        Rule-based risk check as fallback when LLM unavailable.

        Returns:
            Tuple of (alert_level, allow_trading, reason)
        """
        T = self.THRESHOLDS

        # Emergency conditions - immediate shutoff
        if metrics['daily_drawdown'] >= T['daily_drawdown_emergency']:
            return AlertLevel.EMERGENCY, False, \
                f"Daily drawdown {metrics['daily_drawdown']:.2f}% exceeds emergency limit"

        if metrics['max_drawdown'] >= T['max_drawdown_emergency']:
            return AlertLevel.EMERGENCY, False, \
                f"Max drawdown {metrics['max_drawdown']:.2f}% exceeds emergency limit"

        # Critical conditions - stop new trades
        if metrics['daily_drawdown'] >= T['daily_drawdown_critical']:
            return AlertLevel.CRITICAL, False, \
                f"Daily drawdown {metrics['daily_drawdown']:.2f}% at critical level"

        if metrics['max_drawdown'] >= T['max_drawdown_critical']:
            return AlertLevel.CRITICAL, False, \
                f"Max drawdown {metrics['max_drawdown']:.2f}% at critical level"

        if metrics['consecutive_losses'] >= T['consecutive_losses_critical']:
            return AlertLevel.CRITICAL, False, \
                f"{metrics['consecutive_losses']} consecutive losses - trading halted"

        # Warning conditions - allow trading with caution
        if metrics['daily_drawdown'] >= T['daily_drawdown_warning']:
            return AlertLevel.WARNING, True, \
                f"Daily drawdown {metrics['daily_drawdown']:.2f}% - trade with caution"

        if metrics['max_drawdown'] >= T['max_drawdown_warning']:
            return AlertLevel.WARNING, True, \
                f"Max drawdown {metrics['max_drawdown']:.2f}% - reduce position sizes"

        if metrics['consecutive_losses'] >= T['consecutive_losses_warning']:
            return AlertLevel.CAUTION, True, \
                f"{metrics['consecutive_losses']} consecutive losses - be careful"

        if metrics['win_rate'] < T['win_rate_minimum'] and metrics['total_trades'] >= 10:
            return AlertLevel.CAUTION, True, \
                f"Win rate {metrics['win_rate']*100:.1f}% below target"

        return AlertLevel.NORMAL, True, "All metrics within acceptable range"

    def _llm_analysis(self, account_name: str, metrics: Dict,
                      quantum_features: Dict = None,
                      market_context: str = None) -> WatchdogDecision:
        """
        Use LLM for intelligent risk analysis.

        Args:
            account_name: Account identifier
            metrics: Calculated risk metrics
            quantum_features: Optional quantum features
            market_context: Optional market context

        Returns:
            WatchdogDecision
        """
        # Build prompt
        prompt = f"""BLUE GUARDIAN RISK ASSESSMENT

ACCOUNT: {account_name}
TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

RISK METRICS:
- Daily Drawdown: {metrics['daily_drawdown']:.2f}%
- Max Drawdown: {metrics['max_drawdown']:.2f}%
- Current Balance: ${metrics['current_balance']:.2f}
- Current Equity: ${metrics['current_equity']:.2f}
- Open Positions: {metrics['open_positions']}
- Unrealized P/L: ${metrics['unrealized_pnl']:.2f}
- Recent Win Rate: {metrics['win_rate']*100:.1f}%
- Consecutive Losses: {metrics['consecutive_losses']}
- Total Trades Today: {metrics['total_trades']}
- Average Win: ${metrics['avg_win']:.2f}
- Average Loss: ${metrics['avg_loss']:.2f}
"""

        if quantum_features:
            prompt += f"""
QUANTUM ANALYSIS:
- Quantum Entropy: {quantum_features.get('quantum_entropy', 0):.3f}
- Dominant State Probability: {quantum_features.get('dominant_state_prob', 0):.3f}
- Phase Coherence: {quantum_features.get('phase_coherence', 0):.3f}
- Entanglement Degree: {quantum_features.get('entanglement_degree', 0):.3f}
"""

        if market_context:
            prompt += f"""
MARKET CONTEXT:
{market_context}
"""

        prompt += """
PROP FIRM LIMITS:
- Daily Drawdown Limit: 5%
- Max Drawdown Limit: 10%

TASK:
Analyze the risk situation and provide a decision in this EXACT format:

ALERT_LEVEL: [NORMAL/CAUTION/WARNING/CRITICAL/EMERGENCY]
ALLOW_TRADING: [YES/NO]
CLOSE_POSITIONS: [YES/NO]
CONFIDENCE: [0-100]%

REASON:
[1-2 sentence explanation of your decision]

RECOMMENDATIONS:
- [Recommendation 1]
- [Recommendation 2]

ANALYSIS:
[Brief analysis of the situation, noting any concerns]
"""

        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": 0.3,
                    "num_predict": 500,
                }
            )

            return self._parse_llm_response(response['response'], metrics)

        except Exception as e:
            log.error(f"LLM analysis failed: {e}")
            # Fall back to rule-based
            alert_level, allow_trading, reason = self._rule_based_check(metrics)
            return WatchdogDecision(
                timestamp=datetime.now().isoformat(),
                alert_level=alert_level,
                allow_trading=allow_trading,
                close_positions=(alert_level == AlertLevel.EMERGENCY),
                reason=reason,
                recommendations=["LLM unavailable - using rule-based fallback"],
                analysis="Rule-based risk assessment",
                confidence=0.8
            )

    def _parse_llm_response(self, response: str, metrics: Dict) -> WatchdogDecision:
        """Parse LLM response into WatchdogDecision"""
        import re

        # Default values
        alert_level = AlertLevel.NORMAL
        allow_trading = True
        close_positions = False
        confidence = 0.7
        reason = "Unable to parse LLM response"
        recommendations = []
        analysis = response

        try:
            # Parse alert level
            level_match = re.search(r'ALERT_LEVEL:\s*(\w+)', response, re.I)
            if level_match:
                level_str = level_match.group(1).upper()
                if level_str in [e.value for e in AlertLevel]:
                    alert_level = AlertLevel(level_str)

            # Parse allow trading
            trading_match = re.search(r'ALLOW_TRADING:\s*(YES|NO)', response, re.I)
            if trading_match:
                allow_trading = trading_match.group(1).upper() == 'YES'

            # Parse close positions
            close_match = re.search(r'CLOSE_POSITIONS:\s*(YES|NO)', response, re.I)
            if close_match:
                close_positions = close_match.group(1).upper() == 'YES'

            # Parse confidence
            conf_match = re.search(r'CONFIDENCE:\s*(\d+)', response, re.I)
            if conf_match:
                confidence = int(conf_match.group(1)) / 100.0

            # Parse reason
            reason_match = re.search(r'REASON:\s*(.+?)(?=RECOMMENDATIONS:|ANALYSIS:|$)',
                                     response, re.I | re.S)
            if reason_match:
                reason = reason_match.group(1).strip()

            # Parse recommendations
            rec_match = re.search(r'RECOMMENDATIONS:\s*(.+?)(?=ANALYSIS:|$)',
                                  response, re.I | re.S)
            if rec_match:
                rec_text = rec_match.group(1)
                recommendations = [
                    line.strip().lstrip('- ')
                    for line in rec_text.split('\n')
                    if line.strip() and line.strip() != '-'
                ]

            # Parse analysis
            analysis_match = re.search(r'ANALYSIS:\s*(.+?)$', response, re.I | re.S)
            if analysis_match:
                analysis = analysis_match.group(1).strip()

        except Exception as e:
            log.error(f"Error parsing LLM response: {e}")

        # Override with rule-based if metrics are critical
        rule_level, rule_allow, rule_reason = self._rule_based_check(metrics)
        if rule_level.value > alert_level.value:  # Rule-based is more severe
            alert_level = rule_level
            allow_trading = rule_allow
            reason = f"{reason} [Rule override: {rule_reason}]"
            if rule_level == AlertLevel.EMERGENCY:
                close_positions = True

        return WatchdogDecision(
            timestamp=datetime.now().isoformat(),
            alert_level=alert_level,
            allow_trading=allow_trading,
            close_positions=close_positions,
            reason=reason,
            recommendations=recommendations,
            analysis=analysis,
            confidence=confidence
        )

    def check_account(self, account_name: str,
                      quantum_features: Dict = None,
                      market_context: str = None,
                      use_llm: bool = True) -> WatchdogDecision:
        """
        Perform a risk check on an account.

        Args:
            account_name: Account identifier
            quantum_features: Optional quantum features
            market_context: Optional market context
            use_llm: Whether to use LLM analysis

        Returns:
            WatchdogDecision
        """
        metrics = self._calculate_metrics(account_name)

        if use_llm and OLLAMA_AVAILABLE:
            decision = self._llm_analysis(account_name, metrics,
                                         quantum_features, market_context)
        else:
            alert_level, allow_trading, reason = self._rule_based_check(metrics)
            decision = WatchdogDecision(
                timestamp=datetime.now().isoformat(),
                alert_level=alert_level,
                allow_trading=allow_trading,
                close_positions=(alert_level == AlertLevel.EMERGENCY),
                reason=reason,
                recommendations=[],
                analysis="Rule-based assessment",
                confidence=0.9
            )

        # Store decision
        with self.lock:
            self.last_decision = decision
            self.decision_history.append(decision)
            # Keep only last 100 decisions
            self.decision_history = self.decision_history[-100:]

        # Log decision
        log.info(f"[{account_name}] {decision.alert_level.value}: {decision.reason}")

        # Handle emergency
        if decision.alert_level == AlertLevel.EMERGENCY:
            self._handle_emergency(account_name, decision)

        return decision

    def _handle_emergency(self, account_name: str, decision: WatchdogDecision):
        """Handle emergency situation"""
        log.critical(f"EMERGENCY for {account_name}: {decision.reason}")

        if self.emergency_callback:
            try:
                self.emergency_callback(account_name, decision)
            except Exception as e:
                log.error(f"Emergency callback failed: {e}")

    def start_monitoring(self):
        """Start continuous monitoring in background thread"""
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        log.info("Watchdog monitoring started")

    def stop_monitoring(self):
        """Stop monitoring"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        log.info("Watchdog monitoring stopped")

    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Check all accounts
                for account_name in list(self.account_states.keys()):
                    self.check_account(account_name, use_llm=True)

                time.sleep(self.check_interval)

            except Exception as e:
                log.error(f"Monitoring loop error: {e}")
                time.sleep(10)

    def get_status(self) -> Dict:
        """Get current watchdog status"""
        return {
            'running': self.running,
            'model': self.model_name,
            'accounts_monitored': len(self.account_states),
            'last_decision': self.last_decision.__dict__ if self.last_decision else None,
            'decisions_count': len(self.decision_history),
        }


# Singleton instance
_watchdog_instance = None

def get_watchdog(model_name: str = None) -> LLMWatchdog:
    """Get or create the singleton watchdog instance"""
    global _watchdog_instance
    if _watchdog_instance is None:
        _watchdog_instance = LLMWatchdog(model_name)
    return _watchdog_instance


if __name__ == "__main__":
    # Test the watchdog
    watchdog = get_watchdog()

    # Update test account state
    watchdog.update_account_state("TEST_ACCOUNT", {
        'balance': 10000,
        'equity': 9800,
        'daily_drawdown': 2.0,
        'max_drawdown': 3.5,
        'open_positions': 1,
        'unrealized_pnl': -200,
    })

    # Add some test trades
    watchdog.add_trade_result("TEST_ACCOUNT", {'profit_usd': 50})
    watchdog.add_trade_result("TEST_ACCOUNT", {'profit_usd': -30})
    watchdog.add_trade_result("TEST_ACCOUNT", {'profit_usd': 25})

    # Check account
    decision = watchdog.check_account("TEST_ACCOUNT", use_llm=False)

    print(f"\nDecision: {decision.alert_level.value}")
    print(f"Allow Trading: {decision.allow_trading}")
    print(f"Close Positions: {decision.close_positions}")
    print(f"Reason: {decision.reason}")
    print(f"Confidence: {decision.confidence * 100:.0f}%")
