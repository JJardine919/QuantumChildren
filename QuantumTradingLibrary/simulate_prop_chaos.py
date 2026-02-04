import random
import statistics

# --- CONFIGURATION (The "System") ---
ACCOUNT_SIZE = 1000000.00
PROFIT_TARGET = 100000.00  # 10%
MAX_DAILY_LOSS = 50000.00  # 5%
MAX_TOTAL_DRAWDOWN = 100000.00 # 10%
DAYS = 21

# --- STRATEGY STATS (Your "Sniper" Logic) ---
WIN_RATE = 0.40           # 40% Win Rate (Conservative estimate for 1:3)
RISK_PER_TRADE_PCT = 0.50 # Risk 0.5% per trade ($500)
REWARD_RATIO = 3.0        # 1:3 Risk to Reward
TRADES_PER_DAY_AVG = 3    # Average signals per day

# --- CHAOS FACTOR (0.0 to 1.0) ---
# 0.0 = Perfect Execution
# 0.5 = Real World (Slippage, occasional missed entry)
# 1.0 = Nightmare Market (High slippage, stops blown, spreads widening)
CHAOS_LEVEL = 0.8 

def run_challenge(sim_id):
    balance = ACCOUNT_SIZE
    equity_high = ACCOUNT_SIZE
    daily_starting_balance = ACCOUNT_SIZE
    
    risk_amount = ACCOUNT_SIZE * (RISK_PER_TRADE_PCT / 100)
    
    history = []
    
    for day in range(1, DAYS + 1):
        daily_pnl = 0
        
        # Reset daily balance reference
        daily_starting_balance = balance
        
        # Determine trades for the day (Chaos affects volume)
        if random.random() < (0.1 * CHAOS_LEVEL):
            num_trades = 0 # Dead market day
        else:
            num_trades = int(random.gauss(TRADES_PER_DAY_AVG, 1))
            if num_trades < 0: num_trades = 0
            
        for _ in range(num_trades):
            # CHAOS: Slippage & Spread logic
            # In chaos, losers lose MORE, winners make slightly LESS
            slippage_loss = 1.0 + (random.uniform(0, 0.2) * CHAOS_LEVEL) # Up to 20% extra loss
            slippage_win = 1.0 - (random.uniform(0, 0.05) * CHAOS_LEVEL) # Up to 5% profit skim
            
            # The Trade Result
            is_win = random.random() < WIN_RATE
            
            if is_win:
                pnl = (risk_amount * REWARD_RATIO) * slippage_win
            else:
                pnl = -(risk_amount * slippage_loss)
            
            balance += pnl
            daily_pnl += pnl
            
            # Max Drawdown Check
            if balance < (ACCOUNT_SIZE - MAX_TOTAL_DRAWDOWN):
                return False, "MAX TOTAL DRAWDOWN HIT", balance, day
            
            # Daily Loss Check (Floating)
            # Most firms calculate based on Equity at start of day vs current equity
            current_drawdown = daily_starting_balance - balance
            if current_drawdown > MAX_DAILY_LOSS:
                return False, "MAX DAILY LOSS HIT", balance, day

        # Update Equity High
        if balance > equity_high:
            equity_high = balance
            
        # Check Profit Target
        if balance >= (ACCOUNT_SIZE + PROFIT_TARGET):
            return True, "PROFIT TARGET REACHED", balance, day
            
    # End of time
    if balance >= (ACCOUNT_SIZE + PROFIT_TARGET):
        return True, "PASSED ON LAST DAY", balance, DAYS
    else:
        return False, "TIME EXPIRED (DID NOT REACH TARGET)", balance, DAYS

def main():
    print(f"--- SIMULATING {DAYS} DAY PROP FIRM CHALLENGE ---")
    print(f"Account Size: ${ACCOUNT_SIZE:,.2f}")
    print(f"Chaos Level: {CHAOS_LEVEL*100}%")
    print(f"Strategy: {WIN_RATE*100}% Win Rate | 1:{REWARD_RATIO} RR")
    print("-" * 50)
    
    SIMULATIONS = 5000
    passes = 0
    failures = 0
    reasons = {}
    balances = []
    
    for i in range(SIMULATIONS):
        result, reason, final_bal, day = run_challenge(i)
        balances.append(final_bal)
        if result:
            passes += 1
        else:
            failures += 1
            reasons[reason] = reasons.get(reason, 0) + 1
            
    pass_rate = (passes / SIMULATIONS) * 100
    
    print(f"\nRESULTS ({SIMULATIONS} Runs):")
    print(f"PASS RATE: {pass_rate:.1f}%")
    print(f"FAIL RATE: {100 - pass_rate:.1f}%")
    print("\nFAILURE REASONS:")
    for r, count in reasons.items():
        print(f"  - {r}: {count} times ({count/SIMULATIONS*100:.1f}%)")
        
    print("\nFINANCIALS:")
    print(f"  - Avg End Balance: ${statistics.mean(balances):,.2f}")
    print(f"  - Worst Case: ${min(balances):,.2f}")
    print(f"  - Best Case:  ${max(balances):,.2f}")
    
    print("-" * 50)
    if pass_rate > 90:
        print("VERDICT: SYSTEM IS READY FOR WAR. 🟢")
    elif pass_rate > 50:
        print("VERDICT: RISKY BUT VIABLE. 🟡")
    else:
        print("VERDICT: DO NOT BUY CHALLENGE YET. 🔴")

if __name__ == "__main__":
    main()
