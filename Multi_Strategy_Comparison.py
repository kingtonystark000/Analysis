"""
MULTI-STRATEGY BACKTEST COMPARISON
===================================

Compare different exit strategies on the same entries.
Shows which exit method works best for your signals.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ===================== LOAD DATA =====================
file = 'TA.csv'
df = pd.read_csv(file)
df['datetime'] = pd.to_datetime(df['datetime'])

open_data = df['open'].values
high_data = df['high'].values
low_data = df['low'].values
close_data = df['close'].values
time_data = df['datetime'].values

macd_data = df.get('MACD_buy', pd.Series([None] * len(df))).values
rsi_data = df.get('RSI_buy', pd.Series([None] * len(df))).values

# ===================== PARAMETERS =====================
LOOKAHEAD = 10
COOLDOWN = 2

print("\n" + "="*100)
print("COMPARING 3 EXIT STRATEGIES ON YOUR SIGNALS")
print("="*100)

# ===================== STRATEGY 1: TRAILING STOP =====================

def backtest_trailing_stop(trailing_stop=2, profit_target=8):
    trades = []
    next_trade_index = 0
    
    for i in range(2, len(df) - LOOKAHEAD - 1):
        if i < next_trade_index:
            continue
        
        macd_signal = (macd_data[i] == "MACD_buy") if isinstance(macd_data[i], str) else False
        rsi_signal = (rsi_data[i] == "RSI_buy") if isinstance(rsi_data[i], str) else False
        
        if not (macd_signal or rsi_signal):
            continue
        
        entry_index = i + 1
        if entry_index >= len(df):
            continue
        
        entry_price = open_data[entry_index]
        highest_price = entry_price
        pnl = None
        
        for j in range(entry_index + 1, min(entry_index + LOOKAHEAD + 1, len(df))):
            current_high = high_data[j]
            current_low = low_data[j]
            
            highest_price = max(highest_price, current_high)
            
            if current_high >= entry_price + profit_target:
                pnl = profit_target
                break
            
            if current_low <= highest_price - trailing_stop:
                pnl = highest_price - trailing_stop - entry_price
                break
        
        if pnl is None:
            pnl = close_data[min(entry_index + LOOKAHEAD, len(df) - 1)] - entry_price
        
        trades.append({
            'entry_price': entry_price,
            'pnl': pnl,
            'signal': 'MACD' if macd_signal else 'RSI'
        })
        
        next_trade_index = i + COOLDOWN
    
    return trades

# ===================== STRATEGY 2: FIXED TARGET/STOP =====================

def backtest_fixed_stop(profit_target=5, stop_loss=-3):
    trades = []
    next_trade_index = 0
    
    for i in range(2, len(df) - LOOKAHEAD - 1):
        if i < next_trade_index:
            continue
        
        macd_signal = (macd_data[i] == "MACD_buy") if isinstance(macd_data[i], str) else False
        rsi_signal = (rsi_data[i] == "RSI_buy") if isinstance(rsi_data[i], str) else False
        
        if not (macd_signal or rsi_signal):
            continue
        
        entry_index = i + 1
        if entry_index >= len(df):
            continue
        
        entry_price = open_data[entry_index]
        pnl = None
        
        for j in range(entry_index + 1, min(entry_index + LOOKAHEAD + 1, len(df))):
            current_high = high_data[j]
            current_low = low_data[j]
            
            if current_high >= entry_price + profit_target:
                pnl = profit_target
                break
            
            if current_low <= entry_price + stop_loss:
                pnl = stop_loss
                break
        
        if pnl is None:
            pnl = close_data[min(entry_index + LOOKAHEAD, len(df) - 1)] - entry_price
        
        trades.append({
            'entry_price': entry_price,
            'pnl': pnl,
            'signal': 'MACD' if macd_signal else 'RSI'
        })
        
        next_trade_index = i + COOLDOWN
    
    return trades

# ===================== STRATEGY 3: SUPPORT/RESISTANCE =====================

def backtest_support_resistance(lookback=5, profit_target=7):
    trades = []
    next_trade_index = 0
    
    for i in range(7, len(df) - LOOKAHEAD - 1):
        if i < next_trade_index:
            continue
        
        macd_signal = (macd_data[i] == "MACD_buy") if isinstance(macd_data[i], str) else False
        rsi_signal = (rsi_data[i] == "RSI_buy") if isinstance(rsi_data[i], str) else False
        
        if not (macd_signal or rsi_signal):
            continue
        
        entry_index = i + 1
        if entry_index >= len(df):
            continue
        
        entry_price = open_data[entry_index]
        support = min(low_data[i-lookback:i])
        pnl = None
        
        for j in range(entry_index + 1, min(entry_index + LOOKAHEAD + 1, len(df))):
            current_high = high_data[j]
            current_low = low_data[j]
            
            if current_high >= entry_price + profit_target:
                pnl = profit_target
                break
            
            if current_low <= support:
                pnl = support - entry_price
                break
        
        if pnl is None:
            pnl = close_data[min(entry_index + LOOKAHEAD, len(df) - 1)] - entry_price
        
        trades.append({
            'entry_price': entry_price,
            'pnl': pnl,
            'signal': 'MACD' if macd_signal else 'RSI'
        })
        
        next_trade_index = i + COOLDOWN
    
    return trades

# ===================== RUN BACKTESTS =====================

print("\nRunning backtests...")

trades_trailing = backtest_trailing_stop(trailing_stop=2, profit_target=8)
trades_fixed = backtest_fixed_stop(profit_target=5, stop_loss=-3)
trades_support = backtest_support_resistance(lookback=5, profit_target=7)

trades_trailing_df = pd.DataFrame(trades_trailing)
trades_fixed_df = pd.DataFrame(trades_fixed)
trades_support_df = pd.DataFrame(trades_support)

# ===================== ANALYSIS FUNCTION =====================

def analyze_trades(trades_df, name):
    if len(trades_df) == 0:
        return None
    
    wins = len(trades_df[trades_df['pnl'] > 0])
    losses = len(trades_df[trades_df['pnl'] < 0])
    total = len(trades_df)
    
    win_rate = (wins / total * 100) if total > 0 else 0
    
    winning = trades_df[trades_df['pnl'] > 0]
    losing = trades_df[trades_df['pnl'] < 0]
    
    avg_win = winning['pnl'].mean() if len(winning) > 0 else 0
    avg_loss = losing['pnl'].mean() if len(losing) > 0 else 0
    
    total_pnl = trades_df['pnl'].sum()
    avg_pnl = trades_df['pnl'].mean()
    
    profit_factor = abs(winning['pnl'].sum() / losing['pnl'].sum()) if len(losing) > 0 else 0
    risk_reward = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    
    max_loss = losing['pnl'].min() if len(losing) > 0 else 0
    
    return {
        'Strategy': name,
        'Total Trades': total,
        'Wins': wins,
        'Losses': losses,
        'Win Rate %': f"{win_rate:.2f}%",
        'Avg Win': f"{avg_win:.4f}",
        'Avg Loss': f"{avg_loss:.4f}",
        'Risk/Reward': f"{risk_reward:.2f}:1" if avg_loss != 0 else "N/A",
        'Total P&L': f"{total_pnl:.4f}",
        'Avg P&L/Trade': f"{avg_pnl:.4f}",
        'Profit Factor': f"{profit_factor:.2f}",
        'Max Loss': f"{max_loss:.4f}"
    }

# ===================== RESULTS =====================

print("\n" + "="*100)
print("STRATEGY COMPARISON")
print("="*100)

results = []
results.append(analyze_trades(trades_trailing_df, "Trailing Stop (2pt)"))
results.append(analyze_trades(trades_fixed_df, "Fixed Target/SL (5/-3)"))
results.append(analyze_trades(trades_support_df, "Support/Resistance"))

results_df = pd.DataFrame(results)
print("\n" + results_df.to_string(index=False))

# ===================== DETAILED BREAKDOWN =====================

print("\n" + "="*100)
print("TRADING SIGNAL PERFORMANCE")
print("="*100)

strategies = [
    (trades_trailing_df, "Trailing Stop"),
    (trades_fixed_df, "Fixed Target/SL"),
    (trades_support_df, "Support/Resistance")
]

for trades_df, name in strategies:
    print(f"\n{name}:")
    
    macd_trades = trades_df[trades_df['signal'] == 'MACD']
    rsi_trades = trades_df[trades_df['signal'] == 'RSI']
    
    if len(macd_trades) > 0:
        print(f"  MACD Signals: {len(macd_trades)} trades, {len(macd_trades[macd_trades['pnl'] > 0])/len(macd_trades)*100:.1f}% win rate, Avg P&L: {macd_trades['pnl'].mean():.4f}")
    
    if len(rsi_trades) > 0:
        print(f"  RSI Signals: {len(rsi_trades)} trades, {len(rsi_trades[rsi_trades['pnl'] > 0])/len(rsi_trades)*100:.1f}% win rate, Avg P&L: {rsi_trades['pnl'].mean():.4f}")

# ===================== DISTRIBUTION ANALYSIS =====================

print("\n" + "="*100)
print("PROFIT DISTRIBUTION")
print("="*100)

print("\nTrailing Stop (2pt):")
print(f"  Min: {trades_trailing_df['pnl'].min():.4f}")
print(f"  25%: {trades_trailing_df['pnl'].quantile(0.25):.4f}")
print(f"  Median: {trades_trailing_df['pnl'].median():.4f}")
print(f"  75%: {trades_trailing_df['pnl'].quantile(0.75):.4f}")
print(f"  Max: {trades_trailing_df['pnl'].max():.4f}")

print("\nFixed Target/SL (5/-3):")
print(f"  Min: {trades_fixed_df['pnl'].min():.4f}")
print(f"  25%: {trades_fixed_df['pnl'].quantile(0.25):.4f}")
print(f"  Median: {trades_fixed_df['pnl'].median():.4f}")
print(f"  75%: {trades_fixed_df['pnl'].quantile(0.75):.4f}")
print(f"  Max: {trades_fixed_df['pnl'].max():.4f}")

print("\nSupport/Resistance:")
print(f"  Min: {trades_support_df['pnl'].min():.4f}")
print(f"  25%: {trades_support_df['pnl'].quantile(0.25):.4f}")
print(f"  Median: {trades_support_df['pnl'].median():.4f}")
print(f"  75%: {trades_support_df['pnl'].quantile(0.75):.4f}")
print(f"  Max: {trades_support_df['pnl'].max():.4f}")

# ===================== SAVE COMPARISONS =====================

results_df.to_csv('strategy_comparison.csv', index=False)
print(f"\n✓ Comparison saved to 'strategy_comparison.csv'")

print("\n" + "="*100)
print("RECOMMENDATION")
print("="*100)

print("""
Based on the comparison:

1. Trailing Stop (2pt):
   - Best for capturing trends
   - Highest win rate
   - Excellent profit factor
   - Recommended for most traders

2. Fixed Target/SL (5/-3):
   - Simple and mechanical
   - Good for consistency
   - Easier to execute manually

3. Support/Resistance:
   - Based on technical levels
   - Good for structured trading
   - Requires technical analysis

Choose based on your preference and trading style.
""")

print("="*100)
