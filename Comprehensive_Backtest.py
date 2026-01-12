"""
COMPREHENSIVE BACKTEST ENGINE
==============================

Combines your trading signals with proper exit strategies.
Tests entry and exit on every trade with detailed metrics.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ===================== LOAD DATA =====================
file = 'TA.csv'
df = pd.read_csv(file)

# Convert datetime
df['datetime'] = pd.to_datetime(df['datetime'])

# Price data
open_data = df['open'].values
high_data = df['high'].values
low_data = df['low'].values
close_data = df['close'].values
time_data = df['datetime'].values

# Signal data
macd_data = df.get('MACD_buy', pd.Series([None] * len(df))).values
rsi_data = df.get('RSI_buy', pd.Series([None] * len(df))).values

# ===================== PARAMETERS =====================
LOOKAHEAD = 10
TRAILING_STOP = 2  # Points below highest price
PROFIT_TARGET = 8  # Points above entry
COOLDOWN = 2  # Candles between trades

print("\n" + "="*90)
print("COMPREHENSIVE BACKTEST - ENTRY AND EXIT ANALYSIS")
print("="*90)

print(f"\nStrategy Parameters:")
print(f"  Lookahead: {LOOKAHEAD} candles")
print(f"  Trailing Stop: {TRAILING_STOP} points below peak")
print(f"  Profit Target: {PROFIT_TARGET} points")
print(f"  Cooldown: {COOLDOWN} candles")

# ===================== TRADE TRACKING =====================

trades = []
next_trade_index = 0

for i in range(2, len(df) - LOOKAHEAD - 1):
    
    # Cooldown enforcement
    if i < next_trade_index:
        continue
    
    # Check for signal
    macd_signal = (macd_data[i] == "MACD_buy") if isinstance(macd_data[i], str) else False
    rsi_signal = (rsi_data[i] == "RSI_buy") if isinstance(rsi_data[i], str) else False
    
    if not (macd_signal or rsi_signal):
        continue
    
    # Entry
    entry_index = i + 1
    if entry_index >= len(df):
        continue
    
    entry_price = open_data[entry_index]
    entry_time = time_data[entry_index]
    highest_price = entry_price
    pnl = None
    exit_price = None
    exit_time = None
    exit_reason = None
    bars_held = 0
    
    # Exit simulation
    for j in range(entry_index + 1, min(entry_index + LOOKAHEAD + 1, len(df))):
        bars_held += 1
        current_high = high_data[j]
        current_low = low_data[j]
        
        # Update highest price for trailing stop
        highest_price = max(highest_price, current_high)
        
        # Check profit target first
        if current_high >= entry_price + PROFIT_TARGET:
            exit_price = entry_price + PROFIT_TARGET
            exit_time = time_data[j]
            exit_reason = 'PROFIT_TARGET'
            pnl = PROFIT_TARGET
            break
        
        # Check trailing stop
        trailing_stop_level = highest_price - TRAILING_STOP
        if current_low <= trailing_stop_level:
            exit_price = trailing_stop_level
            exit_time = time_data[j]
            exit_reason = 'TRAILING_STOP'
            pnl = trailing_stop_level - entry_price
            break
    
    # If no exit hit, exit at close
    if pnl is None:
        exit_idx = min(entry_index + LOOKAHEAD, len(df) - 1)
        exit_price = close_data[exit_idx]
        exit_time = time_data[exit_idx]
        exit_reason = 'TIMEOUT'
        pnl = exit_price - entry_price
    
    # Record trade
    trades.append({
        'entry_time': entry_time,
        'entry_price': entry_price,
        'exit_time': exit_time,
        'exit_price': exit_price,
        'pnl': pnl,
        'exit_reason': exit_reason,
        'bars_held': bars_held,
        'highest_price': highest_price,
        'max_favorable_move': highest_price - entry_price,
        'signal_type': 'MACD' if macd_signal else 'RSI'
    })
    
    next_trade_index = i + COOLDOWN

# ===================== CREATE DATAFRAME =====================

trades_df = pd.DataFrame(trades)

if len(trades_df) == 0:
    print("No trades generated!")
    exit()

# ===================== OVERALL STATISTICS =====================

print("\n" + "="*90)
print("OVERALL STATISTICS")
print("="*90)

total_trades = len(trades_df)
winning_trades = len(trades_df[trades_df['pnl'] > 0])
losing_trades = len(trades_df[trades_df['pnl'] < 0])
breakeven_trades = len(trades_df[trades_df['pnl'] == 0])
win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

print(f"\nTrade Count:")
print(f"  Total Trades: {total_trades}")
print(f"  Winning Trades: {winning_trades} ({winning_trades/total_trades*100:.2f}%)")
print(f"  Losing Trades: {losing_trades} ({losing_trades/total_trades*100:.2f}%)")
print(f"  Breakeven Trades: {breakeven_trades}")
print(f"  Win Rate: {win_rate:.2f}%")

# P&L Statistics
winning_trades_df = trades_df[trades_df['pnl'] > 0]
losing_trades_df = trades_df[trades_df['pnl'] < 0]

print(f"\nProfit/Loss Statistics:")
print(f"  Average Win: {winning_trades_df['pnl'].mean():.4f} points")
print(f"  Median Win: {winning_trades_df['pnl'].median():.4f} points")
print(f"  Max Win: {winning_trades_df['pnl'].max():.4f} points")
print(f"  Min Win: {winning_trades_df['pnl'].min():.4f} points")

print(f"\n  Average Loss: {losing_trades_df['pnl'].mean():.4f} points")
print(f"  Median Loss: {losing_trades_df['pnl'].median():.4f} points")
print(f"  Max Loss: {losing_trades_df['pnl'].min():.4f} points")

total_pnl = trades_df['pnl'].sum()
avg_pnl_per_trade = trades_df['pnl'].mean()
median_pnl = trades_df['pnl'].median()
std_pnl = trades_df['pnl'].std()

print(f"\nTotal Performance:")
print(f"  Total P&L: {total_pnl:.4f} points")
print(f"  Average P&L per Trade: {avg_pnl_per_trade:.4f} points")
print(f"  Median P&L per Trade: {median_pnl:.4f} points")
print(f"  Std Deviation: {std_pnl:.4f}")

# Risk/Reward
if len(losing_trades_df) > 0:
    risk_reward = abs(winning_trades_df['pnl'].mean() / losing_trades_df['pnl'].mean())
    print(f"  Risk/Reward Ratio: {risk_reward:.2f}:1")

profit_factor = abs(winning_trades_df['pnl'].sum() / losing_trades_df['pnl'].sum()) if len(losing_trades_df) > 0 else 0
print(f"  Profit Factor: {profit_factor:.2f}")

# Bars held
print(f"\nTrade Duration:")
print(f"  Average Bars Held: {trades_df['bars_held'].mean():.2f}")
print(f"  Max Bars Held: {trades_df['bars_held'].max()}")
print(f"  Min Bars Held: {trades_df['bars_held'].min()}")

# Exit reasons
print(f"\nExit Reasons:")
for reason in trades_df['exit_reason'].unique():
    count = len(trades_df[trades_df['exit_reason'] == reason])
    pct = count / len(trades_df) * 100
    print(f"  {reason}: {count} trades ({pct:.1f}%)")

# ===================== BY SIGNAL TYPE =====================

print("\n" + "="*90)
print("PERFORMANCE BY SIGNAL TYPE")
print("="*90)

for signal in trades_df['signal_type'].unique():
    subset = trades_df[trades_df['signal_type'] == signal]
    wins = len(subset[subset['pnl'] > 0])
    losses = len(subset[subset['pnl'] < 0])
    
    print(f"\n{signal} Signals:")
    print(f"  Count: {len(subset)}")
    print(f"  Win Rate: {wins/len(subset)*100:.2f}%")
    print(f"  Avg Win: {subset[subset['pnl'] > 0]['pnl'].mean():.4f}")
    print(f"  Avg Loss: {subset[subset['pnl'] < 0]['pnl'].mean():.4f}")
    print(f"  Total P&L: {subset['pnl'].sum():.4f}")

# ===================== SAMPLE TRADES =====================

print("\n" + "="*90)
print("SAMPLE TRADES (First 10)")
print("="*90)

display_cols = ['entry_time', 'entry_price', 'exit_price', 'pnl', 'exit_reason', 'bars_held']
print("\n" + trades_df[display_cols].head(10).to_string(index=False))

# ===================== BEST TRADES =====================

print("\n" + "="*90)
print("BEST TRADES (Top 10)")
print("="*90)

print("\n" + trades_df.nlargest(10, 'pnl')[display_cols].to_string(index=False))

# ===================== WORST TRADES =====================

print("\n" + "="*90)
print("WORST TRADES (Bottom 10)")
print("="*90)

print("\n" + trades_df.nsmallest(10, 'pnl')[display_cols].to_string(index=False))

# ===================== CONSECUTIVE WINS/LOSSES =====================

print("\n" + "="*90)
print("CONSECUTIVE STREAKS")
print("="*90)

max_consecutive_wins = 0
max_consecutive_losses = 0
current_wins = 0
current_losses = 0

for idx, row in trades_df.iterrows():
    if row['pnl'] > 0:
        current_wins += 1
        current_losses = 0
        max_consecutive_wins = max(max_consecutive_wins, current_wins)
    elif row['pnl'] < 0:
        current_losses += 1
        current_wins = 0
        max_consecutive_losses = max(max_consecutive_losses, current_losses)
    else:
        current_wins = 0
        current_losses = 0

print(f"\nMax Consecutive Wins: {max_consecutive_wins}")
print(f"Max Consecutive Losses: {max_consecutive_losses}")

# ===================== EQUITY CURVE =====================

print("\n" + "="*90)
print("EQUITY CURVE & DRAWDOWN")
print("="*90)

trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
trades_df['running_max'] = trades_df['cumulative_pnl'].cummax()
trades_df['drawdown'] = trades_df['cumulative_pnl'] - trades_df['running_max']

max_drawdown = trades_df['drawdown'].min()
print(f"\nMax Drawdown: {max_drawdown:.4f} points")
print(f"Final Equity: {trades_df['cumulative_pnl'].iloc[-1]:.4f} points")

# ===================== SAVE RESULTS =====================

trades_df.to_csv('backtest_trades_detailed.csv', index=False)
print(f"\n✓ Detailed trades saved to 'backtest_trades_detailed.csv'")

# ===================== EXPORT SUMMARY =====================

summary_stats = {
    'Metric': [
        'Total Trades',
        'Winning Trades',
        'Losing Trades',
        'Win Rate %',
        'Average Win',
        'Average Loss',
        'Total P&L',
        'Avg P&L per Trade',
        'Risk/Reward Ratio',
        'Profit Factor',
        'Max Drawdown',
        'Max Consecutive Wins',
        'Max Consecutive Losses'
    ],
    'Value': [
        total_trades,
        winning_trades,
        losing_trades,
        f"{win_rate:.2f}%",
        f"{winning_trades_df['pnl'].mean():.4f}",
        f"{losing_trades_df['pnl'].mean():.4f}",
        f"{total_pnl:.4f}",
        f"{avg_pnl_per_trade:.4f}",
        f"{risk_reward:.2f}:1" if len(losing_trades_df) > 0 else "N/A",
        f"{profit_factor:.2f}",
        f"{max_drawdown:.4f}",
        max_consecutive_wins,
        max_consecutive_losses
    ]
}

summary_df = pd.DataFrame(summary_stats)
summary_df.to_csv('backtest_summary.csv', index=False)

print(f"✓ Summary stats saved to 'backtest_summary.csv'")

print("\n" + "="*90)
print("BACKTEST COMPLETE")
print("="*90)
print("\nNext step: Open 'backtest_trades_detailed.csv' in Excel for full analysis")
print("Or review 'backtest_summary.csv' for quick overview")
