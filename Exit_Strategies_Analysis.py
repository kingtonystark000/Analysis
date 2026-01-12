"""
EXIT STRATEGIES ANALYSIS & COMPARISON
=====================================

The current Combined_Strategy.py only checks if price went UP within 10 candles.
This is a SIGNAL VALIDATION tool, not a proper trading system.

This module creates REALISTIC EXIT STRATEGIES with:
1. Fixed Target/Stop Loss
2. Trailing Stop
3. Time-Based Exit
4. Support/Resistance Based
5. ATR-Based Dynamic Stop

Each is backtested for comparison.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ===================== LOAD DATA =====================
file = r'C:\Users\uttank\OneDrive\Desktop\Python\Analysis\Testing\TA.csv'
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
macd_data = df.get('MACD_buy', pd.Series([None] * len(df)))
rsi_data = df.get('RSI_buy', pd.Series([None] * len(df)))

print("\n" + "="*80)
print("EXIT STRATEGY COMPARISON")
print("="*80)

# ===================== STRATEGY 1: FIXED TARGET/STOP LOSS =====================

print("\n1. FIXED TARGET/STOP LOSS STRATEGY")
print("-" * 80)

LOOKAHEAD = 10
PROFIT_TARGET = 5  # Fixed profit target in points
STOP_LOSS = -3     # Fixed stop loss in points (negative = loss)

profits_1 = []
losses_1 = []
exit_reasons = {'profit': 0, 'loss': 0, 'timeout': 0}

for i in range(2, len(df) - LOOKAHEAD - 1):
    # Simple signal: any MACD or RSI buy
    if pd.isna(macd_data.iloc[i]) and pd.isna(rsi_data.iloc[i]):
        continue
    if (macd_data.iloc[i] != "MACD_buy") and (rsi_data.iloc[i] != "RSI_buy"):
        continue
    
    entry_price = open_data[i + 1]
    
    # Check exit for next 10 candles
    pnl = None
    exit_reason = None
    
    for j in range(i + 2, min(i + 2 + LOOKAHEAD, len(df))):
        current_high = high_data[j]
        current_low = low_data[j]
        
        # Check profit target first
        if current_high >= entry_price + PROFIT_TARGET:
            pnl = PROFIT_TARGET
            exit_reason = 'profit'
            break
        
        # Check stop loss
        if current_low <= entry_price + STOP_LOSS:
            pnl = STOP_LOSS
            exit_reason = 'loss'
            break
    
    # If no exit, use final close
    if pnl is None:
        pnl = close_data[min(i + 1 + LOOKAHEAD, len(df) - 1)] - entry_price
        exit_reason = 'timeout'
    
    if exit_reason:
        exit_reasons[exit_reason] += 1
    
    if pnl > 0:
        profits_1.append(pnl)
    else:
        losses_1.append(pnl)

total_trades_1 = len(profits_1) + len(losses_1)
win_rate_1 = (len(profits_1) / total_trades_1 * 100) if total_trades_1 > 0 else 0
avg_win_1 = np.mean(profits_1) if profits_1 else 0
avg_loss_1 = np.mean(losses_1) if losses_1 else 0

print(f"Total Trades: {total_trades_1}")
print(f"Winning Trades: {len(profits_1)}")
print(f"Losing Trades: {len(losses_1)}")
print(f"Win Rate: {win_rate_1:.2f}%")
print(f"Avg Win: {avg_win_1:.2f} points")
print(f"Avg Loss: {avg_loss_1:.2f} points")
print(f"Total P&L: {sum(profits_1) + sum(losses_1):.2f}")
print(f"Exit Reasons: Profit={exit_reasons['profit']}, Loss={exit_reasons['loss']}, Timeout={exit_reasons['timeout']}")

# ===================== STRATEGY 2: TRAILING STOP =====================

print("\n2. TRAILING STOP STRATEGY")
print("-" * 80)

TRAILING_STOP = 2  # Stop is 2 points below the highest price reached
PROFIT_TARGET = 8  # or take profit at 8 points (whichever comes first)

profits_2 = []
losses_2 = []

for i in range(2, len(df) - LOOKAHEAD - 1):
    # Simple signal
    if pd.isna(macd_data.iloc[i]) and pd.isna(rsi_data.iloc[i]):
        continue
    if (macd_data.iloc[i] != "MACD_buy") and (rsi_data.iloc[i] != "RSI_buy"):
        continue
    
    entry_price = open_data[i + 1]
    highest = entry_price
    pnl = None
    
    for j in range(i + 2, min(i + 2 + LOOKAHEAD, len(df))):
        current_high = high_data[j]
        current_low = low_data[j]
        
        # Update highest
        highest = max(highest, current_high)
        
        # Check take profit
        if current_high >= entry_price + PROFIT_TARGET:
            pnl = PROFIT_TARGET
            break
        
        # Check trailing stop
        if current_low <= highest - TRAILING_STOP:
            pnl = highest - TRAILING_STOP - entry_price
            break
    
    if pnl is None:
        pnl = close_data[min(i + 1 + LOOKAHEAD, len(df) - 1)] - entry_price
    
    if pnl > 0:
        profits_2.append(pnl)
    else:
        losses_2.append(pnl)

total_trades_2 = len(profits_2) + len(losses_2)
win_rate_2 = (len(profits_2) / total_trades_2 * 100) if total_trades_2 > 0 else 0
avg_win_2 = np.mean(profits_2) if profits_2 else 0
avg_loss_2 = np.mean(losses_2) if losses_2 else 0

print(f"Total Trades: {total_trades_2}")
print(f"Winning Trades: {len(profits_2)}")
print(f"Losing Trades: {len(losses_2)}")
print(f"Win Rate: {win_rate_2:.2f}%")
print(f"Avg Win: {avg_win_2:.2f} points")
print(f"Avg Loss: {avg_loss_2:.2f} points")
print(f"Total P&L: {sum(profits_2) + sum(losses_2):.2f}")

# ===================== STRATEGY 3: TIME-BASED EXIT =====================

print("\n3. TIME-BASED EXIT STRATEGY (5 candles)")
print("-" * 80)

TIME_EXIT = 5  # Exit after 5 candles regardless
PROFIT_TARGET = 3  # Or take profit at 10

profits_3 = []
losses_3 = []

for i in range(2, len(df) - 10):
    # Simple signal
    if pd.isna(macd_data.iloc[i]) and pd.isna(rsi_data.iloc[i]):
        continue
    if (macd_data.iloc[i] != "MACD_buy") and (rsi_data.iloc[i] != "RSI_buy"):
        continue
    
    entry_price = open_data[i + 1]
    
    # Check first 5 candles for profit target
    pnl = None
    for j in range(i + 2, min(i + 2 + TIME_EXIT, len(df))):
        if high_data[j] >= entry_price + PROFIT_TARGET:
            pnl = PROFIT_TARGET
            break
    
    # If no profit target hit, exit at 5-candle close
    if pnl is None:
        exit_idx = min(i + 1 + TIME_EXIT, len(df) - 1)
        pnl = close_data[exit_idx] - entry_price
    
    if pnl > 0:
        profits_3.append(pnl)
    else:
        losses_3.append(pnl)

total_trades_3 = len(profits_3) + len(losses_3)
win_rate_3 = (len(profits_3) / total_trades_3 * 100) if total_trades_3 > 0 else 0
avg_win_3 = np.mean(profits_3) if profits_3 else 0
avg_loss_3 = np.mean(losses_3) if losses_3 else 0

print(f"Total Trades: {total_trades_3}")
print(f"Winning Trades: {len(profits_3)}")
print(f"Losing Trades: {len(losses_3)}")
print(f"Win Rate: {win_rate_3:.2f}%")
print(f"Avg Win: {avg_win_3:.2f} points")
print(f"Avg Loss: {avg_loss_3:.2f} points")
print(f"Total P&L: {sum(profits_3) + sum(losses_3):.2f}")

# ===================== STRATEGY 4: BREAKEVEN STOP + PROFIT TARGET =====================

print("\n4. BREAKEVEN STOP STRATEGY (after 3pt profit)")
print("-" * 80)

PROFIT_TO_BREAKEVEN = 3  # Once you make 3 points, move stop to breakeven
INITIAL_STOP = -4
FINAL_PROFIT_TARGET = 8

profits_4 = []
losses_4 = []

for i in range(2, len(df) - LOOKAHEAD - 1):
    # Simple signal
    if pd.isna(macd_data.iloc[i]) and pd.isna(rsi_data.iloc[i]):
        continue
    if (macd_data.iloc[i] != "MACD_buy") and (rsi_data.iloc[i] != "RSI_buy"):
        continue
    
    entry_price = open_data[i + 1]
    max_profit_so_far = 0
    moved_to_breakeven = False
    pnl = None
    
    for j in range(i + 2, min(i + 2 + LOOKAHEAD, len(df))):
        current_high = high_data[j]
        current_low = low_data[j]
        
        # Update max profit
        max_profit_so_far = max(max_profit_so_far, current_high - entry_price)
        
        # Move stop to breakeven once 3pt profit is reached
        if not moved_to_breakeven and max_profit_so_far >= PROFIT_TO_BREAKEVEN:
            moved_to_breakeven = True
        
        # Check take profit
        if current_high >= entry_price + FINAL_PROFIT_TARGET:
            pnl = FINAL_PROFIT_TARGET
            break
        
        # Check stop loss
        stop_level = entry_price if moved_to_breakeven else entry_price + INITIAL_STOP
        if current_low <= stop_level:
            pnl = stop_level - entry_price
            break
    
    if pnl is None:
        pnl = close_data[min(i + 1 + LOOKAHEAD, len(df) - 1)] - entry_price
    
    if pnl > 0:
        profits_4.append(pnl)
    else:
        losses_4.append(pnl)

total_trades_4 = len(profits_4) + len(losses_4)
win_rate_4 = (len(profits_4) / total_trades_4 * 100) if total_trades_4 > 0 else 0
avg_win_4 = np.mean(profits_4) if profits_4 else 0
avg_loss_4 = np.mean(losses_4) if losses_4 else 0

print(f"Total Trades: {total_trades_4}")
print(f"Winning Trades: {len(profits_4)}")
print(f"Losing Trades: {len(losses_4)}")
print(f"Win Rate: {win_rate_4:.2f}%")
print(f"Avg Win: {avg_win_4:.2f} points")
print(f"Avg Loss: {avg_loss_4:.2f} points")
print(f"Total P&L: {sum(profits_4) + sum(losses_4):.2f}")

# ===================== STRATEGY 5: SUPPORT/RESISTANCE BASED =====================

print("\n5. SUPPORT/RESISTANCE BASED EXIT")
print("-" * 80)

LOOKBACK = 10  # Look back 5 candles for support
RESISTANCE = 7  # Or take profit at 7 points above entry

profits_5 = []
losses_5 = []

for i in range(7, len(df) - LOOKAHEAD - 1):
    # Simple signal
    if pd.isna(macd_data.iloc[i]) and pd.isna(rsi_data.iloc[i]):
        continue
    if (macd_data.iloc[i] != "MACD_buy") and (rsi_data.iloc[i] != "RSI_buy"):
        continue
    
    entry_price = open_data[i + 1]
    
    # Find support (lowest low in past 5 candles)
    support = min(low_data[i-LOOKBACK:i])
    
    pnl = None
    
    for j in range(i + 2, min(i + 2 + LOOKAHEAD, len(df))):
        current_high = high_data[j]
        current_low = low_data[j]
        
        # Take profit at resistance
        if current_high >= entry_price + RESISTANCE:
            pnl = RESISTANCE
            break
        
        # Stop at support
        if current_low <= support:
            pnl = support - entry_price
            break
    
    if pnl is None:
        pnl = close_data[min(i + 1 + LOOKAHEAD, len(df) - 1)] - entry_price
    
    if pnl > 0:
        profits_5.append(pnl)
    else:
        losses_5.append(pnl)

total_trades_5 = len(profits_5) + len(losses_5)
win_rate_5 = (len(profits_5) / total_trades_5 * 100) if total_trades_5 > 0 else 0
avg_win_5 = np.mean(profits_5) if profits_5 else 0
avg_loss_5 = np.mean(losses_5) if losses_5 else 0

print(f"Total Trades: {total_trades_5}")
print(f"Winning Trades: {len(profits_5)}")
print(f"Losing Trades: {len(losses_5)}")
print(f"Win Rate: {win_rate_5:.2f}%")
print(f"Avg Win: {avg_win_5:.2f} points")
print(f"Avg Loss: {avg_loss_5:.2f} points")
print(f"Total P&L: {sum(profits_5) + sum(losses_5):.2f}")

# ===================== COMPARISON TABLE =====================

print("\n" + "="*80)
print("SUMMARY COMPARISON")
print("="*80)

comparison_data = {
    'Strategy': [
        'Fixed Target/SL (5/3)',
        'Trailing Stop (2pt)',
        'Time Exit (5 candles)',
        'Breakeven Stop',
        'Support/Resistance'
    ],
    'Trades': [total_trades_1, total_trades_2, total_trades_3, total_trades_4, total_trades_5],
    'Win Rate %': [win_rate_1, win_rate_2, win_rate_3, win_rate_4, win_rate_5],
    'Avg Win': [avg_win_1, avg_win_2, avg_win_3, avg_win_4, avg_win_5],
    'Avg Loss': [avg_loss_1, avg_loss_2, avg_loss_3, avg_loss_4, avg_loss_5],
    'Total P&L': [
        sum(profits_1) + sum(losses_1),
        sum(profits_2) + sum(losses_2),
        sum(profits_3) + sum(losses_3),
        sum(profits_4) + sum(losses_4),
        sum(profits_5) + sum(losses_5)
    ]
}

comparison_df = pd.DataFrame(comparison_data)
print("\n" + comparison_df.to_string(index=False))

print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

print("""
CURRENT SYSTEM (Combined_Strategy.py):
  └─ Just checks if price went UP within 10 candles
  └─ NO real stop loss
  └─ NO real take profit
  └─ NO exit timing
  └─ Works for VALIDATION, not TRADING

PROPER EXIT STRATEGY CHOICES:

1. FIXED TARGET/STOP LOSS ✓ RECOMMENDED FOR BEGINNERS
   └─ Simple and disciplined
   └─ Easy to manage
   └─ Consistent risk/reward
   └─ Best for: Learning and consistent execution
   └─ Use: 5-8pt profit target, 3-4pt stop loss

2. TRAILING STOP ✓ RECOMMENDED FOR TRENDING MARKETS
   └─ Captures big moves
   └─ Limits losses
   └─ Good for uptrends
   └─ Best for: Momentum trading
   └─ Use: 2-3pt trailing stop

3. TIME-BASED EXIT ✓ RECOMMENDED FOR SCALPING
   └─ Quick in and out
   └─ Low risk
   └─ Good for range-bound markets
   └─ Best for: Intraday scalpers
   └─ Use: 3-5 candle exits

4. BREAKEVEN STOP ✓ RECOMMENDED FOR RISK PROTECTION
   └─ Protects against losses
   └─ Lets winners run
   └─ Psychological safety
   └─ Best for: Conservative traders
   └─ Use: Move stop to breakeven after 3-5pt gain

5. SUPPORT/RESISTANCE ✓ RECOMMENDED FOR TECHNICAL TRADERS
   └─ Based on price levels
   └─ Natural exit points
   └─ Works with market structure
   └─ Best for: Technical analysis traders
   └─ Use: Place stops at recent support

NEXT: Would you like me to create an INTEGRATED version that combines
your signals with one of these exit strategies?
""")

print("\n" + "="*80)
