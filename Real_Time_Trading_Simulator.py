"""
Real-Time Trading Simulator with Database
==========================================
Simulates live trading on NIFTY 50 minute data using the combined strategy.
Tracks every trade with entry time, signal type, exit strategy, and P&L.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sqlite3
import os

# Configuration
PROFIT_TARGET = 8  # Points
TRAILING_STOP = 2  # Points below peak
LOOKAHEAD = 10  # Candles for signal validation
COOLDOWN = 2  # Candles between trades

# Read the large dataset
print("Loading NIFTY 50 minute data...")
df = pd.read_csv('NIFTY 50_minute_data.csv')
print(f"Loaded {len(df)} candles")
print(f"Date range: {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}")

# Calculate indicators using TA-Lib functions
def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD indicator"""
    ema_fast = data.ewm(span=fast).mean()
    ema_slow = data.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_rsi(data, period=14):
    """Calculate RSI indicator"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def detect_morning_star(idx, opens, highs, lows, closes):
    """Detect Morning Star pattern (3 candles)"""
    if idx < 3:
        return False
    
    i = idx - 2
    # First: Long bearish candle
    c1_down = closes[i] < opens[i] and (opens[i] - closes[i]) > 0.5 * (highs[i] - lows[i])
    # Second: Small candle (star) at bottom
    c2_small = abs(opens[i+1] - closes[i+1]) < 0.3 * (highs[i+1] - lows[i+1])
    c2_gap_down = max(opens[i+1], closes[i+1]) < min(opens[i], closes[i])
    # Third: Long bullish candle
    c3_up = closes[i+2] > opens[i+2] and (closes[i+2] - opens[i+2]) > 0.5 * (highs[i+2] - lows[i+2])
    c3_gap_up = min(opens[i+2], closes[i+2]) > max(opens[i+1], closes[i+1])
    
    return c1_down and c2_small and c2_gap_down and c3_up and c3_gap_up

def detect_evening_star(idx, opens, highs, lows, closes):
    """Detect Evening Star pattern (3 candles)"""
    if idx < 3:
        return False
    
    i = idx - 2
    # First: Long bullish candle
    c1_up = closes[i] > opens[i] and (closes[i] - opens[i]) > 0.5 * (highs[i] - lows[i])
    # Second: Small candle (star) at top
    c2_small = abs(opens[i+1] - closes[i+1]) < 0.3 * (highs[i+1] - lows[i+1])
    c2_gap_up = min(opens[i+1], closes[i+1]) > max(opens[i], closes[i])
    # Third: Long bearish candle
    c3_down = closes[i+2] < opens[i+2] and (opens[i+2] - closes[i+2]) > 0.5 * (highs[i+2] - lows[i+2])
    c3_gap_down = max(opens[i+2], closes[i+2]) < min(opens[i+1], closes[i+1])
    
    return c1_up and c2_small and c2_gap_up and c3_down and c3_gap_down

def detect_hammer(idx, opens, highs, lows, closes):
    """Detect Hammer pattern (1 candle)"""
    if idx < 1:
        return False
    
    i = idx
    body = abs(closes[i] - opens[i])
    lower_wick = min(opens[i], closes[i]) - lows[i]
    upper_wick = highs[i] - max(opens[i], closes[i])
    total_range = highs[i] - lows[i]
    
    return (lower_wick > 2 * body and upper_wick < 0.3 * total_range and 
            body > 0.2 * total_range and closes[i] > opens[i])

def calculate_buy_score(idx, macd_line, macd_signal, rsi, opens, highs, lows, closes):
    """Calculate buy score (0-100) based on all indicators"""
    score = 0
    components = {}
    
    # MACD signal (0-20 points)
    if idx > 0:
        macd_cross = (macd_line[idx-1] < macd_signal[idx-1] and 
                     macd_line[idx] > macd_signal[idx])
        if macd_cross:
            score += 20
            components['MACD'] = 20
        else:
            components['MACD'] = 0
    
    # RSI signal (0-20 points)
    rsi_value = rsi[idx]
    if not pd.isna(rsi_value):
        if rsi_value < 30:
            score += 20
            components['RSI'] = 20
        elif rsi_value < 40:
            score += 10
            components['RSI'] = 10
        else:
            components['RSI'] = 0
    
    # Morning Star pattern (0-25 points)
    if detect_morning_star(idx, opens, highs, lows, closes):
        score += 25
        components['Morning_Star'] = 25
    else:
        components['Morning_Star'] = 0
    
    # Hammer pattern (0-20 points)
    if detect_hammer(idx, opens, highs, lows, closes):
        score += 20
        components['Hammer'] = 20
    else:
        components['Hammer'] = 0
    
    # Evening Star (detector only, not additive)
    components['Evening_Star'] = 1 if detect_evening_star(idx, opens, highs, lows, closes) else 0
    
    # MACD + RSI Confirmation bonus (0-15 points)
    if components['MACD'] > 0 and components['RSI'] > 0:
        bonus = min(15, (components['MACD'] + components['RSI']) // 3)
        score += bonus
        components['Confirmation_Bonus'] = bonus
    else:
        components['Confirmation_Bonus'] = 0
    
    return min(100, score), components

# Initialize arrays for calculations
opens = df['open'].values
highs = df['high'].values
lows = df['low'].values
closes = df['close'].values
datetimes = df['datetime'].values

# Calculate indicators
print("Calculating indicators...")
macd_line, macd_signal, macd_hist = calculate_macd(df['close'])
rsi = calculate_rsi(df['close'])

# Initialize database
print("Creating trade database...")
db_file = 'trading_database.db'
if os.path.exists(db_file):
    os.remove(db_file)

conn = sqlite3.connect(db_file)
cursor = conn.cursor()

# Create trades table
cursor.execute('''
    CREATE TABLE trades (
        trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
        entry_datetime TEXT,
        entry_price REAL,
        entry_index INTEGER,
        signal_type TEXT,
        macd_score REAL,
        rsi_score REAL,
        morning_star INTEGER,
        hammer INTEGER,
        total_score REAL,
        exit_datetime TEXT,
        exit_price REAL,
        exit_index INTEGER,
        exit_reason TEXT,
        bars_held INTEGER,
        highest_price REAL,
        pnl_points REAL,
        pnl_percent REAL,
        profit_or_loss TEXT,
        exit_strategy TEXT
    )
''')

# Simulate real-time trading
print("Running real-time trading simulation...")
trades = []
next_trade_index = 0
total_signals = 0

for i in range(LOOKAHEAD + 3, len(df) - LOOKAHEAD - 1):
    # Check cooldown
    if i < next_trade_index:
        continue
    
    # Check if valid signal index
    if pd.isna(rsi[i]) or pd.isna(macd_line[i]) or pd.isna(macd_signal[i]):
        continue
    
    # Calculate score
    score, components = calculate_buy_score(i, macd_line.values, macd_signal.values, rsi.values, 
                                            opens, highs, lows, closes)
    
    # Entry signal: score >= 30 or MACD cross + RSI < 40
    macd_cross = (macd_line.values[i-1] < macd_signal.values[i-1] and 
                 macd_line.values[i] > macd_signal.values[i])
    rsi_value = rsi.values[i]
    
    has_signal = (score >= 30) or (macd_cross and rsi_value < 40)
    
    if not has_signal:
        continue
    
    total_signals += 1
    entry_index = i
    entry_price = opens[i]
    entry_datetime = datetimes[i]
    highest_price = entry_price
    
    # Determine primary signal
    signal_parts = []
    if components.get('MACD', 0) > 0:
        signal_parts.append('MACD')
    if components.get('RSI', 0) > 0:
        signal_parts.append('RSI')
    if components.get('Morning_Star', 0) > 0:
        signal_parts.append('Morning_Star')
    if components.get('Hammer', 0) > 0:
        signal_parts.append('Hammer')
    
    signal_type = '+'.join(signal_parts) if signal_parts else 'Score'
    
    # Simulate exit
    exit_price = None
    exit_index = None
    exit_datetime = None
    exit_reason = None
    pnl = None
    highest_price = entry_price
    
    for j in range(entry_index + 1, min(entry_index + LOOKAHEAD + 1, len(df))):
        current_high = highs[j]
        current_low = lows[j]
        highest_price = max(highest_price, current_high)
        
        # Check profit target (8 points)
        if current_high >= entry_price + PROFIT_TARGET:
            exit_price = entry_price + PROFIT_TARGET
            exit_index = j
            exit_datetime = datetimes[j]
            exit_reason = 'PROFIT_TARGET'
            break
        
        # Check trailing stop (2 points below peak)
        if current_low <= highest_price - TRAILING_STOP:
            exit_price = highest_price - TRAILING_STOP
            exit_index = j
            exit_datetime = datetimes[j]
            exit_reason = 'TRAILING_STOP'
            break
    
    # If no exit within lookahead, exit at last candle
    if exit_price is None:
        exit_price = closes[entry_index + LOOKAHEAD]
        exit_index = entry_index + LOOKAHEAD
        exit_datetime = datetimes[exit_index]
        exit_reason = 'LOOKAHEAD_END'
    
    # Calculate P&L
    pnl_points = exit_price - entry_price
    pnl_percent = (pnl_points / entry_price) * 100
    profit_or_loss = 'PROFIT' if pnl_points >= 0 else 'LOSS'
    
    # Store trade in database
    cursor.execute('''
        INSERT INTO trades (
            entry_datetime, entry_price, entry_index, signal_type,
            macd_score, rsi_score, morning_star, hammer, total_score,
            exit_datetime, exit_price, exit_index, exit_reason,
            bars_held, highest_price, pnl_points, pnl_percent,
            profit_or_loss, exit_strategy
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        str(entry_datetime), entry_price, entry_index, signal_type,
        components.get('MACD', 0), components.get('RSI', 0),
        components.get('Morning_Star', 0), components.get('Hammer', 0),
        score,
        str(exit_datetime), exit_price, exit_index, exit_reason,
        exit_index - entry_index, highest_price,
        pnl_points, pnl_percent,
        profit_or_loss, 'TRAILING_STOP' if exit_reason == 'TRAILING_STOP' else 'PROFIT_TARGET'
    ))
    
    trades.append({
        'entry_datetime': entry_datetime,
        'entry_price': entry_price,
        'signal_type': signal_type,
        'score': score,
        'exit_datetime': exit_datetime,
        'exit_price': exit_price,
        'exit_reason': exit_reason,
        'pnl_points': pnl_points,
        'pnl_percent': pnl_percent,
        'profit_or_loss': profit_or_loss
    })
    
    # Cooldown
    next_trade_index = entry_index + COOLDOWN

conn.commit()

print(f"\nTotal signals detected: {total_signals}")
print(f"Total trades executed: {len(trades)}")

# Generate summary statistics
if len(trades) > 0:
    trades_df = pd.DataFrame(trades)
    
    winning_trades = trades_df[trades_df['profit_or_loss'] == 'PROFIT']
    losing_trades = trades_df[trades_df['profit_or_loss'] == 'LOSS']
    
    win_rate = (len(winning_trades) / len(trades_df)) * 100 if len(trades_df) > 0 else 0
    avg_win = winning_trades['pnl_points'].mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades['pnl_points'].mean() if len(losing_trades) > 0 else 0
    total_pnl = trades_df['pnl_points'].sum()
    avg_pnl = trades_df['pnl_points'].mean()
    
    print(f"\n{'='*60}")
    print(f"TRADING PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    print(f"Total Trades: {len(trades_df)}")
    print(f"Winning Trades: {len(winning_trades)} ({win_rate:.2f}%)")
    print(f"Losing Trades: {len(losing_trades)} ({100-win_rate:.2f}%)")
    print(f"\nAverage Win: {avg_win:.4f} points")
    print(f"Average Loss: {avg_loss:.4f} points")
    if avg_loss != 0:
        print(f"Risk/Reward Ratio: {abs(avg_win/avg_loss):.2f}:1")
    print(f"\nTotal P&L: {total_pnl:.2f} points")
    print(f"Average P&L per Trade: {avg_pnl:.4f} points")
    print(f"Max Win: {trades_df['pnl_points'].max():.4f} points")
    print(f"Max Loss: {trades_df['pnl_points'].min():.4f} points")
    
    # Profit factor
    total_wins = winning_trades['pnl_points'].sum()
    total_losses = abs(losing_trades['pnl_points'].sum())
    profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
    print(f"Profit Factor: {profit_factor:.2f}")
    
    print(f"{'='*60}\n")
    
    # Export to CSV
    trades_df.to_csv('real_time_trading_results.csv', index=False)
    print("✓ Trades exported to: real_time_trading_results.csv")

# Export database data to CSV for easier analysis
print("Exporting database to CSV...")
query = "SELECT * FROM trades ORDER BY entry_datetime"
trades_db_df = pd.read_sql_query(query, conn)
trades_db_df.to_csv('trading_database_export.csv', index=False)
print("✓ Database exported to: trading_database_export.csv")

conn.close()
print(f"✓ Database saved to: {db_file}")

print("\nTrade database structure:")
print("Columns: trade_id, entry_datetime, entry_price, entry_index, signal_type,")
print("         macd_score, rsi_score, morning_star, hammer, total_score,")
print("         exit_datetime, exit_price, exit_index, exit_reason, bars_held,")
print("         highest_price, pnl_points, pnl_percent, profit_or_loss, exit_strategy")
