"""
COMBINED MULTI-STRATEGY SCORING SYSTEM
=====================================
Combines: MACD, RSI, Morning Star, Evening Star, Hammer Pattern
Produces a confidence score (0-100) for each signal
Tests if price actually goes UP after the signal
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ===================== LOAD DATA =====================
file = r'C:\Users\uttank\OneDrive\Desktop\Python\Analysis\Testing\TA.csv'
df = pd.read_csv(file)

# Convert datetime if needed
if 'datetime' in df.columns:
    df['datetime'] = pd.to_datetime(df['datetime'])

# Basic price data
open_data = df['open']
high_data = df['high']
low_data = df['low']
close_data = df['close']
time_data = df['datetime'] if 'datetime' in df.columns else df.index

# Signal columns (from existing analysis)
macd_data = df.get('MACD_buy', pd.Series([None] * len(df)))
rsi_data = df.get('RSI_buy', pd.Series([None] * len(df)))

# ===================== PARAMETERS =====================
LOOKAHEAD = 10
COOLDOWN = 5

# ===================== CANDLE ANATOMY =====================
df["body"] = abs(df["close"] - df["open"])
df["range"] = df["high"] - df["low"]
df["direction"] = np.where(df["close"] > df["open"], "bull", "bear")
df["body_ratio"] = df["body"] / df["range"].replace(0, np.nan)
df["lower_wick"] = df["open"] - df["low"]
df["upper_wick"] = df["high"] - df["close"]

# ===================== PATTERN DETECTION FUNCTIONS =====================

def detect_macd_signal(i):
    """MACD buy signal detected"""
    if pd.isna(macd_data.iloc[i]):
        return False
    return macd_data.iloc[i] == "MACD_buy"

def detect_rsi_signal(i):
    """RSI buy signal detected"""
    if pd.isna(rsi_data.iloc[i]):
        return False
    return rsi_data.iloc[i] == "RSI_buy"

def detect_macd_rsi_confirmation(i):
    """Check if MACD and RSI confirmations exist within past 10 candles"""
    if i < 10:
        return 0
    
    recent_macd = list(macd_data[i-10:i])
    recent_rsi = list(rsi_data[i-10:i])
    
    score = 0
    if "MACD_buy" in recent_macd:
        score += 1
    if "RSI_buy" in recent_rsi:
        score += 1
    
    return score

def detect_morning_star(i):
    """
    Morning Star Pattern:
    - 3 candles
    - First: Bearish, strong body
    - Second: Small body (gap down possible)
    - Third: Bullish, closes above midpoint of first candle
    Bullish reversal signal
    """
    if i < 2:
        return False
    
    try:
        c1 = df.iloc[i-2]
        c2 = df.iloc[i-1]
        c3 = df.iloc[i]
        
        is_morning_star = (
            c1["direction"] == "bear" and c1["body_ratio"] > 0.5 and
            c2["body_ratio"] < 0.3 and
            c3["direction"] == "bull" and
            c3["close"] > (c1["open"] + c1["close"]) / 2
        )
        return is_morning_star
    except:
        return False

def detect_evening_star(i):
    """
    Evening Star Pattern:
    - 3 candles
    - First: Bullish, strong body
    - Second: Small body (gap up possible)
    - Third: Bearish, closes below midpoint of first candle
    Bearish reversal signal (NOT used for buying)
    """
    if i < 2:
        return False
    
    try:
        c1 = df.iloc[i-2]
        c2 = df.iloc[i-1]
        c3 = df.iloc[i]
        
        is_evening_star = (
            c1["direction"] == "bull" and c1["body_ratio"] > 0.5 and
            c2["body_ratio"] < 0.3 and
            c3["direction"] == "bear" and
            c3["close"] < (c1["open"] + c1["close"]) / 2
        )
        return is_evening_star
    except:
        return False

def detect_hammer(i):
    """
    Hammer Pattern:
    - Small body (top half of range)
    - Long lower wick (at least 2x body length)
    - Minimal or no upper wick
    Bullish reversal signal
    """
    if i < 1:
        return False
    
    try:
        candle = df.iloc[i]
        
        # For a hammer to be valid, it should follow a downtrend or at least a bearish candle
        if i > 0:
            prev = df.iloc[i-1]
            in_downtrend = prev["direction"] == "bear"
        else:
            in_downtrend = True
        
        # Hammer criteria
        is_hammer = (
            candle["body"] > 0 and
            candle["body_ratio"] < 0.3 and  # small body (< 30% of range)
            candle["lower_wick"] > 2 * candle["body"] and  # long lower wick
            candle["upper_wick"] < candle["body"]  # minimal upper wick
        )
        
        return is_hammer and in_downtrend
    except:
        return False

def detect_inverted_hammer(i):
    """
    Inverted Hammer:
    - Small body (bottom half)
    - Long upper wick (at least 2x body length)
    - Minimal lower wick
    Bearish signal (NOT used for buying)
    """
    if i < 1:
        return False
    
    try:
        candle = df.iloc[i]
        
        is_inverted = (
            candle["body"] > 0 and
            candle["body_ratio"] < 0.3 and
            candle["upper_wick"] > 2 * candle["body"] and
            candle["lower_wick"] < candle["body"]
        )
        
        return is_inverted
    except:
        return False

# ===================== SCORING SYSTEM =====================

def calculate_buy_score(i):
    """
    Calculate composite score (0-100) based on all strategies
    Score components:
    - MACD signal: 20 points
    - RSI signal: 20 points
    - MACD+RSI confirmation: 15 points
    - Morning Star: 25 points
    - Hammer: 20 points
    Max score: 100
    """
    
    if i < 2 or i + LOOKAHEAD + 2 >= len(df):
        return 0, {}
    
    score = 0
    components = {}
    
    # MACD Signal
    if detect_macd_signal(i):
        score += 20
        components['MACD'] = 20
    
    # RSI Signal
    if detect_rsi_signal(i):
        score += 20
        components['RSI'] = 20
    
    # MACD + RSI Confirmation (bonus if both exist in recent window)
    confirmation = detect_macd_rsi_confirmation(i)
    if confirmation == 2:
        score += 15  # Both confirmed
        components['Confirmation'] = 15
    elif confirmation == 1:
        score += 8   # One confirmed
        components['Confirmation'] = 8
    
    # Morning Star Pattern
    if detect_morning_star(i):
        score += 25
        components['MorningStar'] = 25
    
    # Hammer Pattern
    if detect_hammer(i):
        score += 20
        components['Hammer'] = 20
    
    # Cap at 100
    score = min(score, 100)
    
    return score, components

# ===================== EVALUATION LOGIC =====================

def check_price_movement(i, entry_price):
    """
    Check if price went UP after entry in LOOKAHEAD periods
    Returns:
    - True if max high > entry (profit)
    - False otherwise (loss or flat)
    - Also returns the actual P&L
    """
    if i + LOOKAHEAD + 2 >= len(df):
        return None, None
    
    future_highs = list(high_data[i + 2 : i + 2 + LOOKAHEAD])
    future_closes = list(close_data[i + 2 : i + 2 + LOOKAHEAD])
    
    if not future_highs:
        return None, None
    
    max_high = max(future_highs)
    final_close = future_closes[-1]
    
    pnl = max_high - entry_price
    went_up = pnl > 0
    
    return went_up, pnl

# ===================== BACKTEST LOOP =====================

trade_records = []
next_trade_index = 0

for i in range(len(df)):
    
    # Enforce cooldown
    if i < next_trade_index:
        continue
    
    # Calculate score
    score, components = calculate_buy_score(i)
    
    # Only trade if score > 0
    if score == 0:
        continue
    
    # Get entry price and time
    entry_price = open_data.iloc[i + 1] if i + 1 < len(df) else None
    entry_time = time_data.iloc[i + 1] if i + 1 < len(df) else None
    
    if entry_price is None or pd.isna(entry_price):
        continue
    
    # Check if price went up
    went_up, pnl = check_price_movement(i, entry_price)
    
    if went_up is not None:
        trade_records.append({
            'entry_time': entry_time,
            'entry_price': entry_price,
            'score': score,
            'went_up': went_up,
            'pnl': pnl,
            'components': str(components)
        })
        
        next_trade_index = i + COOLDOWN

# ===================== CREATE RESULTS DATAFRAME =====================

results_df = pd.DataFrame(trade_records)

if len(results_df) > 0:
    results_df['went_up'] = results_df['went_up'].astype(bool)
    
    # ===================== OVERALL STATISTICS =====================
    print("\n" + "="*70)
    print("COMBINED MULTI-STRATEGY BACKTEST RESULTS")
    print("="*70)
    print(f"Total Signals Generated: {len(results_df)}")
    print(f"Signals with Price UP: {results_df['went_up'].sum()}")
    print(f"Signals with Price DOWN: {(~results_df['went_up']).sum()}")
    print(f"Success Rate: {(results_df['went_up'].sum() / len(results_df) * 100):.2f}%")
    
    print("\n" + "-"*70)
    print("SCORE DISTRIBUTION ANALYSIS")
    print("-"*70)
    
    # Analyze by score ranges
    score_ranges = [
        (90, 100, "Very Strong (90-100)"),
        (70, 89, "Strong (70-89)"),
        (50, 69, "Medium (50-69)"),
        (30, 49, "Weak (30-49)"),
        (1, 29, "Very Weak (1-29)")
    ]
    
    for min_score, max_score, label in score_ranges:
        subset = results_df[(results_df['score'] >= min_score) & (results_df['score'] <= max_score)]
        if len(subset) > 0:
            success = subset['went_up'].sum()
            success_rate = (success / len(subset) * 100)
            avg_pnl = subset['pnl'].mean()
            print(f"\n{label}:")
            print(f"  Count: {len(subset)}")
            print(f"  Success Rate: {success_rate:.2f}% ({success}/{len(subset)})")
            print(f"  Avg P&L: {avg_pnl:.4f}")
    
    print("\n" + "-"*70)
    print("P&L SUMMARY")
    print("-"*70)
    
    winning_trades = results_df[results_df['went_up'] == True]
    losing_trades = results_df[results_df['went_up'] == False]
    
    print(f"Total Winning Trades: {len(winning_trades)}")
    print(f"Total Losing Trades: {len(losing_trades)}")
    
    if len(winning_trades) > 0:
        print(f"Average Win: {winning_trades['pnl'].mean():.4f}")
        print(f"Max Win: {winning_trades['pnl'].max():.4f}")
    
    if len(losing_trades) > 0:
        print(f"Average Loss: {losing_trades['pnl'].mean():.4f}")
        print(f"Max Loss: {losing_trades['pnl'].min():.4f}")
    
    total_pnl = results_df['pnl'].sum()
    print(f"\nTotal P&L: {total_pnl:.4f}")
    print(f"Avg P&L per Trade: {results_df['pnl'].mean():.4f}")
    
    print("\n" + "="*70)
    print("SAMPLE TRADES (First 10)")
    print("="*70)
    
    display_cols = ['entry_time', 'score', 'went_up', 'pnl', 'components']
    print(results_df[display_cols].head(10).to_string())
    
    print("\n" + "="*70)
    print("BEST PERFORMING SIGNALS (By P&L)")
    print("="*70)
    
    best_trades = results_df.nlargest(5, 'pnl')
    print(best_trades[display_cols].to_string())
    
    print("\n" + "="*70)
    print("WORST PERFORMING SIGNALS (By P&L)")
    print("="*70)
    
    worst_trades = results_df.nsmallest(5, 'pnl')
    print(worst_trades[display_cols].to_string())
    
    # Save results to CSV
    results_df.to_csv('combined_strategy_results.csv', index=False)
    print(f"\n✓ Results saved to 'combined_strategy_results.csv'")
    
else:
    print("No signals generated. Check data and parameters.")

print("\n" + "="*70)
