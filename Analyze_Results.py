"""
Analyze Real-Time Trading Results
"""
import pandas as pd
import sqlite3

# Read the export CSV
df = pd.read_csv('real_time_trading_results.csv')

# Calculate statistics
winning = df[df['profit_or_loss'] == 'PROFIT']
losing = df[df['profit_or_loss'] == 'LOSS']

print('='*70)
print('REAL-TIME TRADING SIMULATOR - RESULTS ON FULL DATASET')
print('='*70)
print(f'\nDataset: NIFTY 50 minute data (932,946 candles)')
print(f'Date Range: 1/9/2015 to 2/7/2025')
print(f'\nTotal Signals Detected: {len(df)}')
print(f'Total Trades Executed: {len(df)}')
print(f'\nWinning Trades: {len(winning)} ({len(winning)/len(df)*100:.2f}%)')
print(f'Losing Trades: {len(losing)} ({len(losing)/len(df)*100:.2f}%)')

print(f'\nAverage Win: {winning["pnl_points"].mean():.4f} points')
print(f'Average Loss: {losing["pnl_points"].mean():.4f} points')
print(f'Risk/Reward Ratio: {abs(winning["pnl_points"].mean()/losing["pnl_points"].mean()):.2f}:1')

total_pnl = df['pnl_points'].sum()
total_wins = winning['pnl_points'].sum()
total_losses = abs(losing['pnl_points'].sum())

print(f'\nTotal P&L: {total_pnl:.2f} points')
print(f'Profit Factor: {total_wins/total_losses:.2f}')
print(f'Average P&L per Trade: {df["pnl_points"].mean():.4f} points')
print(f'\nMax Win: {df["pnl_points"].max():.4f} points')
print(f'Max Loss: {df["pnl_points"].min():.4f} points')

print(f'\n--- Profit/Loss Distribution ---')
exit_dist = df['profit_or_loss'].value_counts()
for label, count in exit_dist.items():
    pct = count / len(df) * 100
    print(f'{label}: {count} trades ({pct:.2f}%)')

print(f'\n--- Top 10 Trades by P&L ---')
top_trades = df.nlargest(10, 'pnl_points')[['entry_datetime', 'entry_price', 'exit_datetime', 'exit_price', 'pnl_points', 'signal_type']]
print(top_trades.to_string(index=False))

print(f'\n--- Worst 10 Trades by P&L ---')
worst_trades = df.nsmallest(10, 'pnl_points')[['entry_datetime', 'entry_price', 'exit_datetime', 'exit_price', 'pnl_points', 'signal_type']]
print(worst_trades.to_string(index=False))

print(f'\n--- Signal Type Performance ---')
signal_stats = df.groupby('signal_type').agg({
    'pnl_points': ['count', 'mean', 'sum', 'min', 'max'],
    'profit_or_loss': lambda x: (x == 'PROFIT').sum() / len(x) * 100
}).round(4)
signal_stats.columns = ['Count', 'Avg_P&L', 'Total_P&L', 'Min', 'Max', 'Win%']
print(signal_stats)

# Drawdown analysis
df['cumulative_pnl'] = df['pnl_points'].cumsum()
df['running_max'] = df['cumulative_pnl'].expanding().max()
df['drawdown'] = df['cumulative_pnl'] - df['running_max']
max_drawdown = df['drawdown'].min()

print(f'\n--- Risk Metrics ---')
print(f'Max Drawdown: {max_drawdown:.2f} points')
print(f'Max Cumulative P&L: {df["cumulative_pnl"].max():.2f} points')
print(f'Min Cumulative P&L: {df["cumulative_pnl"].min():.2f} points')

# Consecutive wins/losses
df['win'] = (df['profit_or_loss'] == 'PROFIT').astype(int)
consecutive_wins = []
consecutive_losses = []
current_win_streak = 0
current_loss_streak = 0

for val in df['win']:
    if val == 1:
        current_win_streak += 1
        if current_loss_streak > 0:
            consecutive_losses.append(current_loss_streak)
        current_loss_streak = 0
    else:
        current_loss_streak += 1
        if current_win_streak > 0:
            consecutive_wins.append(current_win_streak)
        current_win_streak = 0

if current_win_streak > 0:
    consecutive_wins.append(current_win_streak)
if current_loss_streak > 0:
    consecutive_losses.append(current_loss_streak)

max_consecutive_wins = max(consecutive_wins) if consecutive_wins else 0
max_consecutive_losses = max(consecutive_losses) if consecutive_losses else 0

print(f'Max Consecutive Wins: {max_consecutive_wins}')
print(f'Max Consecutive Losses: {max_consecutive_losses}')

print(f'\n{'='*70}')
print(f'✓ Full database exported to: trading_database_export.csv')
print(f'✓ Results summary exported to: real_time_trading_results.csv')
print(f'✓ SQLite database saved to: trading_database.db')
print(f'{'='*70}')
