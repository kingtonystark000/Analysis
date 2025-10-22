import pandas as pd
import ta

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

df = pd.read_csv('NIFTY_50_25.csv')

df = df.rename(columns={'Datetime': 'date', 'Close': 'close'})

df = df.sort_values('date')

rsi = ta.momentum.RSIIndicator(close=df['close'], window=14)
df['RSI'] = rsi.rsi()

macd = ta.trend.MACD(close=df['close'])
df['MACD'] = macd.macd()
df['Signal'] = macd.macd_signal()

df['RSI_signal'] = df['RSI'].apply(lambda x: 'buy' if x < 40 else '')

df['MACD_diff'] = df['MACD'] - df['Signal']
df['MACD_signal'] = ''

for i in range(1, len(df)):
    if df.loc[i - 1, 'MACD_diff'] < 0 and df.loc[i, 'MACD_diff'] > 0:
        if df.loc[i, 'MACD_diff'] > df.loc[i - 1, 'MACD_diff']:
            df.loc[i, 'MACD_signal'] = 'buy macd'

df.to_excel('TA.xlsx', index=False)

print("Signals added. Output saved to 'TA.xlsx'")
print("\n")
