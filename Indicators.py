import pandas as pd
import ta

# -------- Load data --------
file = r'C:\Users\uttank\OneDrive\Desktop\Python\Analysis\Testing\NIFTY_50_25.csv'
df = pd.read_csv(file)

# Ensure proper datetime (markets respect time)
df["datetime"] = pd.to_datetime(df["datetime"])

# -------- RSI --------
df["RSI"] = ta.momentum.RSIIndicator(
    close=df["close"],
    window=14
).rsi()

df["RSI_buy"] = df["RSI"].apply(
    lambda x: "RSI_buy" if x < 40 else ""
)

# -------- MACD --------
macd = ta.trend.MACD(
    close=df["close"],
    window_slow=26,
    window_fast=12,
    window_sign=9
)

df["MACD"] = macd.macd()
df["MACD_signal"] = macd.macd_signal()
df["MACD_hist"] = macd.macd_diff()

# MACD upward crossover: yesterday below, today above
df["MACD_buy"] = (
    (df["MACD"] > df["MACD_signal"]) &
    (df["MACD"].shift(1) <= df["MACD_signal"].shift(1))
)

df["MACD_buy"] = df["MACD_buy"].apply(
    lambda x: "MACD_buy" if x else ""
)

# -------- Save new CSV --------
df.to_csv("TA.csv", index=False)
