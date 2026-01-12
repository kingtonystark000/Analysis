import yfinance as yf
import pandas as pd

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

ticker_symbol = "^NSEI"
ticker = yf.Ticker(ticker_symbol)

historical_data = ticker.history(period="2d", interval="1m")
historical_data.index = historical_data.index.tz_localize(None)

final_data = historical_data[['Open', 'High', 'Low', 'Close']]

# Lowercase the first row (column headers)
final_data.columns = final_data.columns.str.lower()

print(f"Summary of Historical Data for {ticker_symbol}:")
# print(final_data)

final_data.to_csv("nsei.csv")
