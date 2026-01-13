import pandas as pd
import ta 

#Load the filtered dataset genereated from the main.py file
df = pd.read_csv("NFLX_data.csv")

#To ensure the Date column is in the datetime format
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

#a) Simple Moving Average (SMA) for 5 Days
df['sma_5'] = ta.trend.sma_indicator(close=df['Close'], window=5)

#b) Exponential Moving Average (EMA) for 7 Days
df['ema_7'] = ta.trend.ema_indicator(close=df['Close'], window=7)

#c) Relative Strength Index (RSI) for 9 Days
df['rsi_9'] = ta.momentum.rsi(close=df['Close'], window=9)

#d) Moving Average Convergence Divergence (MACD)
macd = ta.trend.macd(close=df['Close'], window_slow=15, window_fast=5)
df['macd_line'] = macd
df['macd_signal'] = ta.trend.macd_signal(close=df['Close'], window_slow=15, window_fast=5)

#Drop any rows that has Nan values
df.dropna(inplace=True)

#Print first 5 rows of the column
print("\nThe Technical indicators for NFLX has been successfully added :")
print(df[['Date', 'Ticker', 'Close', 'sma_5', 'ema_7', 'rsi_9','macd_line', ]].head())

#Print last 5 rows of the column
print("\nThe Technical indicators has been successfully added:")
print(df[['Date', 'Ticker', 'Close', 'sma_5', 'ema_7', 'rsi_9','macd_line', ]].tail())

#Save the dataset with indicators
df.to_csv("stock_indicators_NFLX.csv", index=False)
print("\nDataset with the technical indicators for NFLX has been saved as 'stock_indicators_NFLX.csv'")