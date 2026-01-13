#Import the needed libraries
import pandas as pd
import ta

#Load the filtered dataset genereated from the main.py file
filtered_df = pd.read_csv('AAPL_filtered_stocks.csv')

#Make sure the dataset is sorted before we apply the indicators
filtered_df = filtered_df.sort_values(by='Date')

#Add the technical indicators
#a) Simple Moving Average (SMA) for 5 days
#This calculates the average of the previous 5 closing prices
#and puts in a new column 'sma_5'.
filtered_df['sma_5'] = ta.trend.sma_indicator(close=filtered_df['Close'], window=5)

#b) Relative Strength Index (RSI) for 9 days
#This measures momentum 
filtered_df['rsi_9'] = ta.momentum.rsi(close=filtered_df['Close'], window=9)

#c) Moving Average Convergence Divergence (MACD) 
#macd_line: detects momentum changes. Tells us how strong trend is
#macd_signal: is the smoothed version of MACD that generates alerts. gives detail about potential changes
macd = ta.trend.macd(close=filtered_df['Close'], window_slow=15, window_fast=5)
filtered_df['macd_line'] =macd
filtered_df['macd_signal'] =ta.trend.macd_signal(close=filtered_df['Close'], window_slow=15, window_fast=5)

#d) Exponential Moving Average (EMA)
#This gives more weight to recent prices and detecks faster trend reversals
filtered_df['ema_7'] =ta.trend.ema_indicator(close=filtered_df['Close'], window=7)

#Print first 5 rows of the column
print("\nThe Technical indicators for AAPL has been successfully added :")
print(filtered_df[['Date', 'Ticker', 'Close', 'sma_5', 'rsi_9','macd_line', 'ema_7']].head())

#Print last 5 rows of the column
print("\nThe Technical indicators for AAPL has been successfully added:")
print(filtered_df[['Date', 'Ticker', 'Close', 'sma_5', 'rsi_9','macd_line', 'ema_7']].tail())

#Save the dataset with indicators
filtered_df.to_csv("stock_indicators_AAPL.csv", index=False)
print("\nDataset with the indicators has been saved as 'stock_indicators_AAPL'")
