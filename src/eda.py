import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Load the dataset to be used
df = pd.read_csv("stock_indicators_AAPL.csv")

#Convert the 'Date' to datetime
df['Date'] = pd.to_datetime(df['Date'])

#Line Plot - Closing Price Over Time
#This is to create a line chart to visualize how Apple's
#stock price has changed over time
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='Date', y='Close')
plt.title('AAPPL Closing Price Trends Over Time')
plt.xlabel('Date')
plt.ylabel('Closing Price ($)')
plt.xticks(rotation=45) #Rotate date labels
plt.tight_layout()
plt.savefig("lineplot_closingprice.png") #To save figures as images
plt.show()



#Histogram - The Distribution of Closing Prices
plt.figure(figsize=(10, 5))
sns.histplot(df['Close'], bins=30, kde=True)
plt.title('The Distribution of AAPL Closing Prices')
plt.xlabel('Closing Price ($)')
plt.tight_layout()
plt.savefig("histogram_closingprice.png") #To save figures as images
plt.show()

#Correlation Heatmap of Technical Indicators
plt.figure(figsize=(10,6))
numeric_cols = ['Close', 'sma_5', 'ema_7', 'rsi_9', 'macd_line']
correlation = df[numeric_cols].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Technical Indicators')
plt.tight_layout()
plt.savefig("correlation_heatmap.png") #To save figures as images
plt.show()

#Price Trend using SMA and EMA
plt.figure(figsize=(12,6))
#Line plot for the actual price closing price
sns.lineplot(data=df, x='Date', y='Close', label= 'Closing Price')
#The line plot for SMA (5 days)
sns.lineplot(data=df, x='Date', y='sma_5', label='SMA (5 days)')
#The line plot for EMA (7 days)
sns.lineplot(data=df, x='Date', y='ema_7', label='EMA (7 days)')
plt.title('Price Trend with SMA and EMA')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()#For clarity when displaying the labels for each line in the plot
plt.xticks(rotation=45)#Rotate labels
plt.tight_layout()
plt.savefig("price_trend.png") #To save figures as images
plt.show()


#Plot for RSI
plt.figure(figsize=(14,6))
plt.plot(df['Date'], df['rsi_9'], color='green', label='RSI (9-Day)')
plt.axhline(70, linestyle='--', color='red', label='Overbought Threshold')
plt.axhline(30, linestyle='--', color='blue', label='Oversold Threshold')
plt.title("RSI Over Time for AAPL")
plt.xlabel('Date')
plt.ylabel('RSI Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("rsi_trend.png") #To save figures as images.
plt.show()

#Plot MACD
plt.figure(figsize=(14,6))
plt.plot(df['Date'], df['macd_line'], label='MACD Line', color='blue')
plt.plot(df['Date'], df['macd_signal'], label='Signal Line', color='red', linestyle='--')
plt.title('MACD Trend for AAPL')
plt.xlabel('Date')
plt.ylabel('MACD Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("macd_trend.png") #To save figures as images.
plt.show()