
#Import library and Load the dataset
import pandas as pd

df = pd.read_csv('stocks.csv')


#View the first few rows
print("The dataset has been successfully loaded!")
print(df.head())

#View bottom five rows
print("\nBottom rows of the dataset")
print(df.tail())

#Check basic info
print("\nDataset Info:")
print(df.info())

#Check if there is any null values
print(df.isnull().sum())


#Convert the 'Date' column to datetime format
if 'Date' in df.columns:
    #to convert text-based date values to proper date time objects
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    #if any error has an invalid date, it get's turned to NaT, to prevent crashing.
    print("\n 'Date' column successfully converted to datetime.")
else:
    print("'Date' column not found. Please verify column names.")
    
#Sort the dataset by Date in chronological (ascending) order
df = df.sort_values(by='Date')

 #Drop rows where invalid dates become NaT
df = df.dropna(subset=['Date'])

# Defining which of the columns to keep
columns_to_keep = [ 'Ticker', 'Date', 'Close', 'Open', 'High', 'Low', 'Volume']
    
 #Filtering the dataset for specific stock tickers
#Choosing AAPL(Apple) 
if 'Ticker' in df.columns:
    filtered_df = df[df['Ticker'].isin(['AAPL'])].copy()
    #copy() makes a new copy to avoid chnages in the original dataset.
    filtered_df = filtered_df[columns_to_keep] #To keep only needed columns
    
    #This sorting the filtered dataset by Date
    filtered_df =filtered_df.sort_values(by='Date')
        
     #Reset index when done sorting and filtering
    filtered_df.reset_index(drop=True, inplace=True)
        
    #To save the filtered dataset
    filtered_df.to_csv('AAPL_filtered_stocks.csv', index=False)
    print("\n The filtered AAPL dataset is saved  as 'AAPL_filtered_stocks.csv'.")
    
    unique_tickers = df['Ticker'].unique()
    for ticker in unique_tickers:
        company_df = df[df['Ticker'] == ticker][columns_to_keep]  
        company_df = company_df.sort_values(by='Date')
        company_df.to_csv(f"{ticker}_data.csv", index=False)
        
        #Statistical overview of the numeric values only
        print(f"\n Statistical Overview for {ticker}:")
        print(company_df.describe(include = 'number'))
        print(f"Saved '{ticker}_data.csv' with {len(company_df)} rows.")  
else:
     print("'Ticker' column not found. This specific stock cannot be filtered.")
        
       