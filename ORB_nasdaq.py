import os
import requests
import json
import numpy as np
# Path to your CSV file
import calendar
import re
import shutil
from pathlib import Path
from dotenv import load_dotenv
from utils import utils
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
load_dotenv() 


from backtesting import Backtest, Strategy



def fetch_ticker_data_5(ticker, start_date, end_date):

    """
    Return: The list of price (candles) for a given ticker in the given date.
    Note: The maximun number of candles is 50 000. Make sure to enter start_date and end_date accordingly.

    Args:
        start_date (string): 2023-02-01.
        end_date (string): 2023-02-01.
        
    Returns:
        list of candlestick [{o,c,h,l,..},.....]
    """

    # Polygon.io API details
    API_KEY = os.getenv("API_KEY", "none")  # Replace with your Polygon.io API key

    start_milliseconds =    int(pd.to_datetime(f'{start_date} 09:00:00').timestamp() * 1000 )
    stop_milliseconds =  int(pd.to_datetime(f'{end_date} 21:30:00').timestamp() * 1000 )
    
    # adjusted=false means that it will not take into account the splits so It will always get the price as it was the given date.
    BASE_URL_30_MINUTES = "https://api.polygon.io/v2/aggs/ticker/{ticker}/range/5/minute/{start_date}/{end_date}?adjusted=true&sort=asc&apiKey={apiKey}&limit=50000"
    url_30_min = BASE_URL_30_MINUTES.format(ticker=ticker, start_date=start_milliseconds, end_date=stop_milliseconds,apiKey=API_KEY)
    

    
    #response = requests.get(url, params=params)
    response_30_min = requests.get(url_30_min)
    if response_30_min.status_code != 200:
        print(f"Error fetching data: {ticker}")
        return []
    
   
    data_30 = response_30_min.json().get("results", [])
    if not data_30:
        #raise Exception("No data returned for the specified period.")
        print(f'No data returned for the specified period; {start_date}, {ticker}')
        return  pd.DataFrame([])
    
    df =  pd.DataFrame(data_30) 
    
    df["date"] = pd.to_datetime(df["t"], unit='ms') - pd.Timedelta(hours=5) # -5 means New York timezone
    df["day"] =  df["date"].dt.date
    df['date_str'] = pd.to_datetime(pd.to_datetime(df["day"])).dt.strftime('%Y-%m-%d') 
    
    return df

def fetch_nasdaq_TQQQ(start_date, end_date):
    
    file_path = "nasdaq/TQQQ.parquet"
    dates = utils.generate_date_interval_to_fetch(start_date, end_date)
    df_nasdaq = pd.DataFrame([])
    for item in dates:
        (date1, date2) = item
        df = fetch_ticker_data_5("TQQQ", date1,  date2)
        df_nasdaq =  pd.concat([df_nasdaq, df], ignore_index=True)
        #df['date_str'] = pd.to_datetime(pd.to_datetime(df["t"])).dt.strftime('%Y-%m-%d') 
        print(df_nasdaq)
        print(date1, date2)
        
    
    if  Path(file_path).is_file():
        df = pd.read_parquet(file_path)
        df_updated = pd.concat([df, df_nasdaq], ignore_index=True)
        df_updated = df_updated.drop_duplicates()
        # Step 4: Write back to parquet (overwrite)
        # df_updated.to_parquet(file_path, index=False)
        # remove the source file since it’s merged
        utils.ingestDataInParquetFile( df_updated , file_path)
    
    
    
    return

def fetch_ticker_in_interval(ticker, start_date, end_date):
    
    file_path = f"nasdaq/{ticker}_v1.parquet"
    dates = utils.generate_date_interval_to_fetch(start_date, end_date)
    df_nasdaq = pd.DataFrame([])
    for item in dates:
        (date1, date2) = item
        df = fetch_ticker_data_5(ticker, date1,  date2)
        df_nasdaq =  pd.concat([df_nasdaq, df], ignore_index=True)
        #df['date_str'] = pd.to_datetime(pd.to_datetime(df["t"])).dt.strftime('%Y-%m-%d') 
        print(df_nasdaq)
        print(date1, date2)
        
    
    if  Path(file_path).is_file():
        df = pd.read_parquet(file_path)
        df_updated = pd.concat([df, df_nasdaq], ignore_index=True)
        df_updated = df_updated.drop_duplicates()
        # Step 4: Write back to parquet (overwrite)
        # df_updated.to_parquet(file_path, index=False)
        # remove the source file since it’s merged
        utils.ingestDataInParquetFile( df_updated , file_path)
    else :
        utils.ingestDataInParquetFile( df_nasdaq , file_path)
    
    return



fetch_ticker_in_interval('QQQ', '2021-01-13', '2026-01-13')



# ticker = 'TQQQ'
# file_path = f"nasdaq/{ticker}.parquet"
# df = pd.read_parquet(file_path)
# df.rename(columns={'o': 'Open', 'c': 'Close', 'h': 'High', 'l': 'Low', 'v': 'Volume', 'date': 'Date'}, inplace=True)
# # Convert Date column → datetime
# df['Date'] = pd.to_datetime(df['Date'])

# # Set index
# df = df.set_index('Date')

# # Ensure ascending order
# df = df.sort_index()

# # Keep only what Backtesting.py wants
# df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'date_str']]
# print(df)




# 2. create a strategy class
class OpeningRangeBreakout(Strategy):
    open_range_minutes = 5 # length of the opening range

    # what do we want to initialize at the beginning
    def init(self):
        self.opening_range_high = None   # opening range high
        self.opening_range_low  = None   # opening range low
        self.current_day        = None   # tracks the current day (YYYY-MM-DD)
        self.current_day_open = None
        self.bais = ""
        self.exit_minute_bar = pd.to_datetime("16:00").time()
        self.risk_prc = 0.01
        
    # wait every day is going to have a different opening range high and low
    def _reset_range(self, day):
        self.current_day        = day
        self.opening_range_high = None
        self.opening_range_low  = None
        self.bais = ""


    def next(self):
        # look at the data for the current bar
        
        t = self.data.index[-1]
        current_bar_date = t.date()
        current_bar_time = t.time()
        #print(current_bar_date)
        
        # detect when the date changes to a new day
        if current_bar_date != self.current_day:
            self._reset_range(current_bar_date)
        
        if current_bar_time == pd.to_datetime("09:30").time():
            self.opening_range_low  = self.data.Low[-1]
            self.opening_range_high = self.data.High[-1]
            self.current_day_open = self.data.Open[-1]
            if self.data.Close[-1] > self.current_day_open:
                self.bais = "long"
            elif self.data.Close[-1] < self.current_day_open:
                self.bais = "short"
            else:
                self.bais = "neutral"
                
            
        
        if self.position and t.time() == self.exit_minute_bar:
          print("closing out position")
          self.position.close()    
          
        if t.time() == pd.to_datetime("09:30").time():
            if not self.position:
                amount = self.equity * self.risk_prc
                size = amount // (self.opening_range_high - self.opening_range_low)
                if self.bais == "long":
                    print("going long")
                    stop_price = self.opening_range_low - 0.05
                    take_profit = self.opening_range_high + (self.opening_range_high - self.opening_range_low)*10
                    order = self.buy(size=size, sl=stop_price, tp=take_profit)
                elif self.bais == "short":
                    print("going short")
                    stop_price = self.opening_range_high + 0.05
                    take_profit = self.opening_range_low - (self.opening_range_high - self.opening_range_low)*10
                    order = self.sell(size=size,sl=stop_price, tp=take_profit)
                else:
                    print("doji, doing nothing")
                
            
          
  

# 1. read a csv of minute data, limit to the first month for now
#df = pd.read_csv("TQQQ_intraday_data.csv", parse_dates=['Date'], index_col='Date')
#df = df.loc['2016-01-05':'2016-02-05'] 

# 3. create a backtest, pass in data and the strategy you want to run on the data
# set various settings like account cash and commissions to strategy from paper
#bt = Backtest(df, OpeningRangeBreakout, cash=25_000)

# run the strategy
# stats = bt.run()
# bt.plot()
# print(stats)
# stats['_trades'].to_csv("trades.csv", index=False)
# stats['_equity_curve'].to_csv("equity_curve.csv", index=False)

# tra = pd.read_csv("trades.csv")
# x = tra[["Size","EntryBar","ExitBar","EntryPrice","ExitPrice","SL","TP","PnL","Commission","ReturnPct","EntryTime","ExitTime"]]
# print(x)
