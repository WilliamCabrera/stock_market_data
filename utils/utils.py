from datetime import datetime, timedelta, time, date
import pandas as pd
import requests
import os
import json
import numpy as np
# Path to your CSV file
import calendar
import re
import shutil
from pathlib import Path
from dotenv import load_dotenv
load_dotenv() 

#import sys
#sys.path.insert(0, os.path.abspath("../pandas_ta"))
#import overlap as ovl
#start_milliseconds =    int(pd.to_datetime(f'{start_date} 09:00:00').timestamp() * 1000 )
#stop_milliseconds =  int(pd.to_datetime(f'{start_date} 21:30:00').timestamp() * 1000 )


def last_day_of_month(year: int, month: int) -> int:
    """
    Return the last day of the month.

    Args:
        year (Int): year.
        month (Int): month.
        
    Returns:
        Int: the last day of the month
    """
    return calendar.monthrange(year, month)[1]
    
# return tuple with 2 month intervals [('YYYY-MM-DD','YYYY-MM-DD'), .......]
def month_ranges(start_date_str, end_date_str=None, monthdelta = 2):
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date() if end_date_str else date.today()
    
    if start_date > end_date:
        raise ValueError("start_date must be <= end_date")
    
    result = []
    current = start_date

    # ── First tuple if start date is not the first of the month
    if current.day != 1:
        # Last day of current month
        year, month = current.year, current.month
        if month == 12:
            first_next_month = date(year + 1, 1, 1)
        else:
            first_next_month = date(year, month + 1, 1)
        last_day_of_month = first_next_month - timedelta(days=1)
        
        result.append((
            current.strftime("%Y-%m-%d"),
            min(last_day_of_month, end_date).strftime("%Y-%m-%d")
        ))
        
        # Move current to first day of next month
        current = last_day_of_month + timedelta(days=1)
    
    # ── Two-month interval tuples
    while current <= end_date:
        year, month = current.year, current.month
        first_day = date(year, month, 1)
        
        # Compute last day of next month
        if month == 12:
            after_next = date(year + 1, 2, 1)
        elif month == 11:
            after_next = date(year + 1, 1, 1)
        else:
            after_next = date(year, month + monthdelta, 1)
            
        last_day_next_month = after_next - timedelta(days=1)
        
        tuple_end = min(last_day_next_month, end_date)
        result.append((
            first_day.strftime("%Y-%m-%d"),
            tuple_end.strftime("%Y-%m-%d")
        ))
        
        # Move to the month after next (two-month step)
        current = last_day_next_month + timedelta(days=1)
    
    return result

def generate_date_interval_to_fetch(start, end = datetime.today().strftime("%Y-%m-%d")):
    """
    Generate the dates need to make calls to the API.

    Args:
        start (string): The beging date.
        

    Returns:
        List: date list. dates as string
    """
    result = [] 
    start_date = datetime.strptime(start, "%Y-%m-%d")
    end_date = datetime.strptime(end, "%Y-%m-%d")
    total_days = 40;
    
    difference_in_days = (end_date - start_date).days
    
    if(difference_in_days <= total_days):
        return [(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))]
    
    current_date = start_date
    
    while current_date < end_date - timedelta(days=total_days):
        dates =  (current_date.strftime('%Y-%m-%d'), (current_date + timedelta(days=total_days)).strftime('%Y-%m-%d'))
        result.append(dates)
        current_date = current_date + timedelta(days=total_days + 1)
        
    result.append((current_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))


       

    return result

def generate_date_interval_to_fetch_30_minutes(start, end = datetime.today().strftime("%Y-%m-%d")):
    """
    Generate the dates need to make calls to the API.

    Args:
        start (string): The beging date.
        

    Returns:
        List: date list. dates as string
    """
    result = [] 
    start_date = datetime.strptime(start, "%Y-%m-%d")
    end_date = datetime.strptime(end, "%Y-%m-%d")
    
    
    difference_in_days = (end_date - start_date).days
    
    if(difference_in_days <= 199):
        return [(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))]
    
    current_date = start_date
    
    total_days = 365*4;
    
    while current_date < end_date - timedelta(days=total_days):
        dates =  (current_date.strftime('%Y-%m-%d'), (current_date + timedelta(days=total_days)).strftime('%Y-%m-%d'))
        result.append(dates)
        current_date = current_date + timedelta(days=total_days + 1)
        
    result.append((current_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))


       

    return result

def fetch_ticker_data_30(ticker, start_date, end_date):

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
    BASE_URL_30_MINUTES = "https://api.polygon.io/v2/aggs/ticker/{ticker}/range/30/minute/{start_date}/{end_date}?adjusted=false&sort=asc&apiKey={apiKey}&limit=50000"
    url_30_min = BASE_URL_30_MINUTES.format(ticker=ticker, start_date=start_milliseconds, end_date=stop_milliseconds,apiKey=API_KEY)
    
    #print(url_30_min)
    
    #response = requests.get(url, params=params)
    response_30_min = requests.get(url_30_min)
    if response_30_min.status_code != 200:
        print(f"Error fetching data: {ticker}")
        return []
    
   
    data_30 = response_30_min.json().get("results", [])
    if not data_30:
        #raise Exception("No data returned for the specified period.")
        print(f'No data returned for the specified period; {start_date}, {ticker}')
        return []
    
   
    
    return data_30

def fetch_ticker_data_5min(ticker, start_date, end_date):

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

    start_milliseconds =    int(pd.to_datetime(f'{start_date} 00:00:00').timestamp() * 1000 )
    stop_milliseconds =  int(pd.to_datetime(f'{end_date} 23:59:00').timestamp() * 1000 )
    
    # adjusted=false means that it will not take into account the splits so It will always get the price as it was the given date.
    BASE_URL_30_MINUTES = "https://api.polygon.io/v2/aggs/ticker/{ticker}/range/5/minute/{start_date}/{end_date}?adjusted=false&sort=asc&apiKey={apiKey}&limit=50000"
    url_30_min = BASE_URL_30_MINUTES.format(ticker=ticker, start_date=start_milliseconds, end_date=stop_milliseconds,apiKey=API_KEY)
    
    #print(url_30_min)
    
    #response = requests.get(url, params=params)
    response_30_min = requests.get(url_30_min)
    if response_30_min.status_code != 200:
        print(f"Error fetching data: {ticker}")
        return []
    
   
    data_30 = response_30_min.json().get("results", [])
    if not data_30:
        #raise Exception("No data returned for the specified period.")
        print(f'No data returned for the specified period; {start_date}, {ticker}')
        return []
    
   
    
    return data_30

def fetch_ticker_data_1min(ticker, start_date, end_date, adjusted=False):

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

    start_milliseconds =    int(pd.to_datetime(f'{start_date} 04:00:00').timestamp() * 1000 )
    stop_milliseconds =  int(pd.to_datetime(f'{end_date} 23:59:00').timestamp() * 1000 )
    
    adjusted = False
    # adjusted=false means that it will not take into account the splits so It will always get the price as it was the given date.
    BASE_URL_30_MINUTES = "https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{start_date}/{end_date}?adjusted={adjusted}&sort=asc&apiKey={apiKey}&limit=50000"
    url_30_min = BASE_URL_30_MINUTES.format(ticker=ticker, start_date=start_milliseconds, end_date=stop_milliseconds, adjusted = adjusted,apiKey=API_KEY)
    
    #print(url_30_min)
    
    
    #response = requests.get(url, params=params)
    response_30_min = requests.get(url_30_min)
    if response_30_min.status_code != 200:
        print(f"Error fetching data: {ticker}")
        return []
    
   
    data_30 = response_30_min.json().get("results", [])
    if not data_30:
        #raise Exception("No data returned for the specified period.")
        print(f'No data returned for the specified period; {start_date}, {ticker}')
        return []
    
   
    
    return data_30

def fetch_ticker_data_4H(ticker, start_date, end_date):

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

    start_milliseconds =    int(pd.to_datetime(f'{start_date} 00:00:00').timestamp() * 1000 )
    stop_milliseconds =  int(pd.to_datetime(f'{end_date} 23:59:00').timestamp() * 1000 )
    
    # adjusted=false means that it will not take into account the splits so It will always get the price as it was the given date.
    BASE_URL_4_H = "https://api.polygon.io/v2/aggs/ticker/{ticker}/range/4/hour/{start_date}/{end_date}?adjusted=false&sort=asc&apiKey={apiKey}&limit=50000"
    url_4_h = BASE_URL_4_H.format(ticker=ticker, start_date=start_milliseconds, end_date=stop_milliseconds,apiKey=API_KEY)
    
    print(url_4_h)
    
    response_30_min = requests.get(url_4_h)
    if response_30_min.status_code != 200:
        print(f"Error fetching data: {ticker}")
        return []
    
   
    data_30 = response_30_min.json().get("results", [])
    if not data_30:
        #raise Exception("No data returned for the specified period.")
        print(f'No data returned for the specified period; {start_date}, {ticker}')
        return []
    
   
    
    return data_30

def fetch_ticker_data_daily(ticker, start_date, end_date):

    # Polygon.io API details
    API_KEY = os.getenv("API_KEY", "none")  # Replace with your Polygon.io API key
    url_daily = f'https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}?adjusted=false&sort=asc&apiKey={API_KEY}&limit=50000'
    #print("*********** daily *************")
    #print(url_daily)
    response = requests.get(url_daily)
    if response.status_code != 200 or response.status_code != 200 :
        raise Exception(f"Error fetching data: {response.json()}")
    
   
    data= response.json().get("results", [])
    if not data:
        #raise Exception("No data returned for the specified period.")
        print(f'No data returned for the specified period; {start_date}, {ticker}')
        return []
    
    
   
    return data

def process_data_30_minutes(data):
    
    df = pd.DataFrame(data)

    try:

        df.rename(columns={'o': 'open', 'c': 'close', 'h': 'high', 'l': 'low', 'v': 'volume', 't': 'time'}, inplace=True)
        # Convert timestamp to datetime
      

        if df.empty :
            return df

        df["date"] =  pd.to_datetime(df["time"], unit='ms', utc=True) # pd.to_datetime(df["time"], unit='ms') - pd.Timedelta(hours=5) # -5 means New York timezone 
        df["date"] = df["date"].dt.tz_convert("America/New_York")
        df["day"] =  df["date"].dt.date #pd.to_datetime(pd.to_datetime(df["time"], unit='ms').dt.date).dt.strftime('%Y-%m-%d') 
        df["time"] = df["date"].dt.time
        df["v"] = df["volume"]
        
        
        # Identify pre-market (before 9:30 AM)
        df["is_premarket"] = df["date"].dt.time < time(9, 30)
        df["is_market_hours"] =( df["date"].dt.time >= time(9, 30)) & (df["date"].dt.time < time(16,0))
        df["is_after_hours"] = df["date"].dt.time >= time(16,0)
        
        
        
        # getting high in Pre-Market
        before_930 = df[df['time'] < time(9, 30)]
        before_930 = before_930.sort_values(by="time").reset_index(drop=True)
        highest_pre_market = before_930.groupby('day', as_index=False)['high'].max()
        highest_pre_market.rename(columns={"high": "high_pm"}, inplace=True)
        
        lowest_pre_market = before_930.groupby('day', as_index=False)['low'].min()
        lowest_pre_market.rename(columns={"low": "low_pm"}, inplace=True)
        
        first_pre_market_open = (
            before_930
            .sort_values(['day', 'time'])
            .groupby('day', as_index=False)
            .first()[['day', 'open']]
            .rename(columns={'open': 'previous_close'})
        )
        
        # Filtrar las filas donde el año es 2026
        #df_2026 = before_930[before_930['date'].dt.year == 2026]
        #print(df_2026)
        
        # getting high in market hours
        # Código corregido:
        after_930 = df[(df['time'] >= time(9, 30)) & (df['time'] < time(16, 0))]
        after_930 = after_930.sort_values(by="time").reset_index(drop=True)
        highest_market_hours = after_930.groupby('day', as_index=False)['high'].max()
        highest_market_hours.rename(columns={"high": "high_mh"}, inplace=True)
        
        lowest_market_hours = after_930.groupby('day', as_index=False)['low'].min()
        lowest_market_hours.rename(columns={"low": "low_mh"}, inplace=True)
        
        after_hours = df[df['time'] >= time(16, 0)]
        after_hours = after_hours.sort_values(by="time").reset_index(drop=True)
        after_hour_daily_ohlc = after_hours.groupby('day', as_index=False).agg(  {
                "open": "first",    # Apertura: Toma el precio de la primera entrada del día
                "close": "last",     # Cierre: Toma el precio de la última entra
                "high": "max",     # High is the highest high of the day
                "low": "min",     # Low is the lowest low of the day
                "volume": "sum"      # Volume is summed up for the day
            }).reset_index()
        after_hour_daily_ohlc['ah_range_perc'] =  100 *  ((after_hour_daily_ohlc['high'] -  after_hour_daily_ohlc['open'])/after_hour_daily_ohlc['open'])
        after_hour_daily_ohlc['ah_range'] =  (after_hour_daily_ohlc['high'] -  after_hour_daily_ohlc['open'])
        after_hour_daily_ohlc.rename(columns={'open': 'ah_open', 'close': 'ah_close', 'high': 'ah_high', 'low': 'ah_low', 'volume': 'ah_volume'}, inplace=True)
        after_hour_daily_ohlc = after_hour_daily_ohlc[["ah_open","ah_close","ah_range_perc","ah_range", "day"]]
        
        
        highest_after_hours = after_hours.groupby('day', as_index=False)['high'].max()
        highest_after_hours.rename(columns={"high": "ah_high"}, inplace=True)
        lowest_after_hours = after_hours.groupby('day', as_index=False)['low'].min()
        lowest_after_hours.rename(columns={"low": "ah_low"}, inplace=True)

        # Get the open at 9:30 AM and close at 4:00 PM
        daily_open =  df[df["time"] == time(9, 30)].groupby("day")["open"].first().reset_index()
        #daily_open ["open"] = open
        daily_close = df[df["time"] == time(16, 0)].groupby("day")["open"].first().reset_index()
        daily_close.rename(columns={"open": "close"}, inplace=True)
     
        # Aggregate to daily OHLC and sum volume
        daily_ohlc = df.groupby("day").agg(
            {
                "high": "max",     # High is the highest high of the day
                "low": "min",     # Low is the lowest low of the day
                "volume": "sum"      # Volume is summed up for the day
            }
        ).reset_index()
            
        # Sum pre-market volume
        daily_premarket_volume = df[df["is_premarket"]].groupby("day")["v"].sum().reset_index()
        daily_premarket_volume.rename(columns={"v": "premarket_volume"}, inplace=True)
        # Sum pre-market volume
        market_hours_volume = df[df["is_market_hours"]].groupby("day")["v"].sum().reset_index()
        market_hours_volume.rename(columns={"v": "market_hours_volume"}, inplace=True)
        
        after_hours_volume = df[df["is_after_hours"]].groupby("day")["v"].sum().reset_index()
        after_hours_volume.rename(columns={"v": "ah_volume"}, inplace=True)
        
        
        # Merge open, close, and premarket volume with daily OHLC
        daily_ohlc = daily_ohlc.merge(daily_open, on="day", how="left")
        daily_ohlc = daily_ohlc.merge(daily_close, on="day", how="left")
        daily_ohlc = daily_ohlc.merge(daily_premarket_volume, on="day", how="left").fillna(0)
        daily_ohlc = daily_ohlc.merge(market_hours_volume, on="day", how="left").fillna(0)
        daily_ohlc = daily_ohlc.merge(highest_market_hours, on="day", how="left").fillna(0)
        daily_ohlc = daily_ohlc.merge(highest_pre_market, on="day", how="left").fillna(0)
        daily_ohlc = daily_ohlc.merge(first_pre_market_open, on="day", how="left").fillna(0)
        daily_ohlc = daily_ohlc.merge(lowest_pre_market, on="day", how="left").fillna(0)
        daily_ohlc = daily_ohlc.merge(after_hour_daily_ohlc, on="day", how="left").fillna(0)
        daily_ohlc = daily_ohlc.merge(highest_after_hours, on="day", how="left").fillna(0)
        daily_ohlc = daily_ohlc.merge(lowest_after_hours, on="day", how="left").fillna(0)
        daily_ohlc = daily_ohlc.merge(lowest_market_hours, on="day", how="left").fillna(0)
        daily_ohlc = daily_ohlc.merge(after_hours_volume, on="day", how="left").fillna(0)
        daily_ohlc['highest_in_pm'] = daily_ohlc['high_pm'] >= daily_ohlc['high_mh']
        daily_ohlc['time'] = pd.to_datetime(daily_ohlc['day']).astype('int64') // 10**6
        daily_ohlc['date_str'] = pd.to_datetime(pd.to_datetime(daily_ohlc["day"])).dt.strftime('%Y-%m-%d') 
        
        daily_ohlc['high'] =  np.maximum(daily_ohlc['high_mh'], daily_ohlc['open'])
        daily_ohlc['low'] =  np.minimum(daily_ohlc['low_mh'], daily_ohlc['low_pm'])
         
        daily_ohlc.fillna(
                    {
                        'gap': -1,              # Replace NaN in 'gap' column with -1
                        'volume': -1,           # Replace NaN in 'volume' column with -1
                        'gap_perc': -1,           # Replace NaN in 'volume' column with -1
                        'premarket_volume': -1,           # Replace NaN in 'volume' column with -1
                        'previous_close': -1,           # Replace NaN in 'volume' column with -1
                        'ah_range': 0,           # Replace NaN in 'volume' column with -1
                        'ah_range_perc': 0 ,        # Replace NaN in 'market_cap' column with 0 (just for example),
                        "ah_open":-1,
                        "ah_close":-1,
                        "ah_high":-1,
                        "ah_low":-1,
                        "ah_volume":-1,
                        "low_pm":-1,
                        "high_pm":-1,
                        "high_mh":-1,
                        "market_hours_volume":-1
                    },
                    inplace=True
                )
        
       
        return  daily_ohlc
    except  Exception as e:
        print(f' error: {e}')

        return None

def raw_data_to_dataframe(data):

   
    df = pd.DataFrame(data)

    try:
        df.rename(columns={'o': 'open', 'c': 'close', 'h': 'high', 'l': 'low', 'v': 'volume', 't': 'time'}, inplace=True)
        # Convert timestamp to datetime
        df["date"] = pd.to_datetime(df["time"], unit='ms', utc=True)  # -5 means New York timezone
        df["date"] = df["date"].dt.tz_convert("America/New_York")
        df["day"] =  df["date"].dt.date #pd.to_datetime(pd.to_datetime(df["time"], unit='ms').dt.date).dt.strftime('%Y-%m-%d') 
        
        #df['dayX'] = pd.to_datetime(df['day'])
        #df["day1"] =  pd.to_datetime(pd.to_datetime(df["time"], unit='ms').dt.date).dt.strftime('%Y-%m-%d') 

    except  Exception as e:
       print(f' error: {e}')
    
    return df

def indexOf(list, ticker):

    indices = [i for i, x in enumerate(list) if x == ticker]
    if len(indices) >= 0 :
        return indices[0]
    return -1


def fetch_split(ticker):

   # Polygon.io API details
    API_KEY = os.getenv("API_KEY", "none")  # Replace with your Polygon.io API key
    url_splits = f'https://api.massive.com/stocks/v1/splits?ticker={ticker}&limit=5000&sort=execution_date.desc&apiKey={API_KEY}'
    # 

    response = requests.get(url_splits)
    if response.status_code != 200 or response.status_code != 200 :
        raise Exception(f"Error fetching data: {response.json()}")
    
   
    data = response.json().get("results", [])
    if not data:
        #raise Exception("No data returned for the specified period.")
        print(f'No  split data returned : {ticker}')
        return []
    
    return data

def sync_data_with_prev_day_close(df):
    
    try:
        
        df['split_adjust_factor'] =  np.nan
        df['split_date_str'] = ""
        len_df = len(df)
        
        for i in range(len_df):
            row = df.loc[i]
            date_str = row['date_str']
            ticker = row['ticker']
        
            if np.isnan(row['split_adjust_factor']): 
                res = fetch_split(ticker)
                
                if res is not None or len(res) > 0:
                    for j in range(len(res)):
                        
                        time_param = int(
                            pd.to_datetime(res[j]['execution_date'], utc=True).timestamp() * 1000
                        ) if res and len(res) > 0 else -1
                        
                        df.loc[
                        (df['ticker'] == ticker) & (df['time'] < time_param),
                        'split_adjust_factor'
                        ] = res[j]['historical_adjustment_factor'] if res and len(res) > 0 else 1
                        
                        df.loc[
                        (df['ticker'] == ticker) & (df['time'] <= time_param),
                        'split_date_str'
                        ] =  f'{pd.to_datetime(time_param, unit='ms', utc=True).strftime('%Y-%m-%d')}'
                    
                    
                    df.loc[
                        (df['ticker'] == ticker) & ( np.isnan(df['split_adjust_factor'])),
                        'split_adjust_factor'
                        ] =  1
                        
                    
                    df['split_date_str'] = df['split_date_str'].str.strip()
                    df['date_str'] = df['date_str'].str.strip()
                    idx =  (df['ticker'] == ticker) & ((df['split_date_str']) == df['date_str'])
                    df.loc[idx, 'previous_close'] = df.loc[idx, 'open']
                    
    except Exception as e:
        
        print(f' ======= sync_data_with_prev_day_close : {e}')
        
                
    return df

def process_pipeline(ticker_array, dates, apiConnectionParams):

    """
    This function take ticker list and dates to fetch and inject the corresponding data into the database

    Args:
        ticker_array (list of string): tickers.
        dates (list of dates (as string)): dates use to create the API calls.
        

    Returns:
        void

    """
    
    STOCK_MARKET_PARQUET_PATH = os.getenv("STOCK_MARKET_PARQUET_PATH", "none") 
   
    filtered = [s for s in ticker_array if re.fullmatch(r"[A-Za-z]+", s)]
    #filtered = ['ACP$A']
    df = pd.DataFrame()
    list_len =  len(filtered)
   
   
    for index in range(0,list_len):
        
        ticker = filtered[index]
        print(f'{ticker}: {index}/{list_len}')
        df = pd.DataFrame()
        
        for item in dates:
            (date1, date2) = item

            data30 = fetch_ticker_data_30(ticker, date1, date2)
            print(f'ticker: {ticker} - from: {date1} to {date2}')
            processedData = process_data_30_minutes(data30)
            dataDaily = raw_data_to_dataframe(fetch_ticker_data_daily(ticker, date1, date2))
            
            if processedData.empty == False and dataDaily.empty == False:
                            
                try:
                    dataDaily = dataDaily.drop(columns=['high', 'low','date', 'time', 'vw', 'volume','n'])
                    processedData = processedData.merge(dataDaily, on="day", how="left")
                    processedData['ticker'] = ticker
                    processedData['date_str'] = pd.to_datetime(pd.to_datetime(processedData["day"])).dt.strftime('%Y-%m-%d') 
                    #processedData['stock_float'] = -1
                    #processedData['market_cap'] = -1
                    processedData = fetch_stock_float_and_marketcap_from_df(processedData)    
                   
                except  Exception as e:
                    print(f' error  droping daily columns or merging dataframes: {e}')

                if df.empty:
                    df = processedData
                else:
                    df =  pd.concat([df, processedData], ignore_index=True)
        
        if df.empty == False:

            try:

                 
                # Sort by time to ensure correct order
                df = df.sort_values(by="day").reset_index(drop=True)
                # Compute the gap (current day's open - previous day's close)
                df['previous_close'] = df['close'].shift(1)
                df['gap'] = df['open'] - df['previous_close']
                df['daily_range'] = (   df['high'] - df['low']) * 100 / df['open']
                df['gap_perc'] = np.where( df['previous_close'] > 0, (df['open'] - df['previous_close']) *100 / df['previous_close'], 0 )
                # Fill NaN gaps with 0 for the first day
                df['previous_close'] = df['previous_close'].fillna(0)
                df['gap'] = df['gap'].fillna(0)
                #df['gap_perc'].fillna(0, inplace=True)
                #df['gap_perc'].fillna(0, inplace=True)
                df['ticker'] = ticker
                #df = df.drop(columns=['previous_close'])
                df['date_str'] = pd.to_datetime(pd.to_datetime(df["day"])).dt.strftime('%Y-%m-%d') 
                        
                df = df.drop(columns=['day'])
                print(df)
          
                #ingestData(df,apiConnectionParams)
                ingestDataInParquetFile(df, f'{STOCK_MARKET_PARQUET_PATH}/{ticker}.parquet')
            except  Exception as e:
                print( f'error in process of injecting data: {e}')
        
        if index == list_len -1:
            with open(f'{STOCK_MARKET_PARQUET_PATH}/last_ticker.txt', 'w') as f:
                f.write(f'------done----------\n')
            
        else:
            # save index of last ticker processed
            with open(f'{STOCK_MARKET_PARQUET_PATH}/last_ticker.txt', 'w') as f:
                f.write(f'{ticker}\n')
    return

def slice_ticker_list_by_ticker(tickers ,start_ticker):
    """
    This function return the sub list of tickers staring by the given ticker.

    Args:
        tickers (list of string): tickers.
        start_ticker (string): ticker to start in in case the user do not want to process the whole list of tickers.
       
    Returns:
        ticker list : list of string

    """
    
    ticker_array = tickers
    if start_ticker != None:
       index = indexOf(ticker_array, start_ticker)
       if index > 0:
            ticker_array = ticker_array[index:]

    return 

def update_stock_data_db_from_date(tickers , start_date, end_date = datetime.today().strftime("%Y-%m-%d") , apiConnectionParams = None):
    """
    This function inject data corresponding from a given date to today for the listed tickers.

    Args:
        tickers (list of string): tickers.
        start_date (string): starting date
        apiConnectionParams: {POSTGREST_H, POSTGREST_P, POSTGREST_TOKEN} , postgRest api connection params

    Returns:
        void

    """
    if start_date is None:
        return
    
    ticker_array = tickers

    dates = generate_date_interval_to_fetch(start_date, end_date)
    process_pipeline(ticker_array, dates, apiConnectionParams)

    return

def fetch_stock_data_years(tickers, number_of_years = 4, apiConnectionParams = None):
    """
    This function inject data corresponding the last <number_of_years> for the listed tickers.

    Args:
        tickers (list of string): tickers.
        number_of_years (number): the number of year you want to fetch (going back from today)
        apiConnectionParams: {POSTGREST_H, POSTGREST_P, POSTGREST_TOKEN} , postgRest api connection params

    Returns:
        void

    """

    today = datetime.today()
    start = datetime(today.year - number_of_years, today.month, today.day).strftime('%Y-%m-%d')
    update_stock_data_db_from_date(tickers,start, apiConnectionParams=apiConnectionParams)
 

    return 

def update_stock_data_db_to_current_date(connectionParams = None):
    """
    This function inject data corresponding from the latest date in the database to the current for the listed tickers.

    Args:
        apiConnectionParams: {POSTGREST_H, POSTGREST_P, POSTGREST_TOKEN} , postgRest api connection params

    Returns:
        void
    """

    df_latest_date = pd.DataFrame()
    if connectionParams != None:
            try:


                POSTGREST_H =  connectionParams['POSTGREST_H'] 
                POSTGREST_P =  connectionParams['POSTGREST_P']  

                # The API endpoint
                url = f'http://{POSTGREST_H}:{POSTGREST_P}/stock_data?order=time.desc&limit=1'

                response = requests.get(url)
              
                if response.status_code != 200:
                    print("error fetching data unable to update")
                    return 
                    
                
                data = response.json()
                df_latest_date = pd.DataFrame(data)

                if df_latest_date.shape[0] > 0:
                    latest_date_on_db = df_latest_date['date_str'][0]
                else: 
                    four_years_ago = datetime.now().replace(year=datetime.now().year - 4)
                    latest_date_on_db = four_years_ago.strftime("%Y-%m-%d")
                     
                stock_list  = get_ticker_list_from_db(connectionParams)
                tickers = stock_list['ticker'].to_numpy()
                #print('**** latest update *****', latest_date_on_db)
                #print("******** starting update *********")
                update_stock_data_db_from_date(tickers, start_date = latest_date_on_db, end_date = datetime.today().strftime("%Y-%m-%d") , apiConnectionParams = connectionParams)
               

            except Exception as e:
                print("Error:", e)

    
    return 

def ingestData(data, connectionParams = None):
    
    if connectionParams != None:
        try:


            POSTGREST_H =  connectionParams['POSTGREST_H']  # os.getenv("POSTGREST_H", "none")
            POSTGREST_P =  connectionParams['POSTGREST_P']  # os.getenv("POSTGREST_P", "none")
            POSTGREST_TOKEN = connectionParams['POSTGREST_TOKEN']  # os.getenv("POSTGREST_TOKEN", "none")
            
            #connect_to_DB_1(user=DB_USER, password=DB_PASSWORD, host=POSTGREST_H, port=POSTGREST_P, database=DB_NAME)
            
            # Headers with Authorization
            headers = {
                "Authorization": f"Bearer {POSTGREST_TOKEN}",
                "Content-Type": "application/json",
                'Accept':'application/json'
            }

            # The API endpoint
            url = f'http://{POSTGREST_H}:{POSTGREST_P}/stock_data'

            result_json = data.to_dict(orient='records')

            # Send POST request with JSON data
            response = requests.post(url, headers=headers, data=json.dumps(result_json))

            # Print the response
            if response.status_code == 200:
                #print("Request successful:", response.json())
                return True
            else:
                print("Error sending data:", response.text)
                return False
                
                #print("Error:", response.status_code, response.text)

        except Exception as e:
            print("Error:", e)

    return False

def ingestDataInParquetFile(data, file_path):
    """
    This function save the data in parquet file.

    Args:
        data (pandas.DataFrame): data to save.
        file_path (string): path to the file where to save the data.

    Returns:
        void

    """
    if data.empty:
        print("No data to save")
        return
    
    try:
        data.to_parquet(file_path, index=False)
        print(f"Data saved to {file_path}")
    except Exception as e:
        print(f"Error saving data: {e}")
               
# get the list of stock from Polygon by pooling the ticker from the market snapshat 
def fetch_ticker_list():
  
   
    # Polygon.io API details
    API_KEY = os.getenv("API_KEY", "none")  # Replace with your Polygon.io API key
    url = f'https://api.polygon.io/v3/reference/tickers?market=stocks&active=true&order=asc&limit=1000&sort=ticker&apiKey={API_KEY}'
   
    next = True
    df = pd.DataFrame()
    counter = 0;

    while next:

        #response = requests.get(url, params=params)
        response = requests.get(url)
        if response.status_code != 200:
            print(f"fetch_ticker_list: Error fetching data")
            return []
        
    
        data = response.json().get("results", [])
        
        newDf = pd.DataFrame(data)
        df = pd.concat([df, newDf], ignore_index=True)
       
       
        next_url = response.json().get("next_url", None)
        if next_url is None:
            next = False
        else:
            url = f'{next_url}&apiKey={API_KEY}'
            #print('********* fetching stock list from polygon ****', url)

     
        
       
    df = df[['ticker', 'name','primary_exchange']]
    df.rename(columns={'name': 'company_name', 'primary_exchange': 'stock_market'}, inplace=True)
    df['stock_market'] = df['stock_market'].fillna("")
    df['company_name'] = df['company_name'].fillna(df['ticker'])
   
    return df

# get the list of stocks in the database
def get_ticker_list_from_db(connectionParams):

    df = pd.DataFrame()
    if connectionParams != None:
            try:


                POSTGREST_H =  connectionParams['POSTGREST_H']  # os.getenv("POSTGREST_H", "none")
                POSTGREST_P =  connectionParams['POSTGREST_P']  # os.getenv("POSTGREST_P", "none")
                POSTGREST_TOKEN = connectionParams['POSTGREST_TOKEN']  # os.getenv("POSTGREST_TOKEN", "none")
                
                #connect_to_DB_1(user=DB_USER, password=DB_PASSWORD, host=POSTGREST_H, port=POSTGREST_P, database=DB_NAME)
                
                # Headers with Authorization
                headers = {
                    "Authorization": f"Bearer {POSTGREST_TOKEN}",
                    "Content-Type": "application/json",
                    'Accept':'application/json'
                }

                # The API endpoint
                url = f'http://{POSTGREST_H}:{POSTGREST_P}/ticker_info'

                response = requests.get(url)
              
                if response.status_code != 200:
                    print(f"error fetching data status: {response.status_code}")
                    return pd.DataFrame()
                
                data = response.json()
                df = pd.DataFrame(data)
               

            except Exception as e:
                print("Error:", e)

    return df

# get list of all tickers with its latest date in the stock_data table
def get_latest_ticker_date(connectionParams):
    
    df = pd.DataFrame()
    if connectionParams != None:
            try:


                POSTGREST_H =  connectionParams['POSTGREST_H']  # os.getenv("POSTGREST_H", "none")
                POSTGREST_P =  connectionParams['POSTGREST_P']  # os.getenv("POSTGREST_P", "none")
                POSTGREST_TOKEN = connectionParams['POSTGREST_TOKEN']  # os.getenv("POSTGREST_TOKEN", "none")
                
            
                # Headers with Authorization
                headers = {
                    "Authorization": f"Bearer {POSTGREST_TOKEN}",
                    "Content-Type": "application/json",
                    'Accept':'application/json'
                }

                # The API endpoint
                url = f'http://{POSTGREST_H}:{POSTGREST_P}/rpc/get_latest_ticker_time'

                response = requests.get(url)
              
                if response.status_code != 200:
                    print(f"error fetching data: get_latest_ticker_date: {response.status_code}")
                    return pd.DataFrame()
                
                data = response.json()
                df = pd.DataFrame(data)
                
               

            except Exception as e:
                print("Error:", e)

    return df

def get_ticker_items_from_db(ticker, connectionParams):
    
    df = pd.DataFrame()
    if connectionParams != None and ticker != None:
            try:


                POSTGREST_H =  connectionParams['POSTGREST_H']  # os.getenv("POSTGREST_H", "none")
                POSTGREST_P =  connectionParams['POSTGREST_P']  # os.getenv("POSTGREST_P", "none")
                POSTGREST_TOKEN = connectionParams['POSTGREST_TOKEN']  # os.getenv("POSTGREST_TOKEN", "none")
                
                #connect_to_DB_1(user=DB_USER, password=DB_PASSWORD, host=POSTGREST_H, port=POSTGREST_P, database=DB_NAME)
                
                # Headers with Authorization
                headers = {
                    "Authorization": f"Bearer {POSTGREST_TOKEN}",
                    "Content-Type": "application/json",
                    'Accept':'application/json'
                }

                # The API endpoint
                url = f'http://{POSTGREST_H}:{POSTGREST_P}/stock_data?ticker=eq.{ticker}&order=time.desc'

                response = requests.get(url)
              
                if response.status_code != 200:
                    print(f"error fetching data: get_ticker_items_from_db: {response.status_code}")
                    return pd.DataFrame()
                
                data = response.json()
                df = pd.DataFrame(data)
               
               

            except Exception as e:
                print("Error:", e)   
    return df

# get the list of the tickers that has been recently listed in the market but are not in the DB yet
def get_new_listed_stocks(connectionParams):
    df_dif_clean = pd.DataFrame([])

    if connectionParams != None:
        
        
        try:
            
            # Fetch data
            df_ticker_from_db = get_ticker_list_from_db(connectionParams)
            stock_list = fetch_ticker_list()
            
            

            # --- Normalize ticker columns (CRITICAL) ---
            df_ticker_from_db['ticker'] = (
                df_ticker_from_db['ticker']
                .astype(str)
                .str.strip()
                .str.upper()
            )
            
            stock_list['ticker'] = (
                stock_list['ticker']
                .astype(str)
                .str.strip()
                .str.upper()
            )
            
            # print('============= get_new_listed_stocks ============')
           
            # print(df_ticker_from_db)
            # print(stock_list)

            if not df_ticker_from_db.empty:
                df_ticker_from_db['ticker'] = (
                    df_ticker_from_db['ticker']
                    .astype(str)
                    .str.strip()
                    .str.upper()
                )

            # --- Compute difference (tickers not in DB) ---
            if not df_ticker_from_db.empty:
                df_dif = stock_list[ ~stock_list['ticker'].isin(df_ticker_from_db['ticker'])]
            else:
                df_dif = stock_list.copy()

            # --- Final ticker validation (realistic US tickers) ---
            # Allows letters, numbers, dots, dashes (BRK.B, BF-A, ABC1, etc.)
           
            print(df_dif)
            
            df_dif_clean = df_dif[
                df_dif['ticker'].str.match(r'^[A-Z0-9.\-]+$', na=False)
            ].reset_index(drop=True)
            
            
            
           
            
            
        except Exception as e:
            print(f'---- get_new_listed_stocks : {e}')
    
    return df_dif_clean


# inject the given list of stock to the stock list in the DB
def update_stock_list_table(connectionParams, df_dif_clean):
    
   
    if connectionParams != None:
            try:
               
                     
                POSTGREST_H =  connectionParams['POSTGREST_H']  # os.getenv("POSTGREST_H", "none")
                POSTGREST_P =  connectionParams['POSTGREST_P']  # os.getenv("POSTGREST_P", "none")
                POSTGREST_TOKEN = connectionParams['POSTGREST_TOKEN']  # os.getenv("POSTGREST_TOKEN", "none")
                
                 # Headers with Authorization
                headers = {
                    "Authorization": f"Bearer {POSTGREST_TOKEN}",
                    "Content-Type": "application/json",
                    'Accept':'application/json'
                }

                # The API endpoint
                url = f'http://{POSTGREST_H}:{POSTGREST_P}/ticker_info'
    
                result_json = df_dif_clean.to_dict(orient='records')
                
              
                # Send POST request with JSON data
                response = requests.post(url, headers=headers, data=json.dumps(result_json))
                
                return response
                
                # Print the response
                if response.status_code > 200:
                    print("Request successful:", response.json())
                else:
                    print("Error sending data:", response)
                    #print("Error:", response.status_code, response.text)

            except Exception as e:
                print("----- update_stock_list_DB:", e)
    
    
    return 
     

# update stock database list 
# if new stock are added to NYSE o NASDAQ they will be added to the DB.
def update_stock_list_DB(connectionParams):

    if connectionParams != None:
            try:
                df_ticker_from_db = get_ticker_list_from_db(connectionParams)
                stock_list = fetch_ticker_list()
                
                df_dif =  stock_list
                if not df_ticker_from_db.empty:
                    df_dif = stock_list[~stock_list['ticker'].isin(df_ticker_from_db['ticker'])]
                             
                df_dif_clean = df_dif[~df_dif['ticker'].str.contains(r'\.', regex=True,  na=False)]
                df_dif_clean = df_dif_clean[df_dif_clean['ticker'].str.match(r'^[A-Za-z]+$', na=False)]
                    
                if df_dif_clean.empty:
                    print('******* the database ticker info is up to date')
                    return
                     
                POSTGREST_H =  connectionParams['POSTGREST_H']  # os.getenv("POSTGREST_H", "none")
                POSTGREST_P =  connectionParams['POSTGREST_P']  # os.getenv("POSTGREST_P", "none")
                POSTGREST_TOKEN = connectionParams['POSTGREST_TOKEN']  # os.getenv("POSTGREST_TOKEN", "none")
                
                 # Headers with Authorization
                headers = {
                    "Authorization": f"Bearer {POSTGREST_TOKEN}",
                    "Content-Type": "application/json",
                    'Accept':'application/json'
                }

                # The API endpoint
                url = f'http://{POSTGREST_H}:{POSTGREST_P}/ticker_info'
    
                result_json = df_dif_clean.to_dict(orient='records')
                #print(df_dif_clean)
                #print(result_json)
                
                # Send POST request with JSON data
                response = requests.post(url, headers=headers, data=json.dumps(result_json))
                
                # Print the response
                if response.status_code > 200:
                    print("Request successful:", response.json())
                else:
                    print("Error sending data:", response)
                    #print("Error:", response.status_code, response.text)

            except Exception as e:
                print("----- update_stock_list_DB:", e)
                
    return

def injectTickerInfo(connectionParams, df_data):
    
   
    
    if connectionParams != None:
            try:
               
                if df_data.empty:
                    print('******* the database ticker info is up to date')
                    return
                     
                POSTGREST_H =  connectionParams['POSTGREST_H']  # os.getenv("POSTGREST_H", "none")
                POSTGREST_P =  connectionParams['POSTGREST_P']  # os.getenv("POSTGREST_P", "none")
                POSTGREST_TOKEN = connectionParams['POSTGREST_TOKEN']  # os.getenv("POSTGREST_TOKEN", "none")
                
                 # Headers with Authorization
                headers = {
                    "Authorization": f"Bearer {POSTGREST_TOKEN}",
                    "Content-Type": "application/json",
                    'Accept':'application/json'
                }

                # The API endpoint
                url = f'http://{POSTGREST_H}:{POSTGREST_P}/ticker_info'
    
                result_json = df_data.to_dict(orient='records')
                
                # Send POST request with JSON data
                response = requests.post(url, headers=headers, data=json.dumps(result_json))
                
                
                # Print the response
                if response.status_code > 200:
                    print("Request successful:", response.json())
                else:
                    print("Error sending data:", response)
                    #print("Error:", response.status_code, response.text)

            except Exception as e:
                print("Error:", e)
    
    return  df_data

def createTickerList(connectionParams):
    
    # ======== code to create a ticker list from zero with the nasdaq, nyse and all tickers extracted from polygon ====
    # df = pd.read_csv("nasdaq_16_8_2024.csv")
    # df = df[df['ACT Symbol'].str.match(r'^[A-Za-z]+$', na=False)]
    # df['stock_market'] = "XNAS"

    # df1 = pd.read_csv("nyse-listed.csv")
    # df1 = df1[df1['ACT Symbol'].str.match(r'^[A-Za-z]+$', na=False)]
    # df1['stock_market'] = "XASE"

    # all = pd.concat([df, df1], ignore_index=True)
    # all = all[~all['ACT Symbol'].duplicated(keep=False)]
    # all.rename(columns={'ACT Symbol': 'ticker', 'Company Name': 'company_name'}, inplace=True)
    
    # stock_list = fetch_ticker_list()
    # all = pd.concat([stock_list, all], ignore_index=True)
    # no_duplicates  = all.drop_duplicates(subset=['ticker'], keep='first')
    # clean_data = no_duplicates[no_duplicates['ticker'].str.match(r'^[A-Za-z]+$', na=False)]
    # print(clean_data)
    # today = date.today()  
    # clean_data.to_csv(f'all_tickers.csv')
    
    # **  when having a list of tickers with all of them already stored previously (previous commented above executed at least once)
    # **  loading that list and joining it with all tickers extracted from polygon in the current day
    all = pd.read_csv("all_tickers.csv")
    stock_list = fetch_ticker_list()
    all = pd.concat([stock_list, all], ignore_index=True)
    no_duplicates  = all.drop_duplicates(subset=['ticker'], keep='first')
    clean_data = no_duplicates[no_duplicates['ticker'].str.match(r'^[A-Za-z]+$', na=False)]
    clean_data = clean_data[["ticker","company_name","stock_market"]]
    
    injectTickerInfo(connectionParams, clean_data)
     
    return

def get_ticker_float_and_marketcap( ticker, date, previous_close ):
    
    payload = {
        "stock_float": -1,
        "market_cap": -1,
        "date_s": date
        
        }
    
    if ticker is None or date is None:
        return payload
    
    
    API_KEY = os.getenv("API_KEY", "none") 
        
    try:
        
        url = f'https://api.polygon.io/v3/reference/tickers/{ticker}?apiKey={API_KEY}&date={date}'
       
      
        response = requests.get(url)
        data= response.json().get("results", [])
    
        stock_float = -1
        
        if 'share_class_shares_outstanding'in data:
            stock_float = data['share_class_shares_outstanding']
        elif 'weighted_shares_outstanding'in data:
             stock_float = data['weighted_shares_outstanding']
        else:
            stock_float = 0
            
        if 'market_cap' in data: 
            market_cap = int(round(data['market_cap']))
        else:
            market_cap =   int(round(stock_float * previous_close))
      
        # Payload to update
        payload = {
                "stock_float": stock_float,
                "market_cap": market_cap,
                "date_s": date
        }
          
        return payload

    except Exception as e:
        print(f"Error fetching overview data for {ticker}: {e}")
        
    
  
    return payload

def get_ticker_float_and_marketcap( ticker, date, previous_close ):
    
    payload = {
        "stock_float": -1,
        "market_cap": -1,
        "date_s": date
        
        }
    
    if ticker is None or date is None:
        return payload
    
    
    API_KEY = os.getenv("API_KEY", "none") 
        
    try:
        
        url = f'https://api.polygon.io/v3/reference/tickers/{ticker}?apiKey={API_KEY}&date={date}'
       
      
        response = requests.get(url)
        data= response.json().get("results", [])
    
        stock_float = -1
        
        if 'share_class_shares_outstanding'in data:
            stock_float = data['share_class_shares_outstanding']
        elif 'weighted_shares_outstanding'in data:
             stock_float = data['weighted_shares_outstanding']
        else:
            stock_float = 0
            
        if 'market_cap' in data: 
            market_cap = int(round(data['market_cap']))
        else:
            market_cap =   int(round(stock_float * previous_close))
      
        # Payload to update
        payload = {
                "stock_float": stock_float,
                "market_cap": market_cap,
                "date_s": date
        }
          
        return payload

    except Exception as e:
        print(f"Error fetching overview data for  get_ticker_float_and_marketcap {ticker}: {e}")
        
    
  
    return {}
 

def get_float_and_marketcap( ticker, date ):
    
    payload = {
        "stock_float": -1,
        "market_cap": -1,
        "date_s": date,
        "ticker": ticker
        
        }
    
    if ticker is None or date is None:
        return payload
    
    
    API_KEY = os.getenv("API_KEY", "none") 
        
    try:
        
        url = f'https://api.polygon.io/v3/reference/tickers/{ticker}?apiKey={API_KEY}&date={date}'
       
        print(f" ==== {url}")
        response = requests.get(url)
        data= response.json().get("results", [])
    
        stock_float = -1
        market_cap = -1
        
        if 'share_class_shares_outstanding'in data:
            stock_float = data['share_class_shares_outstanding']
        elif 'weighted_shares_outstanding'in data:
             stock_float = data['weighted_shares_outstanding']
        else:
            stock_float = 0
            
        if 'market_cap' in data: 
            market_cap = int(round(data['market_cap']))
       
      
        # Payload to update
        payload = {
                "stock_float": stock_float,
                "market_cap": market_cap,
                "date_s": date,
                "ticker": ticker
        }
        

          
        return payload

    except Exception as e:
        print(f" error get_float_and_marketcap: {url}")
        print(f"Error fetching overview data for  get_float_and_marketcap{ticker}: {e}")
        
    
  
    return payload

       
def get_200_DSMA_by_date(ticker, date):
    
    result =  {'date_s':date, 'daily_200_sma': -1, "ticker":ticker}
    
    if ticker is None or date is None:
        return result
    
    
    API_KEY = os.getenv("API_KEY", "none") 
        
    try:
        
        url = f'https://api.polygon.io/v1/indicators/sma/{ticker}?timespan=day&adjusted=false&window=200&series_type=close&expand_underlying=false&order=desc&limit=5000&apiKey={API_KEY}&timestamp={date}'
      
        print(f" *** {url}")
       
        response = requests.get(url)
        data= response.json().get("results", [])

        
        if 'values' in data:
            return  {'date_s':date, 'daily_200_sma': data['values'][0]['value'],  "ticker":ticker}
        
    except Exception as e:
        print(f" error get_200_DSMA_by_date: {url}")
        print(f"Error fetching 200 SMA data for {ticker}: {e}")
        
             
    return result             

def get_200_DSMA(ticker):
    
    if ticker is None:
        return None
    
    
    API_KEY = os.getenv("API_KEY", "none") 
        
    try:
        
        url = f'https://api.polygon.io/v1/indicators/sma/{ticker}?timespan=day&adjusted=false&window=200&series_type=close&expand_underlying=false&order=desc&limit=5000&apiKey={API_KEY}'
      
        response = requests.get(url)
        data= response.json().get("results", [])
        
       
        dsma_200 = -1
        
        if 'values' in data:
            dsma_200 = data['values']
      
        result = pd.DataFrame([])
        if dsma_200 != -1:
            result = pd.DataFrame(dsma_200)
            result = result.rename(columns={'value': 'Daily_200_SMA'}, inplace=False).reset_index(drop=True)
            result['date_s'] = pd.to_datetime(result["timestamp"], unit="ms").dt.strftime("%Y-%m-%d")
            result.drop(columns=['timestamp'])
            

     
       
        return result

    except Exception as e:
        print(f"Error fetching overview data for get_200_DSMA {ticker}: {e}")
        
    return pd.DataFrame()
  
  
def get_items_from_gappers_table(connectionParams, offset = 0, limit=1000):
    
    df = pd.DataFrame()
    if connectionParams != None:
            try:


                POSTGREST_H =  connectionParams['POSTGREST_H']  # os.getenv("POSTGREST_H", "none")
                POSTGREST_P =  connectionParams['POSTGREST_P']  # os.getenv("POSTGREST_P", "none")
                POSTGREST_TOKEN = connectionParams['POSTGREST_TOKEN']  # os.getenv("POSTGREST_TOKEN", "none")
                
                #connect_to_DB_1(user=DB_USER, password=DB_PASSWORD, host=POSTGREST_H, port=POSTGREST_P, database=DB_NAME)
                
                # Headers with Authorization
                headers = {
                    "Authorization": f"Bearer {POSTGREST_TOKEN}",
                    "Content-Type": "application/json",
                    'Accept':'application/json'
                }

                # The API endpoint
                url = f'http://{POSTGREST_H}:{POSTGREST_P}/gappers_pm_and_market_hours?limit={limit}&offset={offset}'

                response = requests.get(url)
              
                if response.status_code != 200:
                    print(f"error fetching data: get_ticker_items_from_db: {response.status_code}")
                    return pd.DataFrame()
                
                data = response.json()
                df = pd.DataFrame(data)
               
               

            except Exception as e:
                print("Error:", e)   
    return df
    
def fetch_stock_float_and_marketcap_from_df(df):
       
    if df.empty:
        return
    
    data = pd.DataFrame([])
    df['stock_float'] = -1
    df['market_cap'] = -1
    df['daily_200_sma'] = -1
    
    for idx, row in df.iterrows():
        data = get_ticker_float_and_marketcap(row['ticker'], row['date_str'], row['previous_close'] )
        data_df = pd.DataFrame([data])
        sma_200 = get_200_DSMA_by_date(row['ticker'], row['date_str'])
        sma_200_df = pd.DataFrame([sma_200])
        
        
        
        df = df.set_index("date_str")
        data_df = data_df.set_index("date_s")
        sma_200_df = sma_200_df.set_index("date_s")
        df.update(data_df[["stock_float", "market_cap"]])
        df["daily_200_sma"] = df["daily_200_sma"].astype(float)
        df.update(sma_200_df[["daily_200_sma"]])

        df = df.reset_index()
        
    return df
    
def resumeProcess(connectionParams):
    """
    This function resume the process of fetching data from the last ticker processed.
    
    Args:
        void
        
    Returns:
        void
    """
    
    STOCK_MARKET_PARQUET_PATH = os.getenv("STOCK_MARKET_PARQUET_PATH", "none") 
    last_ticker_file = f'{STOCK_MARKET_PARQUET_PATH}/last_ticker.txt'
    
    if not os.path.exists(last_ticker_file):
        print("No last ticker file found, starting from the beginning.")
        return
    
    with open(last_ticker_file, 'r') as f:
        last_ticker_info = f.read().strip()
    
    if not last_ticker_info:
        print("Last ticker info is empty, starting from the beginning.")
        return
    
    ticker = last_ticker_info
  
    try:
        # Fetch all tickers
        stock_list = get_ticker_list_from_db(connectionParams)
        tickers = stock_list['ticker'].to_numpy()
        filtered = [s for s in tickers if re.fullmatch(r"[A-Za-z]+", s)]
        index = filtered.index(ticker)
        
        if index >= 0 and index < len(filtered) - 1:
            four_years_ago = datetime.now().replace(year=datetime.now().year - 4)
            latest_date_on_db = four_years_ago.strftime("%Y-%m-%d")
            
            dates = generate_date_interval_to_fetch(latest_date_on_db,  datetime.today().strftime("%Y-%m-%d"))
            
            # Process pipeline
            process_pipeline(filtered[index:], dates, None)  # Assuming apiConnectionParams is None for this example
        
    except Exception as e:
        print(f"Error processing last ticker info: {e}")
        return
    
def inject_parquet_files_in_db(folder_path, apiConnectionParams):
    """
    Reads all .parquet files from a given folder into a single DataFrame.

    Parameters:
        folder_path (str): Path to the folder containing .parquet files.

    Returns:
        pd.DataFrame: Concatenated DataFrame containing all files' data.
    """
    dfs = []
    count = 0
    
    for file in os.listdir(folder_path):
        
        if file.endswith(".parquet") :
            count += 1
            print(f"Reading {file} - {count}")
            file_path = os.path.join(folder_path, file)
            df = pd.read_parquet(file_path)
            ingestData(df,apiConnectionParams)
            
def  update_ticker(ticker, folder_path, start_date, end_date, apiConnectionParams):
    """ For a given ticker fetch data from start_date to end_date and inject it in the parquet database.

    Args:
        ticker (_type_): _description_
        folder_path (_type_): _description_
        start_date (_type_): _description_
        end_date (_type_): _description_
        apiConnectionParams (_type_): _description_
    """
    STOCK_MARKET_PARQUET_PATH = os.getenv("STOCK_MARKET_PARQUET_PATH", "none") 
    df = pd.DataFrame()
    

    dates = generate_date_interval_to_fetch(start_date, end_date)
       
    for item in dates:
            (date1, date2) = item
            data30 = fetch_ticker_data_30(ticker, date1, date2)
            
            print(f'ticker: {ticker} - from: {date1} to {date2}')
            processedData = process_data_30_minutes(data30)
            dataDaily = raw_data_to_dataframe(fetch_ticker_data_daily(ticker, date1, date2))
            
            if processedData.empty == False and dataDaily.empty == False:
                            
                try:
                    dataDaily = dataDaily.drop(columns=['high', 'low','date', 'time', 'vw', 'volume','n'])
                    processedData = processedData.merge(dataDaily, on="day", how="left")
                    processedData['ticker'] = ticker
                    processedData['date_str'] = pd.to_datetime(pd.to_datetime(processedData["day"])).dt.strftime('%Y-%m-%d') 
                   
                    processedData = fetch_stock_float_and_marketcap_from_df(processedData)  
              
                except  Exception as e:
                    print(f' error  droping daily columns or merging dataframes: {e}')

                if df.empty:
                    df = processedData
                else:
                    df =  pd.concat([df, processedData], ignore_index=True)
                    
                    
    if df.empty == False:

            try:

                # Sort by time to ensure correct order
                df = df.sort_values(by="day").reset_index(drop=True)
                # Compute the gap (current day's open - previous day's close)
                df['previous_close'] = df['close'].shift(1)
                df['gap'] = df['open'] - df['previous_close']
                df['daily_range'] = (   df['high'] - df['low']) * 100 / df['open']
                df['gap_perc'] = np.where( df['previous_close'] > 0, (df['open'] - df['previous_close']) *100 / df['previous_close'], 0 )
                # Fill NaN gaps with 0 for the first day
                df['previous_close'] = df['previous_close'].fillna(0)
                df['gap'] = df['gap'].fillna(0)
                #df['gap_perc'].fillna(0, inplace=True)
                #df['gap_perc'].fillna(0, inplace=True)
                df['ticker'] = ticker
                #df = df.drop(columns=['previous_close'])
                df['date_str'] = pd.to_datetime(pd.to_datetime(df["day"])).dt.strftime('%Y-%m-%d') 
                        
                df = df.drop(columns=['day'])
                
                ingestData(df,apiConnectionParams)
                ingestDataInParquetFile(df, f'{STOCK_MARKET_PARQUET_PATH}/{ticker}.parquet')
            except  Exception as e:
                print( f'error in process of injecting data: {e}')
                
      
      
    
    return

def create_database_of_stock_data(folder_path, apiConnectionParams):
    
    STOCK_MARKET_PARQUET_PATH = os.getenv("STOCK_MARKET_PARQUET_PATH", "none") 
    last_ticker_file = f'{STOCK_MARKET_PARQUET_PATH}/last_ticker.txt'
    
    if  os.path.exists(last_ticker_file):
        
        
        with open(last_ticker_file, 'w') as f:
            f.write(f'')
    
       
    update_parquet_database(update_parquet_database)
        
    return 

def update_stock_data(folder_path, apiConnectionParams):
    
    STOCK_MARKET_PARQUET_PATH = os.getenv("STOCK_MARKET_PARQUET_PATH", "none") 
    last_ticker_file = f'{STOCK_MARKET_PARQUET_PATH}/last_ticker.txt'
    if  os.path.exists(last_ticker_file):
        with open(last_ticker_file, 'w') as f:
            f.write(f'')
    
       
    resume_stock_data_update(folder_path, apiConnectionParams)
  
    return

def resume_stock_data_update(folder_path, apiConnectionParams):
    """
    Fetch list of tickers from the database first.
    For each ticker fetch data from the last date in the database to today and inject it in the parquet database.
    
    Note: the function keep track of the last ticker processed and the last date processed in two text files (last_ticker.txt, latest_date.txt)
    in the STOCK_MARKET_PARQUET_PATH folder. In case of failure the process can be resumed from the last ticker processed.
    
    Parameters:
        folder_path (str): Path to the folder containing .parquet files.
        apiConnectionParams: {POSTGREST_H, POSTGREST_P, POSTGREST_TOKEN} , postgRest api connection params

    Returns:
        void
    """
    
    STOCK_MARKET_PARQUET_PATH = os.getenv("STOCK_MARKET_PARQUET_PATH", "none") 
    last_ticker_file = f'{STOCK_MARKET_PARQUET_PATH}/last_ticker.txt'
    last_date_update_file = f'{STOCK_MARKET_PARQUET_PATH}/latest_date.txt'
    
    if not os.path.exists(last_ticker_file):
        print("No last ticker file found, starting from the beginning.")
        last_ticker_info = ''
    else:
        with open(last_ticker_file, 'r') as f:
            last_ticker_info = f.read().strip()  
        
    if not os.path.exists(last_date_update_file):
        print("No last date file found, starting from the beginning.")
        last_date_update  =  ""
    else:
        with open(last_date_update_file, 'r') as f:
            last_date_update = f.read().strip()
             
   
    
    
    end_date =  datetime.today().strftime("%Y-%m-%d")
    ticker = last_ticker_info
    
  
    try:
        # Fetch all tickers
        stock_list = get_ticker_list_from_db(apiConnectionParams)
        tickers = stock_list['ticker'].to_numpy()
        filtered = [s for s in tickers if re.fullmatch(r"[A-Za-z]+", s)]
        if last_ticker_info != '' and last_ticker_info != '---done---':
            index = filtered.index(last_ticker_info)
            if index >= 0 and index < len(filtered) - 1:
                filtered =  filtered[index:]  
                
        if last_ticker_info == "---done---" :
            print(f"ticker: {last_ticker_info} , date: {last_date_update}") 
            print(' all data has been processed')
            return 
            
        
       
        len_tickers = len(filtered)
        for index in range(0,len_tickers):
            ticker = filtered[index]
            if len(last_date_update) == 0:
                last_date_update = get_ticker_latest_date(apiConnectionParams, ticker)
                
            update_ticker(ticker=ticker, folder_path=folder_path, start_date=last_date_update, end_date=end_date, apiConnectionParams=apiConnectionParams)
            if index == len_tickers -1:
                with open(f'{STOCK_MARKET_PARQUET_PATH}/last_ticker.txt', 'w') as f:
                    f.write(f'---done---')
                with open(f'{STOCK_MARKET_PARQUET_PATH}/latest_date.txt', 'w') as f:
                    f.write(f'{end_date}\n')
                
            else:
                # save index of last ticker processed
                with open(f'{STOCK_MARKET_PARQUET_PATH}/last_ticker.txt', 'w') as f:
                    f.write(f'{ticker}\n')
                
                with open(f'{STOCK_MARKET_PARQUET_PATH}/latest_date.txt', 'w') as f:
                    f.write(f'{last_date_update}\n')
        
    except Exception as e:
        print(f"Error processing last ticker info: {e}")
        return

def update_stock_data_from_parquet(folder_path, apiConnectionParams):
    """
    This function reads all .parquet files from a given folder and updates the stock data in the database.

    Args:
        folder_path (str): Path to the folder containing .parquet files.
        apiConnectionParams: {POSTGREST_H, POSTGREST_P, POSTGREST_TOKEN} , postgRest api connection params

    Returns:
        void
    """
    
   
    
    return
    
def sync_files_parquet(folder_path, output_base_path):
    """
    Update the parquet files in the corresponding folders with the last parquet file generated from the data fetch.
    If the folder does not exist it will be created and the parquet file moved to it.
    
    Parameters:
        folder_path (str): Path containing .parquet files.
        output_base_path (str): Path where the folders should be created.
    """
    for file in os.listdir(folder_path):
        if file.endswith(".parquet"):
            print(f"Creating folder for {file}")
            folder_name = os.path.splitext(file)[0]  # remove extension
            new_folder_path = os.path.join(output_base_path, folder_name)
            file_path = os.path.join(new_folder_path,f'{folder_name}.parquet')
            source_file = os.path.join(folder_path, file)
            print(f"Creating folder for {new_folder_path}")
            if  Path(file_path).is_file():
                df = pd.read_parquet(file_path)
                new_rows = pd.read_parquet(source_file)
                df_updated = pd.concat([df, new_rows], ignore_index=True)
                df_updated = df_updated.drop_duplicates()
                # Step 4: Write back to parquet (overwrite)
                df_updated.to_parquet(file_path, index=False)
                # remove the source file since it’s merged
                os.remove(source_file)
                
            else:
                print(f"moving file to new folder for {file}")
                os.makedirs(new_folder_path, exist_ok=True)     
                shutil.move(os.path.join(folder_path, file), new_folder_path)
    
   
  # ======= temporaly function ======= 
 
#========= temporaly function ======= 
    
# inject 200 days simple moving average in the stock_data table for each ticker in the database
def inject_200_dsma_in_db( connectionParams):
    
    STOCK_MARKET_PARQUET_PATH = os.getenv("STOCK_MARKET_PARQUET_PATH", "none") 
    last_ticker_file = f'{STOCK_MARKET_PARQUET_PATH}/last_ticker_200_sma.txt'
    
    if not os.path.exists(last_ticker_file):
        print("No last ticker file found, starting from the beginning.")
        return
    
    with open(last_ticker_file, 'r') as f:
        last_ticker_info = f.read().strip()
    
    if not last_ticker_info:
        print("Last ticker info is empty, starting from the beginning.")
        return
    
    ticker = last_ticker_info
    
    if connectionParams is None:
        return None
     
    ticker_list  = get_ticker_list_from_db(connectionParams) 
    stock_list = get_ticker_list_from_db(connectionParams)
    tickers = stock_list['ticker'].to_numpy()
    filtered = [s for s in tickers if re.fullmatch(r"[A-Za-z]+", s)]
    
    ticker_index = filtered.index(ticker)
        
    if ticker_index >= 0 and ticker_index < len(filtered) - 1:
        filtered = filtered[ticker_index:]
        len_tickers = len(filtered)
        for index in range(0,len_tickers):
            ticker = filtered[index]
            inject_ticker_200_dsma_in_db(ticker, connectionParams)
            
            if index == len_tickers - 1:
                with open(f'{STOCK_MARKET_PARQUET_PATH}/last_ticker_200_sma.txt', 'w') as f:
                    f.write(f'------done----------\n')
            else:
                # save index of last ticker processed
                with open(f'{STOCK_MARKET_PARQUET_PATH}/last_ticker_200_sma.txt', 'w') as f:
                    f.write(f'{ticker}\n')
    
    
    return 

# inject 200 days simple moving average in the stock_data table for a given ticker
def inject_ticker_200_dsma_in_db(ticker, connectionParams):
    
    if ticker is None or connectionParams is None:
        return None
     

    POSTGREST_H =  connectionParams['POSTGREST_H']  # os.getenv("POSTGREST_H", "none")
    POSTGREST_P =  connectionParams['POSTGREST_P']  # os.getenv("POSTGREST_P", "none")
    POSTGREST_TOKEN = connectionParams['POSTGREST_TOKEN'] 
    headers = {
        "Authorization": f"Bearer {POSTGREST_TOKEN}",
        "Content-Type": "application/json",
        'Accept':'application/json'
        }
    
    ticker_200_dsma = get_200_DSMA(ticker)
    ticker_items = get_ticker_items_from_db(ticker, connectionParams)
    
    
    #print("=== injecting 200 DSMA in DB =====")
    #print(f"ticker items: {ticker_items.shape[0]} - 200 DSMA items: {ticker_200_dsma.shape[0]}")
    #print(ticker_200_dsma)
    #print(ticker_items)
    
    if ticker_items.empty or ticker_200_dsma.empty:
        return pd.DataFrame([])
    
    merged = pd.merge( ticker_items,ticker_200_dsma, left_on="date_str", right_on="date_s", how="inner")
    merged['daily_200_sma'] = merged['Daily_200_SMA'].astype(float).round(2)
    merged.drop(columns=['date_s','timestamp','Daily_200_SMA'], inplace=True)
    print(f"=== started: {ticker} =====")
    for _, row in merged.iterrows():
        #print(f'updating {row["ticker"]} - {row["date_str"]} with {row["daily_200_sma"]}')
        json_data = row.to_dict()
        params = {
            "ticker": f"eq.{row['ticker']}",
            "date_str": f"eq.{row['date_str']}"
        }
        url_base = f'http://{POSTGREST_H}:{POSTGREST_P}/stock_data'
        response = requests.patch(url_base, headers=headers, params=params, json=json_data)
        if response.status_code != 204:  # 204 = success with no content
            print("Failed update:", row['ticker'], row['date_str'], response.text)
    
    print("=== finished =====")
    return  merged
 
# inject float and market cap in the stock_data table for each ticker in the database
def inject_float_and_marketcap_in_db( connectionParams):
    
    import time
    
    STOCK_MARKET_PARQUET_PATH = os.getenv("STOCK_MARKET_PARQUET_PATH", "none") 
    last_ticker_file = f'{STOCK_MARKET_PARQUET_PATH}/last_ticker_200_sma.txt'
    
    if connectionParams is None:
        return None
     
    tickers = get_ticker_without_mcap_list_from_db(connectionParams)
    filtered = [s for s in tickers if re.fullmatch(r"[A-Za-z]+", s)]
    ticker_index = 0
        
    if ticker_index >= 0 and ticker_index <= len(filtered) - 1:
        filtered = filtered[ticker_index:]
        len_tickers = len(filtered)
        for index in range(0,len_tickers):
            ticker = filtered[index]
            # if index < len_tickers - 1:
            #     # save index of last ticker processed
            #     with open(f'{STOCK_MARKET_PARQUET_PATH}/last_ticker_200_sma.txt', 'w') as f:
            #         f.write(f'{ticker}\n')
                    
            start = time.time()
            inject_ticker_float_and_marketcap_in_db(ticker, connectionParams)
            end = time.time()
            elapsed = end - start
            print(f"Execution time: {elapsed:.4f} seconds")
            
            
            
            if index == len_tickers - 1:
                with open(f'{last_ticker_file}', 'w') as f:
                    f.write(f'------done----------\n')
            
    
    
    return 

# inject float and market cap in the stock_data table for a given ticker
def inject_ticker_float_and_marketcap_in_db(ticker, connectionParams):
     
    if ticker is None or connectionParams is None:
        print("ticker or connectionParams is None", ticker, connectionParams)
        return 
     

    POSTGREST_H =  connectionParams['POSTGREST_H']  # os.getenv("POSTGREST_H", "none")
    POSTGREST_P =  connectionParams['POSTGREST_P']  # os.getenv("POSTGREST_P", "none")
    POSTGREST_TOKEN = connectionParams['POSTGREST_TOKEN'] 
    headers = {
        "Authorization": f"Bearer {POSTGREST_TOKEN}",
        "Content-Type": "application/json",
        'Accept':'application/json'
        }
    
   
    ticker_items = get_ticker_items_from_db(ticker, connectionParams)
    
    if ticker_items.empty:
        return pd.DataFrame([])
    
    print(f"=== injecting float and market cap in DB for {ticker} =====")
    df_copy = ticker_items.copy()
    ticker_items = ticker_items.set_index("date_str")
    count = 0
    total = ticker_items.shape[0]
    marquetcap_float_data = []
    for _, row in df_copy.iterrows():
        marquetcap_float_data.append(get_ticker_float_and_marketcap(row['ticker'], row['date_str'], row['previous_close'])) 
        
        #data = get_ticker_float_and_marketcap(row['ticker'], row['date_str'])
        #data_df = pd.DataFrame([data])
        #data_df = data_df.set_index("date_s")
        #ticker_items.update(data_df[["stock_float", "market_cap"]])
        count = count + 1
        
    data_df = pd.DataFrame(marquetcap_float_data)
    data_df = data_df.set_index("date_s")
    ticker_items.update(data_df[["stock_float", "market_cap"]])
    
    ticker_items = ticker_items.reset_index()
    
    ticker_items = ticker_items[['ticker','date_str','stock_float','market_cap','daily_200_sma']]
    
    url = f'http://{POSTGREST_H}:{POSTGREST_P}/rpc/bulk_update_stock_data_float_mcap_200sma'
    payload = {
        "jsonb_input": ticker_items.to_dict(orient="records")
    }

    res = requests.post(
        url,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {POSTGREST_TOKEN}"},
        data=json.dumps(payload))
    
    print(res.status_code)
    print(res.text)
    
    print("=== finished =====")
    return 
    
def inject_float_and_marketcap_for_list_of_tickers(ticker_list, connectionParams):
    
     
    if ticker_list is None or connectionParams is None:
        print("ticker_list or connectionParams is None", ticker_list, connectionParams)
        return 
    
    if ticker_list.empty:
        print("ticker_list is empty, nothing to process")
        return
    
    import time
    
    POSTGREST_H =  connectionParams['POSTGREST_H']  # os.getenv("POSTGREST_H", "none")
    POSTGREST_P =  connectionParams['POSTGREST_P']  # os.getenv("POSTGREST_P", "none")
    POSTGREST_TOKEN = connectionParams['POSTGREST_TOKEN'] 
    headers = {
        "Authorization": f"Bearer {POSTGREST_TOKEN}",
        "Content-Type": "application/json",
        'Accept':'application/json'
        }
    

    count = 0
    total = ticker_list.shape[0]
    marquetcap_float_data = []
    for _, row in ticker_list.iterrows():
        print(f'processing {row["ticker"]} - {row["date_str"]}  {count+1}/{total}')
        result = get_ticker_float_and_marketcap(row['ticker'], row['date_str'], row['previous_close'])
        result['ticker'] = row['ticker']
        marquetcap_float_data.append(result) 
        count = count + 1
        
    data_df = pd.DataFrame(marquetcap_float_data)
    
    print(data_df)
    
    url = f'http://{POSTGREST_H}:{POSTGREST_P}/rpc/update_stock_data_float_marketcap_for_list_of_tickers'
    payload = {
        "jsonb_input": data_df.to_dict(orient="records")
    }

    res = requests.post(
        url,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {POSTGREST_TOKEN}"},
        data=json.dumps(payload))
    
    print(res.status_code)
    print(res.text)
    
    print("=== finished =====")
    
  
    
    
        
    return
  
def save_sql_db_into_parquet_file(connectionParams, output_base_path):
    
    STOCK_MARKET_PARQUET_PATH = os.getenv("STOCK_MARKET_PARQUET_PATH", "none") 
    last_ticker_file = f'{STOCK_MARKET_PARQUET_PATH}/last_ticker_200_sma.txt'
    
    if not os.path.exists(last_ticker_file):
        print("No last ticker file found, starting from the beginning.")
        return
    
    with open(last_ticker_file, 'r') as f:
        last_ticker_info = f.read().strip()
    
    if connectionParams is None:
        return None
    
    
        
    
    ticker = last_ticker_info
    ticker_list  = get_ticker_list_from_db(connectionParams) 
    stock_list = get_ticker_list_from_db(connectionParams)
    tickers = stock_list['ticker'].to_numpy()
    filtered = [s for s in tickers if re.fullmatch(r"[A-Za-z]+", s)]
    ticker_index = 0
    
    if not last_ticker_info:
        print("Last ticker info is empty, starting from the beginning.")
        
    else: 
        ticker_index =  filtered.index(ticker)
        
    if ticker_index >= 0 and ticker_index < len(filtered) - 1:
        filtered = filtered[ticker_index:]
        len_tickers = len(filtered)
        for index in range(0,len_tickers):
            
            ticker = filtered[index]
            print(f'processing ticker: {ticker} , {index+1} of {len_tickers}')
            inject_ticker_200_dsma_in_parquet(ticker, output_base_path, connectionParams)
            
            if index == len_tickers - 1:
                with open(f'{STOCK_MARKET_PARQUET_PATH}/last_ticker_200_sma.txt', 'w') as f:
                    f.write(f'------done----------\n')
            else:
                # save index of last ticker processed
                with open(f'{STOCK_MARKET_PARQUET_PATH}/last_ticker_200_sma.txt', 'w') as f:
                    f.write(f'{ticker}\n')
    
    
    return
     
def inject_ticker_200_dsma_in_parquet(ticker, output_base_path, connectionParams):
    
    if ticker is None or output_base_path is None:
        return None
    
    ticker_items = get_ticker_items_from_db(ticker, connectionParams)
     
    new_folder_path = os.path.join(output_base_path, ticker)
    file_path = os.path.join(new_folder_path,f'{ticker}.parquet')
    
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path, exist_ok=True)
    
    ticker_items.to_parquet(file_path, index=False)
    
    df = pd.read_parquet(file_path)
     
# get the list of stocks in the database
def get_ticker_without_mcap_list_from_db(connectionParams):

    df = pd.DataFrame()
    if connectionParams != None:
            try:


                POSTGREST_H =  connectionParams['POSTGREST_H']  # os.getenv("POSTGREST_H", "none")
                POSTGREST_P =  connectionParams['POSTGREST_P']  # os.getenv("POSTGREST_P", "none")
                POSTGREST_TOKEN = connectionParams['POSTGREST_TOKEN']  # os.getenv("POSTGREST_TOKEN", "none")
                
                #connect_to_DB_1(user=DB_USER, password=DB_PASSWORD, host=POSTGREST_H, port=POSTGREST_P, database=DB_NAME)
                
                # Headers with Authorization
                headers = {
                    "Authorization": f"Bearer {POSTGREST_TOKEN}",
                    "Content-Type": "application/json",
                    'Accept':'application/json'
                }

                # The API endpoint
                url = f'http://{POSTGREST_H}:{POSTGREST_P}/rpc/get_unique_tickers_with_missing_cap'

                response = requests.get(url)
              
                if response.status_code != 200:
                    print("error : get_ticker_without_mcap_list_from_db")
                    print(response.text)
                    return pd.DataFrame()
               
                data = response.json()
                df = pd.DataFrame(data)
               

            except Exception as e:
                print("Error:", e)

    return df['ticker'].to_numpy()

# get the list of stocks in the database
def get_ticker_20_perc_gappers_from_db(connectionParams):

    df = pd.DataFrame()
    if connectionParams != None:
            try:


                POSTGREST_H =  connectionParams['POSTGREST_H']  # os.getenv("POSTGREST_H", "none")
                POSTGREST_P =  connectionParams['POSTGREST_P']  # os.getenv("POSTGREST_P", "none")
                POSTGREST_TOKEN = connectionParams['POSTGREST_TOKEN']  # os.getenv("POSTGREST_TOKEN", "none")
                
                #connect_to_DB_1(user=DB_USER, password=DB_PASSWORD, host=POSTGREST_H, port=POSTGREST_P, database=DB_NAME)
                
                # Headers with Authorization
                headers = {
                    "Authorization": f"Bearer {POSTGREST_TOKEN}",
                    "Content-Type": "application/json",
                    'Accept':'application/json'
                }

                # The API endpoint
                url = f'http://{POSTGREST_H}:{POSTGREST_P}/rpc/get_strong_gap_up_stocks_without_marketcap_float'

                response = requests.get(url)
              
                if response.status_code != 200:
                    print(f"error fetching data get_ticker_20_perc_gappers_from_db: {response.status_code}")
                    print(response.text)
                    return pd.DataFrame()
               
                data = response.json()
                df = pd.DataFrame(data)
               

            except Exception as e:
                print("Error:", e)

    return df
  
def gappers_pm_and_market_hours(connectionParams):

    df = pd.DataFrame()
    if connectionParams != None:
            try:


                POSTGREST_H =  connectionParams['POSTGREST_H']  # os.getenv("POSTGREST_H", "none")
                POSTGREST_P =  connectionParams['POSTGREST_P']  # os.getenv("POSTGREST_P", "none")
                POSTGREST_TOKEN = connectionParams['POSTGREST_TOKEN']  # os.getenv("POSTGREST_TOKEN", "none")
                
                #connect_to_DB_1(user=DB_USER, password=DB_PASSWORD, host=POSTGREST_H, port=POSTGREST_P, database=DB_NAME)
                
                # Headers with Authorization
                headers = {
                    "Authorization": f"Bearer {POSTGREST_TOKEN}",
                    "Content-Type": "application/json",
                    'Accept':'application/json'
                }

                # The API endpoint
                url = f'http://{POSTGREST_H}:{POSTGREST_P}/rpc/get_gappers_list'

                response = requests.get(url)
              
                if response.status_code != 200:
                    print(f"error fetching data get_ticker_20_perc_gappers_from_db: {response.status_code}")
                    print(response.text)
                    return pd.DataFrame()
               
                data = response.json()
                df = pd.DataFrame(data)
                
                
               
               

            except Exception as e:
                print("Error get_ticker_20_perc_gappers_from_db:", e)

    return df
    
def get_ticker_latest_date(connectionParams, ticker):
     
    df = pd.DataFrame()
    date_str = None
    
    if connectionParams != None:
        
        try:
            
            
            POSTGREST_H =  connectionParams['POSTGREST_H']  # os.getenv("POSTGREST_H", "none")
            POSTGREST_P =  connectionParams['POSTGREST_P']  # os.getenv("POSTGREST_P", "none")
            POSTGREST_TOKEN = connectionParams['POSTGREST_TOKEN']  # os.getenv("POSTGREST_TOKEN", "none")

            #connect_to_DB_1(user=DB_USER, password=DB_PASSWORD, host=POSTGREST_H, port=POSTGREST_P, database=DB_NAME)

            # Headers with Authorization
            headers = {
            "Authorization": f"Bearer {POSTGREST_TOKEN}",
            "Content-Type": "application/json",
            'Accept':'application/json'
            }

            # The API endpoint
            url = f'http://{POSTGREST_H}:{POSTGREST_P}/stock_data?ticker=eq.{ticker}&order=time.desc&limit=1'

            response = requests.get(url)

            if response.status_code == 200:
                
                data = response.json()
                df = pd.DataFrame(data)

            
        except Exception as e:
            
            print("Error in get_last_after_hour_runners_date:", e)
            return ""
        
        
                
    if(len(df) != 0):
       return  df.iloc[0]['date_str']

    today = date.today()  
    return  f"{today.year-4}-{today.month:02d}-{today.day:02d}"
   
   
# =========== approach asyncio ========

# ===============  
def refresh_materialized_views(connectionParams):
    """
    refresh the materilized view for gappers and after hour runner 
    """
    
    data = None
    if connectionParams != None :
            try:


                POSTGREST_H =  connectionParams['POSTGREST_H']  # os.getenv("POSTGREST_H", "none")
                POSTGREST_P =  connectionParams['POSTGREST_P']  # os.getenv("POSTGREST_P", "none")
                POSTGREST_TOKEN = connectionParams['POSTGREST_TOKEN']  # os.getenv("POSTGREST_TOKEN", "none")
                
                # Headers with Authorization
                headers = {
                    "Authorization": f"Bearer {POSTGREST_TOKEN}",
                    "Content-Type": "application/json",
                    'Accept':'application/json'
                }

                # The API endpoint
                url = f'http://{POSTGREST_H}:{POSTGREST_P}/rpc/refresh_materialized_view'

                response = requests.post(url)
              
                print(response.status_code)
                if response.status_code != 200:
                    print("error fetching data")
                    return 'Error'
                
                data = response.json()

               
            except Exception as e:
                print("Error:", e)   
    return data

# ===============
def log(text):
    
   
    today = datetime.today()
    _date = today.strftime("%Y-%m-%d %H:%M:%S")
   
    
    data = {
    "Date": [_date],
    "message": [text],
   
    }
    df = pd.DataFrame(data)
    old_data = pd.DataFrame()
    
    old_data =  pd.read_csv("./small_caps_strategies/process_logs.csv")
    df_result = pd.concat([old_data, df], ignore_index=True)
    df_result.to_csv("./small_caps_strategies/process_logs.csv", index=False)
    
   