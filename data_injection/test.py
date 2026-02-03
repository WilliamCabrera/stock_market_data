import pandas as pd
import asyncio
import aiohttp
import multiprocessing
from typing import List, Tuple, Dict, Any
import os
import sys
sys.path.insert(0, os.path.abspath("."))
from datetime import datetime, timedelta, time, date
import re
from utils import utils, helpers as utils_helpers
import json
import time
import pandas as pd
import numpy as np
import helpers as helpers


def test_previous_close_sync(ticker, start_date, end_date):
    
    raw_data = utils.fetch_ticker_data_30(ticker, start_date, end_date)
    processed_data = utils.process_data_30_minutes(raw_data)
            
    # Añadir información clave para la consolidación
    processed_data['ticker'] = ticker
    
    print(" ========= test_previous_close_sync ===========")
    print(processed_data[['open','close','date_str','previous_close']])
    
    df = utils.sync_data_with_prev_day_close(processed_data)
    print(df[['open','close','date_str','previous_close']])
       
def test_plot():
    
    df = utils.fetch_ticker_data_5min('TQQQ','2026-01-30', '2026-01-30')
    df =  pd.DataFrame(df)
    df.rename(columns={'o': 'open', 'c': 'close', 'h': 'high', 'l': 'low', 'v': 'volume', 't': 'time'}, inplace=True)

    if df.empty :
        return df

    df["date"] =  pd.to_datetime(df["time"], unit='ms', utc=True) 
    df["date"] = df["date"].dt.tz_convert("America/New_York")
    df["time"] = pd.to_datetime(df['date'])
    df =  utils_helpers.donchainChannel(df)
    df = df.drop(columns=['date'], errors='ignore')
    
    dch_1 =   pd.DataFrame({
        'time': df['time'],
        'donchian_upper': df['donchian_upper']
    }).dropna()
    
    line1 =  ('donchian_upper', dch_1, 'blue')
    
    dch_2 =   pd.DataFrame({
        'time': df['time'],
        'donchian_lower': df['donchian_lower']
    }).dropna()
    
    line2 =  ('donchian_lower', dch_2, 'blue')
    
    dch_3 =   pd.DataFrame({
        'time': df['time'],
        'donchian_basis': df['donchian_basis']
    }).dropna()
    
    line3 =  ('donchian_basis', dch_3, '#FF6D00')
    
    utils_helpers.plot_trades_indicators(df, markers=[], indicators=[line1, line2, line3])
    
    
    return

#test_previous_close_sync('RVPH', '2022-01-08', '2022-01-10')   
#test_plot()