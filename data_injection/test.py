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
import time as tm
import pandas as pd
import numpy as np
import helpers as helpers
from small_caps_strategies import runner
from dotenv import load_dotenv
from multiprocessing import current_process
from pathlib import Path
import yfinance as yf


load_dotenv() 
connectionParams ={}
connectionParams['POSTGREST_H'] = 'localhost'  # os.getenv("POSTGREST_H", "none")
connectionParams['POSTGREST_P'] = '3030' # os.getenv("POSTGREST_P", "none")
connectionParams['POSTGREST_TOKEN'] =  os.getenv("POSTGREST_TOKEN", "none")

def prepare_data_for_dataset_file_5min(ticker, date_str, previous_day_close = 10000000):
    
    _to = datetime.strptime(date_str, "%Y-%m-%d")
    _from = _to - timedelta(days=20)
 
    (df_5min, _) = utils_helpers.get_data_5min_for_backtest(ticker, _from.strftime("%Y-%m-%d"), _to.strftime("%Y-%m-%d"))
        
    df_5min = df_5min.copy()
    df_5min["time"] = df_5min["date"].dt.time
    atr = utils_helpers.compute_atr(df_5min)
    df_5min['atr'] = atr
    df_5min['vwap'] =  utils_helpers.vwap(df_5min)
    df_5min['SMA_VOLUME_20_5m'] = df_5min['volume'].rolling(20).mean()
    df_5min['SMA_VOLUME_50_5m'] = df_5min['volume'].rolling(50).mean()
    df_5min['SMA_VOLUME_200_5m'] = df_5min['volume'].rolling(200).mean()
    df_5min['previous_day_close'] = previous_day_close
    df_5min['date_str'] = date_str
    df_5min['ticker'] = ticker
   
    

    # --- Body sizes (open-close only) ---
    body_size = (df_5min['open'] - df_5min['close']).abs()
    prev_body_size = body_size.shift(1)
    df_5min['bar_size_vs_prev'] =  (df_5min['close'] - df_5min['open']).abs()/prev_body_size
    df_5min['bar_size_prct'] =  (df_5min['close'] - df_5min['open']).abs()/df_5min['close']
    df_5min = df_5min[df_5min["date"].dt.date == pd.to_datetime(date_str).date()]
    
    _to_daily = _to - timedelta(days=1)
    _from_daily = _to - timedelta(days=50)
    # to get RVOL of daily bars
    (df_daily, _) = utils_helpers.get_data_daily_for_backtest(ticker, _from_daily.strftime("%Y-%m-%d"), _to_daily.strftime("%Y-%m-%d"))
    df_daily['SMA_VOLUME_20_daily'] = df_daily['volume'].rolling(20).mean()
    
    lastrow = df_daily.iloc[-1]['SMA_VOLUME_20_daily']
    df_5min['SMA_VOLUME_20_daily'] = lastrow
    df_5min['cummulative_vol'] = df_5min['volume'].cumsum()
    df_5min['RVOL_daily'] = df_5min['cummulative_vol'] / lastrow
 
    return df_5min

def test_previous_close_sync(ticker, start_date, end_date):
    
    raw_data = utils.fetch_ticker_data_30(ticker, start_date, end_date)
    #processed_data = utils.process_data_30_minutes(raw_data)
    processed_data = utils.process_data_minutes(raw_data)
    
            
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

def test_db():

    load_dotenv() 

    connectionParams ={}
    connectionParams['POSTGREST_H'] = 'localhost'  # os.getenv("POSTGREST_H", "none")
    connectionParams['POSTGREST_P'] = '3030' # os.getenv("POSTGREST_P", "none")
    connectionParams['POSTGREST_TOKEN'] =  os.getenv("POSTGREST_TOKEN", "none")

    colums =  ['ticker', 'date_str', 'gap', 'gap_perc', 'daily_range','day_range_perc','gap_pm',
         'previous_close', 'open', 'close', 'low', 'high','high_pm', 
       'highest_in_pm','volume',  'premarket_volume' , 'date' ]

    all_data = utils_helpers.fetch_all_data_from_gappers(connectionParams)
    all_data['date'] = pd.to_datetime(all_data['date_str'])
    all_data = all_data.sort_values(by='date', ascending=True)
    all_data =  all_data[all_data['previous_close'] > 0]
    all_data['gap_pm'] = 100 * (all_data['high_pm'] - all_data['previous_close']) / all_data['previous_close']
    print(all_data.columns)
    print(all_data[colums])
    backtest_data = all_data.iloc[:10000]
    print(backtest_data[colums])
    
    
    df_in_sample_1 = all_data[all_data['date'].between('2021-01-01', '2022-01-01')]
    df_out_sample_1 = all_data[all_data['date'].between('2022-01-02', '2022-07-01')]
    
    df_in_sample_1.to_parquet('backtest_dataset/walk_fordward/in_sample_1.parquet')
    df_out_sample_1.to_parquet('backtest_dataset/walk_fordward/out_sample_1.parquet')
    
    
    df_in_sample_2 = all_data[all_data['date'].between('2021-07-01', '2022-07-01')]
    df_out_sample_2 = all_data[all_data['date'].between('2022-07-02', '2023-01-01')]
    
    df_in_sample_2.to_parquet('backtest_dataset/walk_fordward/in_sample_2.parquet')
    df_out_sample_2.to_parquet('backtest_dataset/walk_fordward/out_sample_2.parquet')
    
    df_in_sample_3 = all_data[all_data['date'].between('2022-01-01', '2023-01-01')]
    df_out_sample_3 = all_data[all_data['date'].between('2023-01-02', '2023-07-01')]
    
    df_in_sample_3.to_parquet('backtest_dataset/walk_fordward/in_sample_3.parquet')
    df_out_sample_3.to_parquet('backtest_dataset/walk_fordward/out_sample_3.parquet')
    
    return
    
def cleaning_data(path="backtest_dataset/walk_fordward"):

    df_in_sample_1 = pd.read_parquet(os.path.join(path, 'walk_fordward_in_sample_1.parquet'))
    df_in_sample_1['date'] = pd.to_datetime(df_in_sample_1['date'])
    df_in_sample_1.sort_values(by=['date_str','ticker','date'], inplace=True)
    df_in_sample_1.drop_duplicates(inplace=True)
    df_in_sample_1.to_parquet('backtest_dataset/walk_fordward/walk_fordward_in_sample_1_v1.parquet')
    
    df_out_sample_1 = pd.read_parquet(os.path.join(path,'walk_fordward_out_sample_1.parquet'))
    df_out_sample_1['date'] = pd.to_datetime(df_out_sample_1['date'])
    df_out_sample_1.sort_values(by=['date_str','ticker','date'], inplace=True)
    df_out_sample_1.drop_duplicates(inplace=True)
    correct_df = df_out_sample_1[df_out_sample_1['date_str'] > '2022-01-01' ]
    correct_df.to_parquet('backtest_dataset/walk_fordward/walk_fordward_out_sample_1_v1.parquet')
    
    correct_df =  pd.read_parquet('backtest_dataset/walk_fordward/walk_fordward_out_sample_1_v1.parquet')
    
    print(" ======= 1 ======= ")
    print(df_in_sample_1[['ticker','date_str','date']])
    print(" ======= correct_df 1======= ")
    print(correct_df[['ticker','date_str','date']])
    
    
    df_in_sample_2 = pd.read_parquet(os.path.join(path, 'walk_fordward_in_sample_2.parquet'))
    df_in_sample_2['date'] = pd.to_datetime(df_in_sample_2['date'])
    df_in_sample_2.sort_values(by=['date_str','ticker','date'], inplace=True)
    df_in_sample_2.drop_duplicates(inplace=True)
    df_in_sample_2.to_parquet('backtest_dataset/walk_fordward/walk_fordward_in_sample_2_v1.parquet')
    
    df_out_sample_2 = pd.read_parquet(os.path.join(path,'walk_fordward_out_sample_2.parquet'))
    df_out_sample_2['date'] = pd.to_datetime(df_out_sample_2['date'])
    df_out_sample_2.sort_values(by=['date_str','ticker','date'], inplace=True)
    df_out_sample_2.drop_duplicates(inplace=True)
    
    print(" ======= 2 ======= ")
    print(df_in_sample_2[['ticker','date_str','date']])
    #print(df_out_sample_2[df_out_sample_2['date_str'] > '2022-07-01' ][['ticker','date_str','date']])
    correct_df = df_out_sample_2[df_out_sample_2['date_str'] > '2022-07-01' ]
    correct_df.to_parquet('backtest_dataset/walk_fordward/walk_fordward_out_sample_2_v1.parquet')
    correct_df =  pd.read_parquet('backtest_dataset/walk_fordward/walk_fordward_out_sample_2_v1.parquet')
    print(" ======= correct_df 2======= ")
    print(correct_df[['ticker','date_str','date']])
   
    
    df_in_sample_3 = pd.read_parquet(os.path.join(path, 'walk_fordward_in_sample_3.parquet'))
    df_in_sample_3['date'] = pd.to_datetime(df_in_sample_3['date'])
    df_in_sample_3.sort_values(by=['date_str','ticker','date'], inplace=True)
    df_in_sample_3 = df_in_sample_3[df_in_sample_3['date_str'] > '2022-01-01']
    df_in_sample_3.drop_duplicates(inplace=True)
   
    df_in_sample_3.to_parquet('backtest_dataset/walk_fordward/walk_fordward_in_sample_3_v1.parquet')
    
   
    df_out_sample_3 = pd.read_parquet(os.path.join(path,'walk_fordward_out_sample_3.parquet'))
    df_out_sample_3['date'] = pd.to_datetime(df_out_sample_3['date'])
    df_out_sample_3.sort_values(by=['date_str','ticker','date'], inplace=True)
    df_out_sample_3.drop_duplicates(inplace=True)
    
    correct_df = df_out_sample_3[df_out_sample_3['date_str'] > '2023-01-01' ]
    correct_df.to_parquet('backtest_dataset/walk_fordward/walk_fordward_out_sample_3_v1.parquet')
    
    correct_df =  pd.read_parquet('backtest_dataset/walk_fordward/walk_fordward_out_sample_3_v1.parquet')
    
    print(" ======= 3 ======= ")
    print(df_in_sample_3[['ticker','date_str','date']])
    print(" ======= correct_df 3======= ")
    print(correct_df[['ticker','date_str','date']])
    
    return

def test_3():
    
    # print(" ======= 1 ======= ")
    # df_in_sample_1 = pd.read_parquet('backtest_dataset/walk_fordward/in_sample_1.parquet')
    # print(df_in_sample_1)
    # df_out_sample_1 = pd.read_parquet('backtest_dataset/walk_fordward/out_sample_1.parquet')
    # print(df_out_sample_1)
    
    print(" ======= 2 ======= ")
    df_in_sample_2 = pd.read_parquet('backtest_dataset/walk_fordward/in_sample_2.parquet')
    print(df_in_sample_2[['ticker','date_str','date']])
    df_out_sample_2 = pd.read_parquet('backtest_dataset/walk_fordward/out_sample_2.parquet')
    print(df_out_sample_2[['ticker','date_str','date']])
    
    # print(" ======= 3 ======= ")
    # df_in_sample_3 = pd.read_parquet('backtest_dataset/walk_fordward/in_sample_3.parquet')
    # print(df_in_sample_3)
    # df_out_sample_3 = pd.read_parquet('backtest_dataset/walk_fordward/out_sample_3.parquet')
    # print(df_out_sample_3)
    
    return

def test_4():
    
    print(" ======= 1 ======= ")
    df_1 = pd.read_parquet('backtest_dataset/walk_fordward/walk_fordward_in_sample_1_v1.parquet')
    df_1_1 = pd.read_parquet('backtest_dataset/walk_fordward/walk_fordward_out_sample_1_v1.parquet')
    print(df_1[['ticker','date_str','date']])
    print(df_1_1[['ticker','date_str','date']])
    
    print(" ======= 2 ======= ")
    df_2 = pd.read_parquet('backtest_dataset/walk_fordward/walk_fordward_in_sample_2_v1.parquet')
    df_2_2 = pd.read_parquet('backtest_dataset/walk_fordward/walk_fordward_out_sample_2_v1.parquet')
    print(df_2[['ticker','date_str','date']])
    print(df_2_2[['ticker','date_str','date']])
    
    print(" ======= 3 ======= ")
    df_3 = pd.read_parquet('backtest_dataset/walk_fordward/walk_fordward_in_sample_3_v1.parquet')
    df_3_3 = pd.read_parquet('backtest_dataset/walk_fordward/walk_fordward_out_sample_3_v1.parquet')
    print(df_3[['ticker','date_str','date']])
    print(df_3_3[['ticker','date_str','date']])
    return

def join_walk_fordward_files(path="backtest_dataset/walk_fordward", sample_type='in_sample') -> pd.DataFrame:
    
    df_storage = []
    
    for i in range(1,4):
        for j in range(1,9):
            file_pattern = f'walk_fordward_{sample_type}_{i}_5min_{j}.parquet'
            
            file_path = Path(f'{path}/walk_fordward_{sample_type}_{i}_5min_{j}.parquet')
            
            if file_path.exists():
               print(f"Leyendo el fichero: {file_pattern}" )
               data = pd.read_parquet(file_path)
               data['date'] = pd.to_datetime(data['date'])
               df_storage.append(data)
               
            else:
                print(f"No existe el fichero: {file_pattern}")
           
            if len(df_storage) >0:
                combined_df = pd.concat(df_storage, ignore_index=True)
                combined_df.to_parquet(os.path.join(path, f'walk_fordward_{sample_type}_{i}.parquet'))
            
            
    return 


@runner.pipeline
def pipeline_backtest_dataset_walk_fordward(task, id):
   
    print(f'🚀  ************** backtest_dataset_walk_fordward  pipeline {id} ******************')
    df, walk_fordward_step, file_name = task
    worker_id = os.getpid()

    start_time = tm.perf_counter()
    dfs = []

    try:
        print("✅ ENTERED TRY")
        for i in range(len(df)):
            #print(f'Processing row {i+1}/{len(df)} | worker {worker_id}')
            row = df.iloc[i]

            df_dataset = prepare_data_for_dataset_file_5min(
                row['ticker'],
                row['date_str'],
                previous_day_close=row['previous_close']
            )
            df_dataset = utils_helpers.donchainChannel(df_dataset, lookback=5, offset=1)
            dfs.append(df_dataset)

        df_total = pd.concat(dfs, ignore_index=True)

        utils_helpers.append_single_parquet(
            df=df_total,
            path=f'backtest_dataset/walk_fordward/walk_fordward_{file_name}_5min_{id}.parquet'
        )

        end_time = tm.perf_counter()
        print(f"⏰ Worker {id} finished in {end_time - start_time:.2f}s")

    except Exception as e:
        print(f"❌ Worker {id} error:", e)
                   
def test5():
    
    stock_list = utils.get_ticker_list_from_db(connectionParams)
    print(stock_list)
    
    for index, row in stock_list.iterrows():
        ticker = row['ticker']
        present = datetime.today().strftime('%Y-%m-%d')
        ticker_data = yf.download(ticker, start="2000-01-01", end=present)
        ticker_data.columns = [col[0] for col in ticker_data.columns]
        ticker_data['previous_day_close'] =  ticker_data['Close'].shift(1)
        ticker_data['gap'] =  100 * (ticker_data['Open']  - ticker_data['previous_day_close']) /ticker_data['previous_day_close'] 
        ticker_data['ticker'] =  ticker
        ticker_data = ticker_data.reset_index()
        ticker_data = ticker_data.dropna()
        #print(ticker_data)
        utils_helpers.append_single_parquet(df=ticker_data, path=f'yf_gappers.parquet')
        print(f'{ticker}-{index}')
        
    return
#runner.walk_fordward_runner(init_worker= runner.init_worker, strategy_func=pipeline_backtest_dataset_walk_fordward, n_cpus= 8, chunk_size= 1000, sample_type='in_sample', walk_fordward_step = 1) 
#runner.walk_fordward_runner(init_worker= runner.init_worker, strategy_func= pipeline_backtest_dataset_walk_fordward, n_cpus= 8, chunk_size= 1000, sample_type='in_sample', walk_fordward_step = 2) 
#runner.walk_fordward_runner(init_worker= runner.init_worker, strategy_func= pipeline_backtest_dataset_walk_fordward, n_cpus= 8, chunk_size= 1000, sample_type='in_sample', walk_fordward_step = 3) 

#runner.walk_fordward_runner(init_worker= runner.init_worker, strategy_func= pipeline_backtest_dataset_walk_fordward, n_cpus= 8, chunk_size= 1000, sample_type='out_sample', walk_fordward_step = 1) 
#runner.walk_fordward_runner(init_worker= runner.init_worker, strategy_func= pipeline_backtest_dataset_walk_fordward, n_cpus= 8, chunk_size= 1000, sample_type='out_sample', walk_fordward_step = 2) 
#runner.walk_fordward_runner(init_worker= runner.init_worker, strategy_func= pipeline_backtest_dataset_walk_fordward, n_cpus= 8, chunk_size= 1000, sample_type='out_sample', walk_fordward_step = 3) 
   
#test_db() 
#test_previous_close_sync('RVPH', '2022-01-08', '2022-01-10')   
#test_plot()

#join_walk_fordward_files()
#join_walk_fordward_files(sample_type='out_sample')

#cleaning_data()

#test_3()

#test_4()

# data = utils.fetch_ticker_data_1min('PHIO', '2026-02-10',  '2026-02-10', adjusted=False)
# data = utils.process_data_minutes(data)
# print(data)

test5()

