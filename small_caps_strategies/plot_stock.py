import os
import sys

sys.path.insert(0, os.path.abspath("."))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import utils
from utils import helpers as utils_helpers
from lightweight_charts import Chart
from datetime import datetime, timedelta, time, date
import time as tm
from dotenv import load_dotenv
from multiprocessing import Pool, cpu_count, current_process
from functools import wraps
from small_caps_strategies import runner

load_dotenv() 

connectionParams ={}
connectionParams['POSTGREST_H'] = 'localhost'  # os.getenv("POSTGREST_H", "none")
connectionParams['POSTGREST_P'] = '3030' # os.getenv("POSTGREST_P", "none")
connectionParams['POSTGREST_TOKEN'] =  os.getenv("POSTGREST_TOKEN", "none")



def prepare_data_parabolic_short_5min(ticker, date_str, previous_day_close = 10000000):
    
    _to = datetime.strptime(date_str, "%Y-%m-%d")
    _from = _to - timedelta(days=20)
 
    (df_5min, _) = utils_helpers.get_data_5min_for_backtest(ticker, _from.strftime("%Y-%m-%d"), _to.strftime("%Y-%m-%d"))
    
    VOL_MULT = 3
    ATR_EXT = 3.5
    
    df_5min = df_5min.copy()
    df_5min["time"] = df_5min["date"].dt.time
    atr = utils_helpers.compute_atr(df_5min)
    df_5min['atr'] = atr
   
    df_5min['vwap'] =  utils_helpers.vwap(df_5min)
    df_5min['atr_vwap'] = df_5min['vwap']  + 3.5 * atr
    df_5min['atr_stop'] = df_5min['close']  + 3.5 * atr
    
    
    
    avg_vol = df_5min['volume'].rolling(20).mean()
    df_5min['rel_vol'] = df_5min['volume'] / avg_vol
    
    df_5min["parabolic_spike"] = (
        #(df_1min['rel_vol'] > 2.5) &
        (df_5min["close"] >= df_5min["vwap"] + ATR_EXT * df_5min["atr"])
    )
    
    
     # --- Candle colors ---
    df_5min['is_red'] = df_5min['close'] < df_5min['open']
    df_5min['is_green_1'] = df_5min['close'].shift(1) > df_5min['open'].shift(1)
    df_5min['is_green_2'] = df_5min['close'].shift(2) > df_5min['open'].shift(2)

    # --- Body sizes (open-close only) ---
    df_5min['body_size'] = (df_5min['open'] - df_5min['close']).abs()
    df_5min['prev_body_size'] = df_5min['body_size'].shift(1)
    df_5min['bar_size_vs_prev'] =  (df_5min['close'] - df_5min['open']).abs()/df_5min['prev_body_size']
    df_5min['bar_size_prct'] =  (df_5min['close'] - df_5min['open']).abs()/df_5min['close']
    
    # --- Conditions ---
    cond_red = df_5min['is_red']

    cond_body_size = (
        df_5min['body_size'] >= 1.2 * df_5min['prev_body_size']
    )

    cond_lower_close = (
        (df_5min['close'] < df_5min['close'].shift(1)) &
        (df_5min['close'] < df_5min['close'].shift(2))
    )

    cond_prev_green = (
        df_5min['is_green_1'] | df_5min['is_green_2']
    )


    previous_high =  df_5min['high'].shift(1) 
    previous_prev_high =  df_5min['high'].shift(2) 

    cond_high_above_vwap = (
    pd.concat([
        df_5min['high'],
        df_5min['high'].shift(1),
        df_5min['high'].shift(2)
    ], axis=1).max(axis=1) > df_5min['vwap']    
    )

    cond_body_pct = (
        df_5min['body_size'] / df_5min['close'] > 0.02
    )

    # --- Final signal ---
    df_5min['red_exhaustion_signal'] = (
        cond_red &
        cond_body_size &
        cond_lower_close &
        cond_prev_green &
        cond_high_above_vwap &
        cond_body_pct
    )
    
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
 
    return (df_5min, None)
  
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


@runner.pipeline
def pipeline_backtest_dataset(df = pd.DataFrame([]), id=0):
    
    print(f'************** backtest_dataset  pipeline {id} ******************')
    utils_helpers.log(f' ***** started backtest_dataset pipeline {id}', file_path=f'./pipeline_logs/backtest_dataset_log_{id}.csv')
    start_time = tm.perf_counter()
   
    date_str = None
    ticker = None
    
    df_total =  pd.DataFrame([])
    
    try:
    
        for i in range(len(df)):
            row = df.iloc[i]
            previous_day_close = row['previous_close']
            date_str = row['date_str']
            ticker = row['ticker']
            df_dataset = prepare_data_for_dataset_file_5min(ticker, date_str, previous_day_close=previous_day_close)
            df_total = pd.concat([df_total, df_dataset], ignore_index=True)
              
        end_time = tm.perf_counter()
            
        # 6. CALCULAR Y MOSTRAR LA DURACIÓN
        total_duration = end_time - start_time
        #tds.to_parquet(f'trades/pipeline_{id}.parquet')
        utils_helpers.append_single_parquet(df=df_total, path=f'backtest_dataset/dataset/backtest_dataset_pipeline_{id}.parquet')
        print(f"⏰ Tiempo total de ejecución for pipeline {id} total: {total_duration:.2f} segundos.")
        utils_helpers.log(f"⏰ Tiempo total de ejecución for pipeline {id} total duration: {total_duration:.2f} segundos.", f'./pipeline_logs/backtest_dataset_log_{id}.csv')
        
        print("==============================================")
       
        
    except Exception as e:
                print("pipeline:", e)   
                end_time = tm.perf_counter()
                utils_helpers.log(f'process fail for ticker {ticker} at {date_str}, error: {e}', f'./pipeline_logs/backtest_dataset_log_{id}.csv')
        
                # 6. CALCULAR Y MOSTRAR LA DURACIÓN
                total_duration = end_time - start_time

                print(f"⏰ Tiempo total de ejecución: {total_duration:.2f} segundos.")
                print("=================error=============================")
                

@runner.pipeline
def pipeline_backtest_dataset_long(df = pd.DataFrame([]), id=0):
    
    print(f'************** backtest_dataset  pipeline {id} ******************')
    utils_helpers.log(f' ***** started backtest_dataset pipeline {id}', file_path=f'./pipeline_logs/backtest_dataset_long_log_{id}.csv')
    start_time = tm.perf_counter()
   
    date_str = None
    ticker = None
    
    df_total =  pd.DataFrame([])
    
    try:
    
        for i in range(len(df)):
            row = df.iloc[i]
            previous_day_close = row['previous_close']
            date_str = row['date_str']
            ticker = row['ticker']
            df_dataset = prepare_data_for_dataset_file_5min(ticker, date_str, previous_day_close=previous_day_close)
            df_dataset = utils_helpers.donchainChannel(df_dataset, lookback=5, offset=1)
            df_total = pd.concat([df_total, df_dataset], ignore_index=True)
              
        end_time = tm.perf_counter()
            
        # 6. CALCULAR Y MOSTRAR LA DURACIÓN
        total_duration = end_time - start_time
        #tds.to_parquet(f'trades/pipeline_{id}.parquet')
        utils_helpers.append_single_parquet(df=df_total, path=f'backtest_dataset/dataset/backtest_dataset_long_pipeline_{id}.parquet')
        print(f"⏰ Tiempo total de ejecución for pipeline {id} total: {total_duration:.2f} segundos.")
        utils_helpers.log(f"⏰ Tiempo total de ejecución for pipeline {id} total duration: {total_duration:.2f} segundos.", f'./pipeline_logs/backtest_dataset_long_log_{id}.csv')
        
        print("==============================================")
       
        
    except Exception as e:
                print("pipeline:", e)   
                end_time = tm.perf_counter()
                utils_helpers.log(f'process fail for ticker {ticker} at {date_str}, error: {e}', f'./pipeline_logs/backtest_dataset_long_log_{id}.csv')
        
                # 6. CALCULAR Y MOSTRAR LA DURACIÓN
                total_duration = end_time - start_time

                print(f"⏰ Tiempo total de ejecución: {total_duration:.2f} segundos.")
                print("=================error=============================")

@runner.pipeline
def pipeline_backtest_dataset_out_of_sample(df = pd.DataFrame([]), id=0):
    
    print(f'************** backtest_dataset_out_of_sample  pipeline {id} ******************')
    utils_helpers.log(f' ***** started backtest_dataset_out_of_sample pipeline {id}', file_path=f'./pipeline_logs/backtest_dataset_out_of_sample_log_{id}.csv')
    start_time = tm.perf_counter()
   
    date_str = None
    ticker = None
    df_total =  pd.DataFrame([])
    
    try:
    
        for i in range(len(df)):
            row = df.iloc[i]
            previous_day_close = row['previous_close']
            date_str = row['date_str']
            ticker = row['ticker']
            df_dataset = prepare_data_for_dataset_file_5min(ticker, date_str, previous_day_close=previous_day_close)
            df_total = pd.concat([df_total, df_dataset], ignore_index=True)
              
        end_time = tm.perf_counter()
            
        # 6. CALCULAR Y MOSTRAR LA DURACIÓN
        total_duration = end_time - start_time
        #tds.to_parquet(f'trades/pipeline_{id}.parquet')
        utils_helpers.append_single_parquet(df=df_total, path=f'backtest_dataset/out_of_sample/backtest_dataset_out_of_sample_pipeline_{id}.parquet')
        print(f"⏰ Tiempo total de ejecución for pipeline {id} total: {total_duration:.2f} segundos.")
        utils_helpers.log(f"⏰ Tiempo total de ejecución for pipeline {id} total duration: {total_duration:.2f} segundos.", f'./pipeline_logs/backtest_dataset_out_of_sample_log_{id}.csv')
        
        print("==============================================")
       
        
    except Exception as e:
                print("pipeline:", e)   
                end_time = tm.perf_counter()
                utils_helpers.log(f'process fail for ticker {ticker} at {date_str}, error: {e}', f'./pipeline_logs/backtest_dataset_out_of_sample_log_{id}.csv')
        
                # 6. CALCULAR Y MOSTRAR LA DURACIÓN
                total_duration = end_time - start_time

                print(f"⏰ Tiempo total de ejecución: {total_duration:.2f} segundos.")
                print("=================error=============================")



@runner.pipeline
def pipeline_backtest_dataset_long_out_of_sample(df = pd.DataFrame([]), id=0):
    
    print(f'************** backtest_dataset_out_of_sample  pipeline {id} ******************')
    utils_helpers.log(f' ***** started backtest_dataset_out_of_sample pipeline {id}', file_path=f'./pipeline_logs/backtest_dataset_long_out_of_sample_log_{id}.csv')
    start_time = tm.perf_counter()
   
    date_str = None
    ticker = None
    df_total =  pd.DataFrame([])
    
    try:
    
        for i in range(len(df)):
            row = df.iloc[i]
            previous_day_close = row['previous_close']
            date_str = row['date_str']
            ticker = row['ticker']
            df_dataset = prepare_data_for_dataset_file_5min(ticker, date_str, previous_day_close=previous_day_close)
            df_dataset = utils_helpers.donchainChannel(df_dataset, lookback=5, offset=1)
            df_total = pd.concat([df_total, df_dataset], ignore_index=True)
              
        end_time = tm.perf_counter()
            
        # 6. CALCULAR Y MOSTRAR LA DURACIÓN
        total_duration = end_time - start_time
        #tds.to_parquet(f'trades/pipeline_{id}.parquet')
        utils_helpers.append_single_parquet(df=df_total, path=f'backtest_dataset/out_of_sample/backtest_dataset_long_out_of_sample_pipeline_{id}.parquet')
        print(f"⏰ Tiempo total de ejecución for pipeline {id} total: {total_duration:.2f} segundos.")
        utils_helpers.log(f"⏰ Tiempo total de ejecución for pipeline {id} total duration: {total_duration:.2f} segundos.", f'./pipeline_logs/backtest_databacktest_dataset_long_out_of_sample_log_set_out_of_sample_log_{id}.csv')
        
        print("==============================================")
       
        
    except Exception as e:
                print("pipeline:", e)   
                end_time = tm.perf_counter()
                utils_helpers.log(f'process fail for ticker {ticker} at {date_str}, error: {e}', f'./pipeline_logs/backtest_dataset_long_out_of_sample_log_{id}.csv')
        
                # 6. CALCULAR Y MOSTRAR LA DURACIÓN
                total_duration = end_time - start_time

                print(f"⏰ Tiempo total de ejecución: {total_duration:.2f} segundos.")
                print("=================error=============================")

def plot_data(ticker, date_str):
    
    _to = datetime.strptime(date_str, "%Y-%m-%d")
    _from = _to - timedelta(days=5)
    
    (df_1min, _) = utils_helpers.get_data_for_backtest(ticker, _from.strftime("%Y-%m-%d"), _to.strftime("%Y-%m-%d"), adjusted=False)
    utils_helpers.plot_trades_indicators(df_1min)
   
def plot_trades(df, markers):
    
    df['time'] = pd.to_datetime(df['date'])
    sma_9 =  utils_helpers.calculate_sma(df, period = 9)
    sma_200 =  utils_helpers.calculate_sma(df, period = 200)
    df = df.drop(columns=['date'], errors='ignore')
   

    vwap =  pd.DataFrame({
        'time': df['time'],
        'vwap': df['vwap']
    }).dropna()
    
    atr_vwap = pd.DataFrame({
        'time': df['time'],
        'atr_vwap': df['atr_vwap']
    }).dropna()
    
    atr_stop = pd.DataFrame({
        'time': df['time'],
        'atr_stop': df['atr_stop']
    }).dropna()
  

    vwap_indicator =  ('vwap', vwap,'yellow')
    sma9_indicator =  ('SMA 9', sma_9,'white')
    sma200_indicator =  ('SMA 200', sma_200,'blue')
    atr_vwap_indicator =  ('atr_vwap', atr_vwap,'red')
    atr_stop_indicator =  ('atr_stop', atr_stop,'green')
    indicators = [vwap_indicator, sma9_indicator, sma200_indicator, atr_vwap_indicator, atr_stop_indicator]
    
    utils_helpers.plot_trades_indicators(df,markers, indicators )
  
def update_dataset_csv():
    
    all_data = utils_helpers.fetch_all_data_from_gappers(connectionParams)
    backtest_data = all_data.iloc[:10000]
    
    x =  backtest_data[backtest_data['gap_perc'] > 50]
    
    print(x)
    # out_of_sample_data = all_data.iloc[10000:20000]
    # #print(backtest_data)

    # #print(out_of_sample_data)
    # backtest_data.to_csv('small_caps_strategies/gappers_backtest_dataset.csv', index=False)
    # out_of_sample_data.to_csv('small_caps_strategies/gappers_out_of_sample_dataset.csv', index=False)
    #gappers_backtest_dataset = pd.read_csv('small_caps_strategies/gappers_backtest_dataset.csv')
    #print(gappers_backtest_dataset)
    #print(f"Total gappers fetched: {len(all_data)}")

    #df =  all_data[(all_data['gap_perc'] > 500) & (all_data['gap_perc'] < 600)]

    #print(all_data.columns)
    # print(df[['ticker', 'date_str', 'gap', 'gap_perc', 'daily_range',
    #        'previous_close',  'open', 'high', 'low', 'volume','close',
    #       'high_mh', 'high_pm',
    #       'day_range_perc',]])
    
    
    return

# trades = pd.read_parquet('trades/strategy_short_break_structure_at_50_end_of_day_pipeline_1.parquet')

# print(trades)

# grouped = trades[trades['ticker'] == 'EYPT']

# markers =  utils_helpers.create_marker_from_signals_from_trades(grouped)



#runner.main(init_worker= runner.init_worker, strategy_func= pipeline_backtest_dataset, n_cpus= 4, chunk_size= 1000)
#runner.out_of_sample(init_worker= runner.init_worker, strategy_func= pipeline_backtest_dataset_out_of_sample, n_cpus= 4, chunk_size= 1000)

#runner.main(init_worker= runner.init_worker, strategy_func= pipeline_backtest_dataset_long, n_cpus= 4, chunk_size= 1000)
#runner.out_of_sample(init_worker= runner.init_worker, strategy_func= pipeline_backtest_dataset_long_out_of_sample, n_cpus= 4, chunk_size= 1000)

# df1  = pd.read_parquet('backtest_dataset/dataset/backtest_dataset_pipeline_1.parquet')
# df2  = pd.read_parquet('backtest_dataset/dataset/backtest_dataset_pipeline_2.parquet')
# df3  = pd.read_parquet('backtest_dataset/dataset/backtest_dataset_pipeline_3.parquet')
# df4  = pd.read_parquet('backtest_dataset/dataset/backtest_dataset_pipeline_4.parquet')
# df_total = pd.concat([df1, df2, df3, df4], ignore_index=True)
# df_total.to_parquet('backtest_dataset/dataset/gappers_backtest_dataset_5min.parquet', index=False)

# df1  = pd.read_parquet('backtest_dataset/out_of_sample/backtest_dataset_out_of_sample_pipeline_5.parquet')
# df2  = pd.read_parquet('backtest_dataset/out_of_sample/backtest_dataset_out_of_sample_pipeline_6.parquet')
# df3  = pd.read_parquet('backtest_dataset/out_of_sample/backtest_dataset_out_of_sample_pipeline_7.parquet')
# df4  = pd.read_parquet('backtest_dataset/out_of_sample/backtest_dataset_out_of_sample_pipeline_8.parquet')
# df_total = pd.concat([df1, df2, df3, df4], ignore_index=True)
# df_total.to_parquet('backtest_dataset/out_of_sample/gappers_backtest_dataset_5min_out_of_sample.parquet', index=False)

# df1  = pd.read_parquet('backtest_dataset/dataset/backtest_dataset_long_pipeline_1.parquet')
# df2  = pd.read_parquet('backtest_dataset/dataset/backtest_dataset_long_pipeline_2.parquet')
# df3  = pd.read_parquet('backtest_dataset/dataset/backtest_dataset_long_pipeline_3.parquet')
# df4  = pd.read_parquet('backtest_dataset/dataset/backtest_dataset_long_pipeline_4.parquet')
# df_total = pd.concat([df1, df2, df3, df4], ignore_index=True)
# df_total.to_parquet('backtest_dataset/dataset/gappers_backtest_dataset_long_5min.parquet', index=False)


# df1  = pd.read_parquet('backtest_dataset/out_of_sample/backtest_dataset_long_out_of_sample_pipeline_5.parquet')
# df2  = pd.read_parquet('backtest_dataset/out_of_sample/backtest_dataset_long_out_of_sample_pipeline_6.parquet')
# df3  = pd.read_parquet('backtest_dataset/out_of_sample/backtest_dataset_long_out_of_sample_pipeline_7.parquet')
# df4  = pd.read_parquet('backtest_dataset/out_of_sample/backtest_dataset_long_out_of_sample_pipeline_8.parquet')
# df_total = pd.concat([df1, df2, df3, df4], ignore_index=True)
# df_total.to_parquet('backtest_dataset/out_of_sample/gappers_backtest_dataset_long_5min_out_of_sample.parquet', index=False)

df = pd.read_parquet('backtest_dataset/dataset/gappers_backtest_dataset_5min.parquet')
df['date'] =  pd.to_datetime(df['date'])
df_930 =  df[df['date'].dt.time == pd.to_datetime("9:30").time()]
df_930['prct'] = 100*(df_930['open'] - df_930['previous_day_close'])/df_930['previous_day_close']



df_trades = df_930[(df_930['prct'] > 50 ) & (df_930['volume'] > 40000) & (df_930['RVOL_daily'] >=3) ][['open','previous_day_close','prct', 'date_str','date', 'ticker','volume', 'RVOL_daily']]
df_trades = df_trades.sort_values(by=['ticker', 'date'])



# df_vbt = pd.read_parquet('vectorbt_trades/gap_and_crap_0.2_3.5.parquet')
# df_vbt = df_vbt[['ticker','entry_time','entry_price','previous_day_close']]
# df_vbt['date_str'] = pd.to_datetime(pd.to_datetime(df_vbt["entry_time"])).dt.strftime('%Y-%m-%d') 
# df_vbt = df_vbt.sort_values(by=['ticker', 'entry_time'])




# # df1 = pd.read_parquet('backtest_dataset/dataset/gappers_backtest_dataset_5min_v1.parquet')
# # df1['date'] =  pd.to_datetime(df1['date'])
# # df1_930 =  df1[df1['date'].dt.time == pd.to_datetime("9:30").time()]
# # df1_930['prct'] = 100*(df1_930['open'] - df1_930['previous_day_close'])/df1_930['previous_day_close']

# # print(df_930[['open','previous_day_close','prct', 'date_str', 'ticker']])
# # print(df1_930[['open','previous_day_close','prct', 'date_str', 'ticker']])
print(len(df_trades))
print(df_trades[0:50])

# print(len(df_vbt))
# print(df_vbt)



# df_allk = df_930[df_930['ticker'] == 'ABOS']
# print(df_allk[['open','previous_day_close','prct', 'date_str','date', 'ticker']])

ticker = 'AAOI'
date_str = '2022-09-16'
(df_5min, _) = prepare_data_parabolic_short_5min(ticker, date_str, date_str)
plot_trades(df_5min, markers=[])



# update_dataset_csv()
#sync_prev_close()
#sync_v1()

#plot_data('RVPH', '2022-01-10')
#plot_data('STRR', '2023-06-20')

#grouped =  df.groupby(['ticker','date_str'])

# for (ticker, date_str) , g_df in grouped:
#     print(ticker)
#     print(g_df[g_df['date'].dt.time == pd.to_datetime("9:30").time()])
    
# print(len(grouped))



