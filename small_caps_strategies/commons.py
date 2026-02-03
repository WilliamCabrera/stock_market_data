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



# REMARKS: revisar divergencias de volumen cuando el hay mucha extension: close  >= ATR* factor + VWAP
# guardar el volumen mas alto de las utlimas 50 velas , cuando hace un nuevo high con menos volumen => mirar que pasa

def create_marker_from_signals(df):
    
    if df is None or df.empty:
        return
    
    df = df.copy()
    df =  df[df['red_exhaustion_signal']]
    df['date'] = pd.to_datetime(df['date'])
    df_len = len(df)
    markers =[]
    
    for idx in range(0,df_len):
        row = df.iloc[idx]
        
        entry_marker = {"time":  row['date'],
                        "position": "above",
                        "shape": "arrow_down" ,
                        "color": "red", 
                        "text":""
                        }
        
        markers.append(entry_marker)
        
    
    return markers

def create_marker_from_signals_from_trades(trades):
    
    df = pd.DataFrame(trades)
    
    if df is None or df.empty:
        return
    
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df['exit_time'] = pd.to_datetime(df['exit_time'])
    df_len = len(df)
    df['type'] = df['type'].str.lower()
    markers =[]
    
    for idx in range(0,df_len):
        row = df.iloc[idx]
        
        entry_marker = {"time":  row['entry_time'],
                        "position":  "above" if row['type'] == 'short' else "below",
                        "shape": "arrow_down"  if row['type'] == 'short' else "arrow_up",
                        "color": "red"  if row['type'] == 'short' else "green", 
                        "text":"open_short"  if row['type'] == 'short' else "open_long"
                        }

        entry_marker_close = {"time":  row['exit_time'],
                        "position":  "above" if row['type'] == 'short' else "below",
                        "shape": "arrow_up"  if row['type'] == 'short' else "arrow_down",
                        "color": "green"  if row['type'] == 'short' else "red", 
                        "text":"close_short"  if row['type'] == 'short' else "close_long"
                        }
        
        markers.append(entry_marker)
        markers.append(entry_marker_close)
        
    
    return markers

def volume_zscore(df, n=30):
    mean = df['volume'].rolling(n).mean()
    std  = df['volume'].rolling(n).std()
    return (df['volume'] - mean) / std

def prepare_data_parabolic_short(ticker, date_str):
    _to = datetime.strptime(date_str, "%Y-%m-%d")
    _from = _to - timedelta(days=20)

    (df_1min, _) = utils_helpers.get_data_for_backtest(ticker, _from.strftime("%Y-%m-%d"), _to.strftime("%Y-%m-%d"))
    
    PCT_BARS = 5
    PCT_THRESHOLD = 0.25     # 25%
    VOL_MULT = 3
    ATR_EXT = 3.5

    df_1min = df_1min.copy()
    df_1min["time"] = df_1min["date"].dt.time
    atr = utils_helpers.compute_atr(df_1min)
    df_1min['atr'] = atr
    df_1min['vwap'] = utils_helpers.vwap(df_1min)
    #df_1min['atr_pct'] = df_1min['atr']/df_1min['close']
    
    df_1min['vol_score'] = volume_zscore(df_1min, 50)
    
    
    df_1min["vol_ma20"] =  df_1min['volume'].rolling(20).mean()
    #
    avg_vol = df_1min['volume'].rolling(50).mean()
    df_1min['rel_vol'] = df_1min['volume'] / avg_vol
    
   
    df_1min["pct_move"] = df_1min["close"] / df_1min["close"].shift(PCT_BARS) - 1
    
    df_1min["vol_spike"] = df_1min["volume"] >= VOL_MULT * df_1min["vol_ma20"]
    
    df_1min["parabolic_spike"] = (
        #(df_1min['rel_vol'] > 2.5) &
        (df_1min["close"] >= df_1min["vwap"] + ATR_EXT * df_1min["atr"])
    )

    # df_1min["parabolic_spike"] = (
    #     (df_1min["pct_move"] >= PCT_THRESHOLD) &
    #     (df_1min["volume"] >= VOL_MULT * df_1min["vol_ma20"]) &
    #     (df_1min["close"] >= df_1min["vwap"] + ATR_EXT * df_1min["atr"])
    # )
    
    _to_daily = _to - timedelta(days=1)
    _from_daily = _to - timedelta(days=50)
    # to get RVOL of daily bars
    (df_daily, _) = utils_helpers.get_data_daily_for_backtest(ticker, _from_daily.strftime("%Y-%m-%d"), _to_daily.strftime("%Y-%m-%d"))
    df_daily['SMA_VOLUME_20_daily'] = df_daily['volume'].rolling(20).mean()
    
    lastrow = df_daily.iloc[-1]['SMA_VOLUME_20_daily']
    df_1min['SMA_VOLUME_20_daily'] = lastrow
    df_1min['cummulative_vol'] = df_1min['volume'].cumsum()
    df_1min['RVOL_daily'] = df_1min['cummulative_vol'] / lastrow

    return (df_1min, None)

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
    df_5min['vwap'] =  df_5min['vw'] 
    df_5min['vwap'] =  utils_helpers.vwap(df_5min)
    df_5min['atr_vwap'] = df_5min['vwap']  + 3.5 * atr
    df_5min['atr_stop'] = df_5min['close']  + 3.5 * atr
    
    
    
    #df_5min['atr_pct'] = df_5min['atr']/df_5min['close']
    df_5min['vol_score'] = volume_zscore(df_5min, 20)
    
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
    
def prepare_data_parabolic_short_5min_v1(ticker, date_str, previous_day_close = 10000000):
    
    _to = datetime.strptime(date_str, "%Y-%m-%d")
    _from = _to - timedelta(days=20)
 
    (df_5min, _) = utils_helpers.get_data_5min_for_backtest(ticker, _from.strftime("%Y-%m-%d"), _to.strftime("%Y-%m-%d"))
    
    VOL_MULT = 3
    ATR_EXT = 3.5
    
    df_5min = df_5min.copy()
    df_5min["time"] = df_5min["date"].dt.time
    atr = utils_helpers.compute_atr(df_5min)
    df_5min['atr'] = atr
    df_5min['vwap'] =  df_5min['vw'] 
    df_5min['vwap'] =  utils_helpers.vwap(df_5min)
    df_5min['atr_vwap'] = df_5min['vwap']  + 3.5 * atr
    df_5min['atr_stop'] = df_5min['close']  + 3.5 * atr
    df_5min['date_str'] = date_str
    avg_vol = df_5min['volume'].rolling(20).mean()
    df_5min['SMA_VOLUME_20_5m'] = avg_vol
    df_5min['rel_vol'] = df_5min['volume'] / avg_vol
    
    # still decent volume 
    df_5min["climactic_volume"] = df_5min['rel_vol'] >= 2

    
     # --- Candle colors ---
    df_5min['is_red'] = df_5min['close'] < df_5min['open']

    # --- Body sizes (open-close only) ---
    df_5min['body_size'] = (df_5min['open'] - df_5min['close']).abs()
    df_5min['prev_body_size'] = df_5min['body_size'].shift(1)
    df_5min['bar_size_vs_prev'] =  (df_5min['close'] - df_5min['open']).abs()/df_5min['prev_body_size']
    df_5min['bar_size_prct'] =  (df_5min['close'] - df_5min['open']).abs()/df_5min['close']
    
    
    df_5min["body_avg"] = df_5min["body_size"].rolling(10).mean()
    
    # price lossing momentum (weak progress)
    df_5min["weak_progress"] = (
        (df_5min["high"] > df_5min["high"].shift(1)) &
        (df_5min["body_size"] < df_5min["body_avg"])
    )
    
    # price rejection (toping tail, weak close)
    df_5min["weak_close"] = (
        (df_5min["close"] - df_5min["low"]) / (df_5min["high"] - df_5min["low"]) < 0.35
    )
    
    # at least 25% above vwap
    df_5min["vwap_extension"] = (
        (df_5min["close"] - df_5min["vwap"]) / df_5min["vwap"] > 0.25
    )
    
    # failure to break higher (no follow through)
    df_5min["no_follow_through"] = (
        (df_5min["high"] > df_5min["high"].shift(1)) &
        (df_5min["close"] < df_5min["close"].shift(1))
    )

    # --- score de agotamiento ---
    df_5min["exhaustion_score"] = (
        df_5min["climactic_volume"].astype(int) +
        df_5min["weak_progress"].astype(int) +
        df_5min["weak_close"].astype(int) +
        df_5min["vwap_extension"].astype(int) +
        df_5min["no_follow_through"].astype(int)
    )

    df_5min["red_exhaustion_signal"] = df_5min["exhaustion_score"] >= 3

    # --- Conditions ---
    cond_red = df_5min['is_red']
    
   
    
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

def plot_trades(df, markers):
    
    #print("**************8")
    #print(df[['date']])
    
    if markers is None or len(markers) == 0:
        print(" ====== no trades =====")
        return
    
    df['time'] = pd.to_datetime(df['date'])
    sma_9 =  utils_helpers.calculate_sma(df, period = 9)
    sma_200 =  utils_helpers.calculate_sma(df, period = 200)
    df = df.drop(columns=['date'], errors='ignore')
    chart = Chart()
    chart.legend(visible=True)
    chart.set(df)
    
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
  
    
    lineVwap = chart.create_line(name='vwap',color='yellow')
    lineVwap.set(vwap)
    
    lineSMA9 = chart.create_line(name='SMA 9',color='white')
    lineSMA9.set(sma_9)
    
    lineSMA200 = chart.create_line(name='SMA 200',color='blue')
    lineSMA200.set(sma_200)
    
    lineatr_vwap = chart.create_line(name='atr_vwap',color='red')
    lineatr_vwap.set(atr_vwap)
    
    lineatr_stop = chart.create_line(name='atr_stop',color='green')
    lineatr_stop.set(atr_stop)
    
    chart.marker_list(markers)
    chart.show(block=True)
    
def plot_trades_v1(df, markers):
    
    df =  df.copy()

    
    if markers is None or len(markers) == 0:
                print(" ====== no trades =====")
                return
        
    df['time'] = pd.to_datetime(df['date'])
    df = df.drop(columns=['date'], errors='ignore')
    df = df.drop(columns=['day'], errors='ignore')
    df = df.drop(columns=['date_str'], errors='ignore')
    
    
    chart = Chart()
    chart.legend(visible=True)
    chart.set(df)

    chart.marker_list(markers)
    chart.show(block=True)