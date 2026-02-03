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




# compute the indicators that will be used for entry/exit conditions.
# ATR, RVOL, SMA-9, SMA-200, AVERAGE Volume (50)
def prepare_data_for_9sma_200sma_strategy(ticker, date_str, initial_equity=10000, slippage_pct=0.0005, previous_day_close = 10000000):
    _to = datetime.strptime(date_str, "%Y-%m-%d")
    _from = _to - timedelta(days=20)
    
    (df_1min, _) = utils_helpers.get_data_for_backtest(ticker, _from.strftime("%Y-%m-%d"), _to.strftime("%Y-%m-%d"))

    atr = utils_helpers.compute_atr(df_1min)
    df_1min['atr'] = atr
    df_1min['atr_pct'] = df_1min['atr']/df_1min['close']
    
    avg_vol = df_1min['volume'].rolling(30).mean()
    df_1min['rel_vol'] = df_1min['volume'] / avg_vol
    df_1min['spread'] = df_1min['high'] - df_1min['low']
    
    # RVOL at least 2.5x,  ATR incrising (above 60% of last 50 candles), 
    df_1min['can_trade'] = (
    (df_1min['atr_pct'] > df_1min['atr_pct'].rolling(50).quantile(0.6)) &
    (df_1min['rel_vol'] > 2.5) &
    (df_1min['atr'] > df_1min['atr'].shift(1)) & True
   
    )
    
   
   
    """
    df_1m: DataFrame de 1 minuto con columnas 'Open', 'high', 'low', 'close', 'Volume'.
           Debe tener índice DatetimeIndex.
    """
    
    # -----------------------------------------------------------
    # 1. PREPARACIÓN DE DATOS (Multi-Timeframe)
    # -----------------------------------------------------------
    #print("Preparando indicadores...")

    # --- INDICADORES 1 MINUTO ---
    df_1min['SMA_9'] = df_1min['close'].rolling(window=9).mean()
    df_1min['SMA_200'] = df_1min['close'].rolling(window=200).mean()
    df_1min['Vol_SMA_50'] = df_1min['volume'].rolling(window=50).mean()
    df_1min['is_5min'] = df_1min['date'].dt.minute % 5 == 0
    
    
    _to_daily = _to - timedelta(days=1)
    _from_daily = _to - timedelta(days=50)
    # to get RVOL of daily bars
    (df_daily, _) = utils_helpers.get_data_daily_for_backtest(ticker, _from_daily.strftime("%Y-%m-%d"), _to_daily.strftime("%Y-%m-%d"))
    df_daily['SMA_VOLUME_20_daily'] = df_daily['volume'].rolling(20).mean()
    
    lastrow = df_daily.iloc[-1]['SMA_VOLUME_20_daily']
    df_1min['SMA_VOLUME_20_daily'] = lastrow
    df_1min['cummulative_vol'] = df_1min['volume'].cumsum()
    df_1min['RVOL_daily'] = df_1min['cummulative_vol'] / lastrow
    
  
    
    
    # --- INDICADORES 5 MINUTOS ---
    #df_5min = df_5min[['date','close']]
    #df_5min['SMA_9_5min'] = df_5min['close'].rolling(window=9).mean()
    #df_5min.rename(columns={'close': 'close_5min'}, inplace=True)
    
   
   
    df_merged = df_1min #  df_1min.merge(df_5min, on='date', how='outer').sort_values('date')
    df_merged = df_merged[df_merged['date'] > date_str]
    #df_merged['previous_day_close'] = previous_day_close
    
    #df_merged.to_csv(f'{ticker}.csv')
    
    sma_200 =  pd.DataFrame({
        'time': df_merged['date'],
        'SMA 200': df_merged['SMA_200']
    }).dropna()
    
    sma_9_1min =  pd.DataFrame({
        'time': df_merged['date'],
        'SMA_9_1min': df_merged['SMA_9']
    }).dropna()
    
    df_5 = df_1min[df_1min['is_5min']]
    df_sma__5 = df_5['close'].rolling(window=9).mean()
    sma_9_5min =  pd.DataFrame({
        'time': df_merged['date'],
        'SMA_9_5min': df_sma__5
    }).dropna()
    

    
    indicators = {"sma_200":sma_200, "sma_9_1min":sma_9_1min, "sma_9_5min":sma_9_5min}
    
    return (df_merged, indicators)


# take profit 1:1
def strategy_1x_tp(df, ticker ="", initial_equity=10000, slippage_pct=0.0005, previous_day_close=1000000):
    
    
    
    is_green = False
    is_9_above_200 = False
    is_volume_above_avg = False
    is_price_10_perc_above_prev_close = False
    all_conditions = False
    in_opened_position = False
    is_stop_loss_hit = False
    is_take_profit_hit = False
    slippage_pct = 0.02/100
    
    trades = []
    
    current_trade = {}
    
    
    for idx in range(len(df) - 1):
        row = df.iloc[idx]
       
        if in_opened_position == True:
            # ------ exit conditions ------
            # Here you can add your exit conditions, for example:
          
            if row['low'] <= stop_loss_price:
                #print(f"Stop loss hit on: {row['date']}")
                exit_price = df['open'].iloc[idx + 1] * (1 - slippage_pct)  # exit at next candle open price with slippage
                in_opened_position = False
                all_conditions = False
                is_take_profit_hit = False
                current_trade['exit_price'] = exit_price
                current_trade['is_profit'] = False
                current_trade['exit_time'] = row['date']
                current_trade["ticker"]  = ticker
                trades.append(current_trade)
                current_trade = {}
                continue
                
            elif row['close'] >= take_profit_price:
                #print(f"Take profit hit on: {row['date']}")
                exit_price = df['open'].iloc[idx + 1] * (1 - slippage_pct)
                in_opened_position = False
                all_conditions = False
                is_take_profit_hit = True
                current_trade['exit_price'] = exit_price
                current_trade['is_profit'] = True
                current_trade['exit_time'] = row['date']
                current_trade["ticker"]  = ticker
                trades.append(current_trade)
                current_trade = {}
                continue
                
            elif  pd.to_datetime(row['date']).time() == time(16,0):
                #print(f"force to close at market hours close on: {row['date']}")
                exit_price = df['open'].iloc[idx + 1] * (1 - slippage_pct)
                in_opened_position = False
                all_conditions = False
                is_stop_loss_hit = False if exit_price < entry_price else True
                is_take_profit_hit = False if is_stop_loss_hit  == True else True
                current_trade['exit_price'] = exit_price
                current_trade['is_profit'] = is_take_profit_hit
                current_trade['exit_time'] = row['date']
                current_trade["ticker"]  = ticker
                trades.append(current_trade)
                current_trade = {}
                
        elif all_conditions == False and in_opened_position == False and pd.to_datetime(row['date']).time() >= time(7,0) and pd.to_datetime(row['date']).time() <= time(20,0):
            
            # ------ entrie conditions ------
            is_green = row['close'] > row['open']
            is_price_above_9 =  row['close'] > row['SMA_9']
            is_9_above_200 = row['SMA_9'] > row['SMA_200']
            is_volume_above_avg = row['volume'] > row['Vol_SMA_50'] and row['volume'] > 5000
            is_price_10_perc_above_prev_close = row['close'] > previous_day_close + (previous_day_close * 0.10)
            all_conditions = is_green and is_price_above_9 and is_9_above_200 and is_volume_above_avg and is_price_10_perc_above_prev_close and row['can_trade']
            
            # entry in the next candle after conditions met to avoid lookahead bias (that why we use row['open'] instead of row['close'])
            if all_conditions == True:
                stop_loss_price = row['low'] * (1 - slippage_pct) 
                
                entry_price = df['open'].iloc[idx + 1] * (1 - slippage_pct) # 
                #print(f'--- entry: {entry_price}, stopL: {stop_loss_price}')
                take_profit_price = (entry_price - stop_loss_price)*1 + entry_price # Risk-Reward 1:1
                in_opened_position = True
                entry_time =  row['date']
                current_trade['stop_loss_price'] = stop_loss_price
                current_trade['entry_price'] = entry_price
                current_trade['entry_time'] = entry_time
                current_trade['type'] = 'LONG'
                current_trade["ticker"]  = ticker
                current_trade['previous_day_close'] = previous_day_close
                current_trade['volume'] = row['volume']
                current_trade["RVOL"]  = row['RVOL_daily']
                
                #print(f"Opened position at {entry_price} on: {entry_time}")
            
            
    return (df, trades) 

# take profit 2:1
def strategy_2x_tp(df, ticker ="", initial_equity=10000, slippage_pct=0.0005, previous_day_close=1000000):
    
    
    
    is_green = False
    is_9_above_200 = False
    is_volume_above_avg = False
    is_price_10_perc_above_prev_close = False
    all_conditions = False
    in_opened_position = False
    is_stop_loss_hit = False
    is_take_profit_hit = False
    slippage_pct = 0.02/100
    
    trades = []
    
    current_trade = {}
    
    
    for idx in range(len(df) - 1):
        row = df.iloc[idx]
       
        if in_opened_position == True:
            # ------ exit conditions ------
            # Here you can add your exit conditions, for example:
          
            if row['low'] <= stop_loss_price:
                #print(f"Stop loss hit on: {row['date']}")
                exit_price = df['open'].iloc[idx + 1] * (1 - slippage_pct)  # exit at next candle open price with slippage
                in_opened_position = False
                all_conditions = False
                is_take_profit_hit = False
                current_trade['exit_price'] = exit_price
                current_trade['is_profit'] = False
                current_trade['exit_time'] = row['date']
                current_trade["ticker"]  = ticker
                trades.append(current_trade)
                current_trade = {}
                continue
                
            elif row['close'] >= take_profit_price:
                #print(f"Take profit hit on: {row['date']}")
                exit_price = df['open'].iloc[idx + 1] * (1 - slippage_pct)
                in_opened_position = False
                all_conditions = False
                is_take_profit_hit = True
                current_trade['exit_price'] = exit_price
                current_trade['is_profit'] = True
                current_trade['exit_time'] = row['date']
                current_trade["ticker"]  = ticker
                trades.append(current_trade)
                current_trade = {}
                continue
                
            elif  pd.to_datetime(row['date']).time() == time(16,0):
                #print(f"force to close at market hours close on: {row['date']}")
                exit_price = df['open'].iloc[idx + 1] * (1 - slippage_pct)
                in_opened_position = False
                all_conditions = False
                is_stop_loss_hit = False if exit_price < entry_price else True
                is_take_profit_hit = False if is_stop_loss_hit  == True else True
                current_trade['exit_price'] = exit_price
                current_trade['is_profit'] = is_take_profit_hit
                current_trade['exit_time'] = row['date']
                current_trade["ticker"]  = ticker
                trades.append(current_trade)
                current_trade = {}
                
        elif all_conditions == False and in_opened_position == False and pd.to_datetime(row['date']).time() >= time(7,0) and pd.to_datetime(row['date']).time() <= time(20,0):
            
            # ------ entrie conditions ------
            is_green = row['close'] > row['open']
            is_price_above_9 =  row['close'] > row['SMA_9']
            is_9_above_200 = row['SMA_9'] > row['SMA_200']
            is_volume_above_avg = row['volume'] > row['Vol_SMA_50'] and row['volume'] > 5000
            is_price_10_perc_above_prev_close = row['close'] > previous_day_close + (previous_day_close * 0.10)
            all_conditions = is_green and is_price_above_9 and is_9_above_200 and is_volume_above_avg and is_price_10_perc_above_prev_close and row['can_trade']
            
            # entry in the next candle after conditions met to avoid lookahead bias (that why we use row['open'] instead of row['close'])
            if all_conditions == True:
                stop_loss_price = row['low'] * (1 - slippage_pct) 
                
                entry_price = df['open'].iloc[idx + 1] * (1 - slippage_pct) # 
                #print(f'--- entry: {entry_price}, stopL: {stop_loss_price}')
                take_profit_price = (entry_price - stop_loss_price)*2 + entry_price # Risk-Reward 2:1
                in_opened_position = True
                entry_time =  row['date']
                current_trade['stop_loss_price'] = stop_loss_price
                current_trade['entry_price'] = entry_price
                current_trade['entry_time'] = entry_time
                current_trade['type'] = 'LONG'
                current_trade["ticker"]  = ticker
                current_trade['previous_day_close'] = previous_day_close
                current_trade['volume'] = row['volume']
                current_trade["RVOL"]  = row['RVOL_daily']
                
                #print(f"Opened position at {entry_price} on: {entry_time}")
            
            
    return (df, trades) 

# take profit 2:1
def strategy_2x_sma9_up(df, ticker ="", initial_equity=10000, slippage_pct=0.0005, previous_day_close=1000000):
    
    
    
    is_green = False
    is_9_above_200 = False
    is_volume_above_avg = False
    is_price_10_perc_above_prev_close = False
    all_conditions = False
    in_opened_position = False
    is_stop_loss_hit = False
    is_take_profit_hit = False
    slippage_pct = 0.02/100
    
    df["sma9_up_4"] = df["SMA_9"].diff().rolling(4).min() > 0
    
    trades = []
    
    current_trade = {}
    
    
    for idx in range(len(df) - 1):
        row = df.iloc[idx]
       
        if in_opened_position == True:
            # ------ exit conditions ------
            # Here you can add your exit conditions, for example:
          
            if row['low'] <= stop_loss_price:
                #print(f"Stop loss hit on: {row['date']}")
                exit_price = df['open'].iloc[idx + 1] * (1 - slippage_pct)  # exit at next candle open price with slippage
                in_opened_position = False
                all_conditions = False
                is_take_profit_hit = False
                current_trade['exit_price'] = exit_price
                current_trade['is_profit'] = False
                current_trade['exit_time'] = row['date']
                current_trade["ticker"]  = ticker
                trades.append(current_trade)
                current_trade = {}
                continue
                
            elif row['close'] >= take_profit_price:
                #print(f"Take profit hit on: {row['date']}")
                exit_price = df['open'].iloc[idx + 1] * (1 - slippage_pct)
                in_opened_position = False
                all_conditions = False
                is_take_profit_hit = True
                current_trade['exit_price'] = exit_price
                current_trade['is_profit'] = True
                current_trade['exit_time'] = row['date']
                current_trade["ticker"]  = ticker
                trades.append(current_trade)
                current_trade = {}
                continue
                
            elif  pd.to_datetime(row['date']).time() == time(16,0):
                #print(f"force to close at market hours close on: {row['date']}")
                exit_price = df['open'].iloc[idx + 1] * (1 - slippage_pct)
                in_opened_position = False
                all_conditions = False
                is_stop_loss_hit = False if exit_price < entry_price else True
                is_take_profit_hit = False if is_stop_loss_hit  == True else True
                current_trade['exit_price'] = exit_price
                current_trade['is_profit'] = is_take_profit_hit
                current_trade['exit_time'] = row['date']
                current_trade["ticker"]  = ticker
                trades.append(current_trade)
                current_trade = {}
                
        elif all_conditions == False and in_opened_position == False and pd.to_datetime(row['date']).time() >= time(7,0) and pd.to_datetime(row['date']).time() <= time(20,0):
            
            # ------ entrie conditions ------
            is_green = row['close'] > row['open']
            is_price_above_9 =  row['close'] > row['SMA_9']
            is_9_above_200 = row['SMA_9'] > row['SMA_200']
            is_volume_above_avg = row['volume'] > row['Vol_SMA_50'] and row['volume'] > 5000
            is_price_10_perc_above_prev_close = row['close'] > previous_day_close + (previous_day_close * 0.10)
            is_sma9_trending_up = row['sma9_up_4']
            all_conditions = (
                is_green
                and is_price_above_9
                and is_9_above_200
                and is_sma9_trending_up
                and is_volume_above_avg
                and is_price_10_perc_above_prev_close
                and row['can_trade']
            )
                    
            
            # entry in the next candle after conditions met to avoid lookahead bias (that why we use row['open'] instead of row['close'])
            if all_conditions == True:
                stop_loss_price = row['low'] * (1 - slippage_pct) 
                
                entry_price = df['open'].iloc[idx + 1] * (1 - slippage_pct) # 
                #print(f'--- entry: {entry_price}, stopL: {stop_loss_price}')
                take_profit_price = (entry_price - stop_loss_price)*2 + entry_price # Risk-Reward 2:1
                in_opened_position = True
                entry_time =  row['date']
                current_trade['stop_loss_price'] = stop_loss_price
                current_trade['entry_price'] = entry_price
                current_trade['entry_time'] = entry_time
                current_trade['type'] = 'LONG'
                current_trade["ticker"]  = ticker
                current_trade['previous_day_close'] = previous_day_close
                current_trade['volume'] = row['volume']
                current_trade["RVOL"]  = row['RVOL_daily']
                
                #print(f"Opened position at {entry_price} on: {entry_time}")
            
            
    return (df, trades) 

# take profit 3:1
def strategy_3x_tp(df, ticker ="", initial_equity=10000, slippage_pct=0.0005, previous_day_close=1000000):
    
    
    
    is_green = False
    is_9_above_200 = False
    is_volume_above_avg = False
    is_price_10_perc_above_prev_close = False
    all_conditions = False
    in_opened_position = False
    is_stop_loss_hit = False
    is_take_profit_hit = False
    slippage_pct = 0.02/100
    
    trades = []
    
    current_trade = {}
    
    
    for idx in range(len(df) - 1):
        row = df.iloc[idx]
       
        if in_opened_position == True:
            # ------ exit conditions ------
            # Here you can add your exit conditions, for example:
          
            if row['low'] <= stop_loss_price:
                #print(f"Stop loss hit on: {row['date']}")
                exit_price = df['open'].iloc[idx + 1] * (1 - slippage_pct)  # exit at next candle open price with slippage
                in_opened_position = False
                all_conditions = False
                is_take_profit_hit = False
                current_trade['exit_price'] = exit_price
                current_trade['is_profit'] = False
                current_trade['exit_time'] = row['date']
                current_trade["ticker"]  = ticker
                trades.append(current_trade)
                current_trade = {}
                continue
                
            elif row['close'] >= take_profit_price:
                #print(f"Take profit hit on: {row['date']}")
                exit_price = df['open'].iloc[idx + 1] * (1 - slippage_pct)
                in_opened_position = False
                all_conditions = False
                is_take_profit_hit = True
                current_trade['exit_price'] = exit_price
                current_trade['is_profit'] = True
                current_trade['exit_time'] = row['date']
                current_trade["ticker"]  = ticker
                trades.append(current_trade)
                current_trade = {}
                continue
                
            elif  pd.to_datetime(row['date']).time() == time(16,0):
                #print(f"force to close at market hours close on: {row['date']}")
                exit_price = df['open'].iloc[idx + 1] * (1 - slippage_pct)
                in_opened_position = False
                all_conditions = False
                is_stop_loss_hit = False if exit_price < entry_price else True
                is_take_profit_hit = False if is_stop_loss_hit  == True else True
                current_trade['exit_price'] = exit_price
                current_trade['is_profit'] = is_take_profit_hit
                current_trade['exit_time'] = row['date']
                current_trade["ticker"]  = ticker
                trades.append(current_trade)
                current_trade = {}
                
        elif all_conditions == False and in_opened_position == False and pd.to_datetime(row['date']).time() >= time(7,0) and pd.to_datetime(row['date']).time() <= time(20,0):
            
            # ------ entrie conditions ------
            is_green = row['close'] > row['open']
            is_price_above_9 =  row['close'] > row['SMA_9']
            is_9_above_200 = row['SMA_9'] > row['SMA_200']
            is_volume_above_avg = row['volume'] > row['Vol_SMA_50'] and row['volume'] > 5000
            is_price_10_perc_above_prev_close = row['close'] > previous_day_close + (previous_day_close * 0.10)
            all_conditions = is_green and is_price_above_9 and is_9_above_200 and is_volume_above_avg and is_price_10_perc_above_prev_close and row['can_trade']
            
            # entry in the next candle after conditions met to avoid lookahead bias (that why we use row['open'] instead of row['close'])
            if all_conditions == True:
                stop_loss_price = row['low'] * (1 - slippage_pct) 
                
                entry_price = df['open'].iloc[idx + 1] * (1 - slippage_pct) # 
                #print(f'--- entry: {entry_price}, stopL: {stop_loss_price}')
                take_profit_price = (entry_price - stop_loss_price)*3 + entry_price # Risk-Reward 3:1
                in_opened_position = True
                entry_time =  row['date']
                current_trade['stop_loss_price'] = stop_loss_price
                current_trade['entry_price'] = entry_price
                current_trade['entry_time'] = entry_time
                current_trade['type'] = 'LONG'
                current_trade["ticker"]  = ticker
                current_trade["RVOL"]  = row['RVOL_daily']
                current_trade['previous_day_close'] = previous_day_close
                #print(f"Opened position at {entry_price} on: {entry_time}")
            
            
    return (df, trades) 

# take profit n:1
def strategy_nx_tp(df, ticker ="", initial_equity=10000, slippage_pct=0.0005, previous_day_close=1000000, factor_tp=3):
    
    
    
    is_green = False
    is_9_above_200 = False
    is_volume_above_avg = False
    is_price_10_perc_above_prev_close = False
    all_conditions = False
    in_opened_position = False
    is_stop_loss_hit = False
    is_take_profit_hit = False
    slippage_pct = 0.02/100
    
    trades = []
    
    current_trade = {}
    
    
    for idx in range(len(df) - 1):
        row = df.iloc[idx]
       
        if in_opened_position == True:
            # ------ exit conditions ------
            # Here you can add your exit conditions, for example:
          
            if row['low'] <= stop_loss_price:
                #print(f"Stop loss hit on: {row['date']}")
                exit_price = df['open'].iloc[idx + 1] * (1 - slippage_pct)  # exit at next candle open price with slippage
                in_opened_position = False
                all_conditions = False
                is_take_profit_hit = False
                current_trade['exit_price'] = exit_price
                current_trade['is_profit'] = False
                current_trade['exit_time'] = row['date']
                current_trade["ticker"]  = ticker
                trades.append(current_trade)
                current_trade = {}
                continue
                
            elif row['close'] >= take_profit_price:
                #print(f"Take profit hit on: {row['date']}")
                exit_price = df['open'].iloc[idx + 1] * (1 - slippage_pct)
                in_opened_position = False
                all_conditions = False
                is_take_profit_hit = True
                current_trade['exit_price'] = exit_price
                current_trade['is_profit'] = True
                current_trade['exit_time'] = row['date']
                current_trade["ticker"]  = ticker
                trades.append(current_trade)
                current_trade = {}
                continue
                
            elif  pd.to_datetime(row['date']).time() == time(16,0):
                #print(f"force to close at market hours close on: {row['date']}")
                exit_price = df['open'].iloc[idx + 1] * (1 - slippage_pct)
                in_opened_position = False
                all_conditions = False
                is_stop_loss_hit = False if exit_price < entry_price else True
                is_take_profit_hit = False if is_stop_loss_hit  == True else True
                current_trade['exit_price'] = exit_price
                current_trade['is_profit'] = is_take_profit_hit
                current_trade['exit_time'] = row['date']
                current_trade["ticker"]  = ticker
                trades.append(current_trade)
                current_trade = {}
                
        elif all_conditions == False and in_opened_position == False and pd.to_datetime(row['date']).time() >= time(7,0) and pd.to_datetime(row['date']).time() <= time(20,0):
            
            # ------ entrie conditions ------
            is_green = row['close'] > row['open']
            is_price_above_9 =  row['close'] > row['SMA_9']
            is_9_above_200 = row['SMA_9'] > row['SMA_200']
            is_volume_above_avg = row['volume'] > row['Vol_SMA_50'] and row['volume'] > 5000
            is_price_10_perc_above_prev_close = row['close'] > previous_day_close + (previous_day_close * 0.10)
            all_conditions = is_green and is_price_above_9 and is_9_above_200 and is_volume_above_avg and is_price_10_perc_above_prev_close and row['can_trade']
            
            # entry in the next candle after conditions met to avoid lookahead bias (that why we use row['open'] instead of row['close'])
            if all_conditions == True:
                stop_loss_price = row['low'] * (1 - slippage_pct) 
                
                entry_price = df['open'].iloc[idx + 1] * (1 - slippage_pct) # 
                #print(f'--- entry: {entry_price}, stopL: {stop_loss_price}')
                take_profit_price = (entry_price - stop_loss_price)*factor_tp + entry_price # Risk-Reward 3:1
                in_opened_position = True
                entry_time =  row['date']
                current_trade['stop_loss_price'] = stop_loss_price
                current_trade['entry_price'] = entry_price
                current_trade['entry_time'] = entry_time
                current_trade['type'] = 'LONG'
                current_trade["ticker"]  = ticker
                current_trade["RVOL"]  = row['RVOL_daily']
                current_trade['previous_day_close'] = previous_day_close
                #print(f"Opened position at {entry_price} on: {entry_time}")
            
            
    return (df, trades) 

# take profit dynamic base on volatiliy
def strategy_dynamic_tp_stp(
    df,
    ticker ="",
    initial_equity=10000, 
    previous_day_close=1000000,
    slippage_entry=0.002,   # 0.2%
    slippage_exit=0.003     # 0.3%
):
    trades = []
    in_position = False
    current_trade = {}

    for i in range(len(df) - 1):
        row = df.iloc[i]
        next_row = df.iloc[i + 1]

        current_time = pd.to_datetime(row['date']).time()

        # =============================
        # EXIT LOGIC
        # =============================
        if in_position:

            # Stop Loss (intra-bar)
            if row['low'] <= current_trade['stop_loss_price']:
                exit_price = current_trade['stop_loss_price'] * (1 - slippage_exit)

                current_trade.update({
                    'exit_price': exit_price,
                    'exit_time': row['date'],
                    'is_profit': False,
                    "ticker":ticker
                })

                trades.append(current_trade)
                current_trade = {}
                in_position = False
                continue

            # Take Profit
            if row['high'] >= current_trade['take_profit_price']:
                exit_price = current_trade['take_profit_price'] * (1 - slippage_exit)

                current_trade.update({
                    'exit_price': exit_price,
                    'exit_time': row['date'],
                    'is_profit': True,
                    "ticker":ticker
                })

                trades.append(current_trade)
                current_trade = {}
                in_position = False
                continue

            # Force close at end of day
            if current_time >= time(16, 0):
                exit_price = next_row['open'] * (1 - slippage_exit)
                is_profit = exit_price > current_trade['entry_price']

                current_trade.update({
                    'exit_price': exit_price,
                    'exit_time': row['date'],
                    'is_profit': is_profit,
                    "ticker":ticker
                })

                trades.append(current_trade)
                current_trade = {}
                in_position = False
                continue

        # =============================
        # ENTRY LOGIC
        # =============================
        if not in_position and time(7, 0) <= current_time <= time(20, 0):

            is_green = row['close'] > row['open']
            is_price_above_9 = row['close'] > row['SMA_9']
            is_9_above_200 = row['SMA_9'] > row['SMA_200']
            is_volume_above_avg = row['volume'] > row['Vol_SMA_50'] and row['volume'] > 5000
            is_price_10_perc_above_prev_close = row['close'] > previous_day_close * 1.10

            all_conditions = (
                is_green and
                is_price_above_9 and
                is_9_above_200 and
                is_volume_above_avg and
                is_price_10_perc_above_prev_close and
                row['can_trade']
            )

            if all_conditions:
                entry_price = next_row['open'] * (1 + slippage_entry)

                # ATR-based stop (más realista)
                stop_loss_price = entry_price - 1.5 * row['atr']

                # Risk / Reward 2:1
                take_profit_price = entry_price + 2 * (entry_price - stop_loss_price)

                current_trade = {
                    'entry_price': entry_price,
                    'stop_loss_price': stop_loss_price,
                    'take_profit_price': take_profit_price,
                    'entry_time': next_row['date'],
                    'type': 'LONG',
                    "ticker":ticker,
                    'previous_day_close':previous_day_close,
                    "volume":row['volume'],
                    "RVOL" : row['RVOL_daily']
                
                }
                

                in_position = True
    return (df, trades) 


def run_tests_strategy_2x_tp():
    start_time = tm.perf_counter()
    #previous_day_close = 3.17
    #(df_prep, indicators) = prepare_data_for_9sma_200sma_strategy("IRBT", "2025-12-10", initial_equity=10000, slippage_pct=0.0005, previous_day_close = previous_day_close)

    #previous_day_close = 3.22
    #(df_prep, indicators) = prepare_data_for_9sma_200sma_strategy("BTTC", "2025-12-11", initial_equity=10000, slippage_pct=0.0005, previous_day_close = previous_day_close)

    #previous_day_close = 0.61
    #(df_prep, indicators) = prepare_data_for_9sma_200sma_strategy("ZYXI", "2025-11-24", initial_equity=10000, slippage_pct=0.0005, previous_day_close = previous_day_close)

    #previous_day_close = 3.49
    #(df_prep, indicators) = prepare_data_for_9sma_200sma_strategy("HTOO", "2025-12-03", initial_equity=10000, slippage_pct=0.0005, previous_day_close = previous_day_close)

    #previous_day_close = 5.07
    #(df_prep, indicators) = prepare_data_for_9sma_200sma_strategy("HTOO", "2025-10-16", initial_equity=10000, slippage_pct=0.0005, previous_day_close = previous_day_close)

    #previous_day_close = 1.09
    #(df_prep, indicators) = prepare_data_for_9sma_200sma_strategy("CETY", "2025-11-25", initial_equity=10000, slippage_pct=0.0005, previous_day_close = previous_day_close)

    #previous_day_close = 5.9
    #(df_prep, indicators) = prepare_data_for_9sma_200sma_strategy("SMX", "2025-11-26", initial_equity=10000, slippage_pct=0.0005, previous_day_close = previous_day_close)

    #previous_day_close = 17
    #(df_prep, indicators) = prepare_data_for_9sma_200sma_strategy("SMX", "2025-11-28", initial_equity=10000, slippage_pct=0.0005, previous_day_close = previous_day_close)


    previous_day_close = 0.53
    (df_prep, indicators) = prepare_data_for_9sma_200sma_strategy("MIGI", "2023-02-09", initial_equity=10000, slippage_pct=0.0005, previous_day_close = previous_day_close)


    #(df, trades) = strategy_dynamic_tp_stp(df_prep, initial_equity=10000,  previous_day_close=previous_day_close)
    (df, trades) = strategy_2x_tp(df_prep, ticker="MIGI", initial_equity=10000,  previous_day_close=previous_day_close)


    #print(pd.DataFrame(trades))

    end_time = tm.perf_counter()
        
    # 6. CALCULAR Y MOSTRAR LA DURACIÓN
    total_duration = end_time - start_time

    print(f"⏰ Tiempo total de ejecución: {total_duration:.2f} segundos.")
    print("==============================================")

    chart = Chart()
    chart.legend(visible=True)
    chart.set(df)
    
    print(df)

    markers = utils_helpers.generate_markers_from_trade_list(trades)


            
    line_9_1min = chart.create_line('SMA_9_1min')
    #sma_9 = utils_helpers.calculate_sma(df, period=9)
    sma_9_1min = indicators['sma_9_1min']
    line_9_1min.set(sma_9_1min)

    # line_9_5min = chart.create_line('SMA_9_5min', color='green')
    # sma_9_5min = indicators['sma_9_5min']
    # line_9_5min.set(sma_9_5min)


    line200 = chart.create_line(name='SMA 200',color='blue')
    sma_200 = indicators['sma_200']
    line200.set(sma_200)

    chart.marker_list(markers)


    chart.show(block=True)
    
    return 


@runner.pipeline
def pipeline_strategy_dynamic_tp_stp(df = pd.DataFrame([]), id=0):
    
    print(f'**************  pipeline pipeline_strategy_dynamic_tp_stp {id} ******************')
    utils_helpers.log(f' ***** started pipeline {id}', file_path=f'./pipeline_logs/strategy_dynamic_tp_stp_log_{id}.csv')
    start_time = tm.perf_counter()
    all_trades = []
    
    date_str = None
    ticker = None
    #print(df)
    
    try:
        
        
        for i in range(len(df)):
            row = df.iloc[i]
            previous_day_close = row['previous_close']
            date_str = row['date_str']
            ticker = row['ticker']
            (df_prep, indicators) = prepare_data_for_9sma_200sma_strategy(ticker, date_str, initial_equity=10000, slippage_pct=0.0005, previous_day_close = previous_day_close)
            (df_res, trades) = strategy_dynamic_tp_stp(df_prep, ticker= ticker, initial_equity=10000,  previous_day_close=previous_day_close)
            all_trades = all_trades + trades
           
            
        end_time = tm.perf_counter()
            
        # 6. CALCULAR Y MOSTRAR LA DURACIÓN
        total_duration = end_time - start_time
        tds = pd.DataFrame(all_trades)
        #tds.to_parquet(f'trades/pipeline_{id}.parquet')
        utils_helpers.append_single_parquet(df=tds, path=f'trades/strategy_dynamic_tp_stp_log__pipeline_{id}.parquet')
        print(f"⏰ Tiempo total de ejecución for pipeline {id}, with {len(df)} days: {total_duration:.2f} segundos. Total trades: {len(all_trades)}")
        utils_helpers.log(f"⏰ Tiempo total de ejecución for pipeline {id}: {total_duration:.2f} segundos. Total trades: {len(all_trades)}", f'./pipeline_logs/strategy_dynamic_tp_stp_log_{id}.csv')
        
        print("==============================================")
       
        
    except Exception as e:
                print("pipeline:", e)   
                end_time = tm.perf_counter()
                utils_helpers.log(f'process fail for ticker {ticker} at {date_str}, error: {e}', f'./pipeline_logs/strategy_dynamic_tp_stp_log_{id}.csv')
        
                # 6. CALCULAR Y MOSTRAR LA DURACIÓN
                total_duration = end_time - start_time

                print(f"⏰ Tiempo total de ejecución: {total_duration:.2f} segundos.")
                print("=================error=============================")
    
    
    
    return

@runner.pipeline
def pipeline_strategy_1x_tp(df = pd.DataFrame([]), id=0):
    
    print(f'************** strategy_1x  pipeline {id} ******************')
    utils_helpers.log(f' ***** started strategy_1x pipeline {id}', file_path=f'./pipeline_logs/strategy_1x_tp_log_{id}.csv')
    start_time = tm.perf_counter()
    all_trades = []
    
    date_str = None
    ticker = None
    
    try:
        
    
        for i in range(len(df)):
            row = df.iloc[i]
            previous_day_close = row['previous_close']
            date_str = row['date_str']
            ticker = row['ticker']
            (df_prep, indicators) = prepare_data_for_9sma_200sma_strategy(ticker, date_str, initial_equity=10000, slippage_pct=0.0005, previous_day_close = previous_day_close)
            (df_res, trades) = strategy_1x_tp(df_prep, ticker= ticker, initial_equity=10000,  previous_day_close=previous_day_close)
            all_trades = all_trades + trades
           
            
        end_time = tm.perf_counter()
            
        # 6. CALCULAR Y MOSTRAR LA DURACIÓN
        total_duration = end_time - start_time
        tds = pd.DataFrame(all_trades)
        #tds.to_parquet(f'trades/pipeline_{id}.parquet')
        utils_helpers.append_single_parquet(df=tds, path=f'trades/strategy_1x_tp_pipeline_{id}.parquet')
        print(f"⏰ Tiempo total de ejecución for pipeline {id}, with {len(df)} days: {total_duration:.2f} segundos. Total trades: {len(all_trades)}")
        utils_helpers.log(f"⏰ Tiempo total de ejecución for pipeline {id}: {total_duration:.2f} segundos. Total trades: {len(all_trades)}", f'./pipeline_logs/strategy_1x_tp_log_{id}.csv')
        
        print("==============================================")
       
        
    except Exception as e:
                print("pipeline:", e)   
                end_time = tm.perf_counter()
                utils_helpers.log(f'process fail for ticker {ticker} at {date_str}, error: {e}', f'./pipeline_logs/strategy_1x_tp_log_{id}.csv')
        
                # 6. CALCULAR Y MOSTRAR LA DURACIÓN
                total_duration = end_time - start_time

                print(f"⏰ Tiempo total de ejecución: {total_duration:.2f} segundos.")
                print("=================error=============================")
    
    
    
    return

@runner.pipeline
def pipeline_strategy_1x_tp_out_of_sample(df = pd.DataFrame([]), id=0):
    
    print(f'************** strategy_1x  pipeline {id} ******************')
    utils_helpers.log(f' ***** started strategy_1x pipeline {id}', file_path=f'./pipeline_logs/strategy_1x_tp_out_of_sample_log_{id}.csv')
    start_time = tm.perf_counter()
    all_trades = []
    
    date_str = None
    ticker = None
    
    try:
        
    
        for i in range(len(df)):
            row = df.iloc[i]
            previous_day_close = row['previous_close']
            date_str = row['date_str']
            ticker = row['ticker']
            (df_prep, indicators) = prepare_data_for_9sma_200sma_strategy(ticker, date_str, initial_equity=10000, slippage_pct=0.0005, previous_day_close = previous_day_close)
            (df_res, trades) = strategy_1x_tp(df_prep, ticker= ticker, initial_equity=10000,  previous_day_close=previous_day_close)
            all_trades = all_trades + trades
           
            
        end_time = tm.perf_counter()
            
        # 6. CALCULAR Y MOSTRAR LA DURACIÓN
        total_duration = end_time - start_time
        tds = pd.DataFrame(all_trades)
        #tds.to_parquet(f'trades/pipeline_{id}.parquet')
        utils_helpers.append_single_parquet(df=tds, path=f'trades/out_of_sample/strategy_1x_tp_pipeline_{id}.parquet')
        print(f"⏰ Tiempo total de ejecución for pipeline {id}, with {len(df)} days: {total_duration:.2f} segundos. Total trades: {len(all_trades)}")
        utils_helpers.log(f"⏰ Tiempo total de ejecución for pipeline {id}: {total_duration:.2f} segundos. Total trades: {len(all_trades)}", f'./pipeline_logs/strategy_1x_tp_out_of_sample_log_{id}.csv')
        
        print("==============================================")
       
        
    except Exception as e:
                print("pipeline:", e)   
                end_time = tm.perf_counter()
                utils_helpers.log(f'process fail for ticker {ticker} at {date_str}, error: {e}', f'./pipeline_logs/strategy_1x_tp_out_of_sample_log_{id}.csv')
        
                # 6. CALCULAR Y MOSTRAR LA DURACIÓN
                total_duration = end_time - start_time

                print(f"⏰ Tiempo total de ejecución: {total_duration:.2f} segundos.")
                print("=================error=============================")
    
    
    
    return

@runner.pipeline
def pipeline_strategy_2x_tp(df = pd.DataFrame([]), id=0):
    
    print(f'************** strategy_2x  pipeline {id} ******************')
    utils_helpers.log(f' ***** started strategy_1x pipeline {id}', file_path=f'./pipeline_logs/strategy_2x_tp_log_{id}.csv')
    start_time = tm.perf_counter()
    all_trades = []
    
    date_str = None
    ticker = None
    
    try:
        
    
        for i in range(len(df)):
            row = df.iloc[i]
            previous_day_close = row['previous_close']
            date_str = row['date_str']
            ticker = row['ticker']
            (df_prep, indicators) = prepare_data_for_9sma_200sma_strategy(ticker, date_str, initial_equity=10000, slippage_pct=0.0005, previous_day_close = previous_day_close)
            (df_res, trades) = strategy_2x_tp(df_prep, ticker= ticker, initial_equity=10000,  previous_day_close=previous_day_close)
            all_trades = all_trades + trades
           
            
        end_time = tm.perf_counter()
            
        # 6. CALCULAR Y MOSTRAR LA DURACIÓN
        total_duration = end_time - start_time
        tds = pd.DataFrame(all_trades)
        #tds.to_parquet(f'trades/pipeline_{id}.parquet')
        utils_helpers.append_single_parquet(df=tds, path=f'trades/strategy_2x_tp_pipeline_{id}.parquet')
        print(f"⏰ Tiempo total de ejecución for pipeline {id}, with {len(df)} days: {total_duration:.2f} segundos. Total trades: {len(all_trades)}")
        utils_helpers.log(f"⏰ Tiempo total de ejecución for pipeline {id}: {total_duration:.2f} segundos. Total trades: {len(all_trades)}", f'./pipeline_logs/strategy_2x_tp_log_{id}.csv')
        
        print("==============================================")
       
        
    except Exception as e:
                print("pipeline:", e)   
                end_time = tm.perf_counter()
                utils_helpers.log(f'process fail for ticker {ticker} at {date_str}, error: {e}', f'./pipeline_logs/strategy_2x_tp_log_{id}.csv')
        
                # 6. CALCULAR Y MOSTRAR LA DURACIÓN
                total_duration = end_time - start_time

                print(f"⏰ Tiempo total de ejecución: {total_duration:.2f} segundos.")
                print("=================error=============================")
    
    
    
    return

@runner.pipeline
def pipeline_strategy_2x_tp_out_of_sample(df = pd.DataFrame([]), id=0):
    
    print(f'************** strategy_2x  pipeline {id} ******************')
    utils_helpers.log(f' ***** started strategy_1x pipeline {id}', file_path=f'./pipeline_logs/strategy_2x_tp_out_of_sample_log_{id}.csv')
    start_time = tm.perf_counter()
    all_trades = []
    
    date_str = None
    ticker = None
    
    try:
        
    
        for i in range(len(df)):
            row = df.iloc[i]
            previous_day_close = row['previous_close']
            date_str = row['date_str']
            ticker = row['ticker']
            (df_prep, indicators) = prepare_data_for_9sma_200sma_strategy(ticker, date_str, initial_equity=10000, slippage_pct=0.0005, previous_day_close = previous_day_close)
            (df_res, trades) = strategy_2x_tp(df_prep, ticker= ticker, initial_equity=10000,  previous_day_close=previous_day_close)
            all_trades = all_trades + trades
           
            
        end_time = tm.perf_counter()
            
        # 6. CALCULAR Y MOSTRAR LA DURACIÓN
        total_duration = end_time - start_time
        tds = pd.DataFrame(all_trades)
        #tds.to_parquet(f'trades/pipeline_{id}.parquet')
        utils_helpers.append_single_parquet(df=tds, path=f'trades/out_of_sample/strategy_2x_tp_pipeline_{id}.parquet')
        print(f"⏰ Tiempo total de ejecución for pipeline {id}, with {len(df)} days: {total_duration:.2f} segundos. Total trades: {len(all_trades)}")
        utils_helpers.log(f"⏰ Tiempo total de ejecución for pipeline {id}: {total_duration:.2f} segundos. Total trades: {len(all_trades)}", f'./pipeline_logs/strategy_2x_tp_out_of_sample_log_{id}.csv')
        
        print("==============================================")
       
        
    except Exception as e:
                print("pipeline:", e)   
                end_time = tm.perf_counter()
                utils_helpers.log(f'process fail for ticker {ticker} at {date_str}, error: {e}', f'./pipeline_logs/strategy_2x_tp_out_of_sample_log_{id}.csv')
        
                # 6. CALCULAR Y MOSTRAR LA DURACIÓN
                total_duration = end_time - start_time

                print(f"⏰ Tiempo total de ejecución: {total_duration:.2f} segundos.")
                print("=================error=============================")
    
    
    
    return

@runner.pipeline
def pipeline_strategy_2x_sma9_up(df = pd.DataFrame([]), id=0):
    
    print(f'************** strategy_2x_sma9_up  pipeline {id} ******************')
    utils_helpers.log(f' ***** started strategy_1x pipeline {id}', file_path=f'./pipeline_logs/strategy_2x_sma9_up_log_{id}.csv')
    start_time = tm.perf_counter()
    all_trades = []
    
    date_str = None
    ticker = None
    
    try:
        
    
        for i in range(len(df)):
            row = df.iloc[i]
            previous_day_close = row['previous_close']
            date_str = row['date_str']
            ticker = row['ticker']
            (df_prep, indicators) = prepare_data_for_9sma_200sma_strategy(ticker, date_str, initial_equity=10000, slippage_pct=0.0005, previous_day_close = previous_day_close)
            (df_res, trades) = strategy_2x_sma9_up(df_prep, ticker= ticker, initial_equity=10000,  previous_day_close=previous_day_close)
            all_trades = all_trades + trades
           
            
        end_time = tm.perf_counter()
            
        # 6. CALCULAR Y MOSTRAR LA DURACIÓN
        total_duration = end_time - start_time
        tds = pd.DataFrame(all_trades)
        #tds.to_parquet(f'trades/pipeline_{id}.parquet')
        utils_helpers.append_single_parquet(df=tds, path=f'trades/strategy_2x_sma9_up_pipeline_{id}.parquet')
        print(f"⏰ Tiempo total de ejecución for pipeline {id}, with {len(df)} days: {total_duration:.2f} segundos. Total trades: {len(all_trades)}")
        utils_helpers.log(f"⏰ Tiempo total de ejecución for pipeline {id}: {total_duration:.2f} segundos. Total trades: {len(all_trades)}", f'./pipeline_logs/strategy_2x_sma9_up_log_{id}.csv')
        
        print("==============================================")
       
        
    except Exception as e:
                print("pipeline:", e)   
                end_time = tm.perf_counter()
                utils_helpers.log(f'process fail for ticker {ticker} at {date_str}, error: {e}', f'./pipeline_logs/strategy_2x_sma9_up_log_{id}.csv')
        
                # 6. CALCULAR Y MOSTRAR LA DURACIÓN
                total_duration = end_time - start_time

                print(f"⏰ Tiempo total de ejecución: {total_duration:.2f} segundos.")
                print("=================error=============================")
    
    
    
    return

@runner.pipeline
def pipeline_strategy_2x_sma9_up_out_of_sample(df = pd.DataFrame([]), id=0):
    
    print(f'************** strategy_2x_sma9_up  pipeline {id} ******************')
    utils_helpers.log(f' ***** started strategy_1x pipeline {id}', file_path=f'./pipeline_logs/strategy_2x_sma9_up_out_of_sample_log_{id}.csv')
    start_time = tm.perf_counter()
    all_trades = []
    
    date_str = None
    ticker = None
    
    try:
        
    
        for i in range(len(df)):
            row = df.iloc[i]
            previous_day_close = row['previous_close']
            date_str = row['date_str']
            ticker = row['ticker']
            (df_prep, indicators) = prepare_data_for_9sma_200sma_strategy(ticker, date_str, initial_equity=10000, slippage_pct=0.0005, previous_day_close = previous_day_close)
            (df_res, trades) = strategy_2x_sma9_up(df_prep, ticker= ticker, initial_equity=10000,  previous_day_close=previous_day_close)
            all_trades = all_trades + trades
           
            
        end_time = tm.perf_counter()
            
        # 6. CALCULAR Y MOSTRAR LA DURACIÓN
        total_duration = end_time - start_time
        tds = pd.DataFrame(all_trades)
        #tds.to_parquet(f'trades/pipeline_{id}.parquet')
        utils_helpers.append_single_parquet(df=tds, path=f'trades/out_of_sample/strategy_2x_sma9_up_pipeline_{id}.parquet')
        print(f"⏰ Tiempo total de ejecución for pipeline {id}, with {len(df)} days: {total_duration:.2f} segundos. Total trades: {len(all_trades)}")
        utils_helpers.log(f"⏰ Tiempo total de ejecución for pipeline {id}: {total_duration:.2f} segundos. Total trades: {len(all_trades)}", f'./pipeline_logs/strategy_2x_sma9_up_out_of_sample_log_{id}.csv')
        
        print("==============================================")
       
        
    except Exception as e:
                print("pipeline:", e)   
                end_time = tm.perf_counter()
                utils_helpers.log(f'process fail for ticker {ticker} at {date_str}, error: {e}', f'./pipeline_logs/strategy_2x_sma9_up_out_of_sample_log_{id}.csv')
        
                # 6. CALCULAR Y MOSTRAR LA DURACIÓN
                total_duration = end_time - start_time

                print(f"⏰ Tiempo total de ejecución: {total_duration:.2f} segundos.")
                print("=================error=============================")
    
    
    
    return

@runner.pipeline
def pipeline_strategy_3x_tp(df = pd.DataFrame([]), id=0):
    
    print(f'************** strategy_3x  pipeline {id} ******************')
    utils_helpers.log(f' ***** started strategy_3x pipeline {id}', file_path=f'./pipeline_logs/strategy_3x_tp_log_{id}.csv')
    start_time = tm.perf_counter()
    all_trades = []
    
    date_str = None
    ticker = None
    
    try:
        
    
        for i in range(len(df)):
            row = df.iloc[i]
            previous_day_close = row['previous_close']
            date_str = row['date_str']
            ticker = row['ticker']
            (df_prep, indicators) = prepare_data_for_9sma_200sma_strategy(ticker, date_str, initial_equity=10000, slippage_pct=0.0005, previous_day_close = previous_day_close)
            (df_res, trades) = strategy_1x_tp(df_prep, ticker= ticker, initial_equity=10000,  previous_day_close=previous_day_close)
            all_trades = all_trades + trades
           
            
        end_time = tm.perf_counter()
            
        # 6. CALCULAR Y MOSTRAR LA DURACIÓN
        total_duration = end_time - start_time
        tds = pd.DataFrame(all_trades)
        #tds.to_parquet(f'trades/pipeline_{id}.parquet')
        utils_helpers.append_single_parquet(df=tds, path=f'trades/strategy_3x_tp_pipeline_{id}.parquet')
        print(f"⏰ Tiempo total de ejecución for pipeline {id}, with {len(df)} days: {total_duration:.2f} segundos. Total trades: {len(all_trades)}")
        utils_helpers.log(f"⏰ Tiempo total de ejecución for pipeline {id}: {total_duration:.2f} segundos. Total trades: {len(all_trades)}", f'./pipeline_logs/strategy_3x_tp_log_{id}.csv')
        
        print("==============================================")
       
        
    except Exception as e:
                print("pipeline:", e)   
                end_time = tm.perf_counter()
                utils_helpers.log(f'process fail for ticker {ticker} at {date_str}, error: {e}', f'./pipeline_logs/strategy_3x_tp_log_{id}.csv')
        
                # 6. CALCULAR Y MOSTRAR LA DURACIÓN
                total_duration = end_time - start_time

                print(f"⏰ Tiempo total de ejecución: {total_duration:.2f} segundos.")
                print("=================error=============================")
    
    
    
    return

@runner.pipeline
def pipeline_strategy_3x_tp_out_of_sample(df = pd.DataFrame([]), id=0):
    
    print(f'************** strategy_3x  pipeline {id} ******************')
    utils_helpers.log(f' ***** started strategy_3x pipeline {id}', file_path=f'./pipeline_logs/strategy_3x_tp_out_of_sample_log_{id}.csv')
    start_time = tm.perf_counter()
    all_trades = []
    
    date_str = None
    ticker = None
    
    try:
        
    
        for i in range(len(df)):
            row = df.iloc[i]
            previous_day_close = row['previous_close']
            date_str = row['date_str']
            ticker = row['ticker']
            (df_prep, indicators) = prepare_data_for_9sma_200sma_strategy(ticker, date_str, initial_equity=10000, slippage_pct=0.0005, previous_day_close = previous_day_close)
            (df_res, trades) = strategy_1x_tp(df_prep, ticker= ticker, initial_equity=10000,  previous_day_close=previous_day_close)
            all_trades = all_trades + trades
           
            
        end_time = tm.perf_counter()
            
        # 6. CALCULAR Y MOSTRAR LA DURACIÓN
        total_duration = end_time - start_time
        tds = pd.DataFrame(all_trades)
        #tds.to_parquet(f'trades/pipeline_{id}.parquet')
        utils_helpers.append_single_parquet(df=tds, path=f'trades/out_of_sample/strategy_3x_tp_pipeline_{id}.parquet')
        print(f"⏰ Tiempo total de ejecución for pipeline {id}, with {len(df)} days: {total_duration:.2f} segundos. Total trades: {len(all_trades)}")
        utils_helpers.log(f"⏰ Tiempo total de ejecución for pipeline {id}: {total_duration:.2f} segundos. Total trades: {len(all_trades)}", f'./pipeline_logs/strategy_3x_tp_out_of_sample_log_{id}.csv')
        
        print("==============================================")
       
        
    except Exception as e:
                print("pipeline:", e)   
                end_time = tm.perf_counter()
                utils_helpers.log(f'process fail for ticker {ticker} at {date_str}, error: {e}', f'./pipeline_logs/strategy_3x_tp_out_of_sample_log_{id}.csv')
        
                # 6. CALCULAR Y MOSTRAR LA DURACIÓN
                total_duration = end_time - start_time

                print(f"⏰ Tiempo total de ejecución: {total_duration:.2f} segundos.")
                print("=================error=============================")
    
    
    
    return

@runner.pipeline
def pipeline_strategy_4x_tp(df = pd.DataFrame([]), id=0):
    
    print(f'************** strategy_4x  pipeline {id} ******************')
    utils_helpers.log(f' ***** started strategy_4x pipeline {id}', file_path=f'./pipeline_logs/strategy_4x_tp_log_{id}.csv')
    start_time = tm.perf_counter()
    all_trades = []
    
    date_str = None
    ticker = None
    
    try:
        
    
        for i in range(len(df)):
            row = df.iloc[i]
            previous_day_close = row['previous_close']
            date_str = row['date_str']
            ticker = row['ticker']
            (df_prep, indicators) = prepare_data_for_9sma_200sma_strategy(ticker, date_str, initial_equity=10000, slippage_pct=0.0005, previous_day_close = previous_day_close)
            (df_res, trades) = strategy_nx_tp(df_prep, ticker= ticker, initial_equity=10000,  previous_day_close=previous_day_close, factor_tp=4)
            all_trades = all_trades + trades
           
            
        end_time = tm.perf_counter()
            
        # 6. CALCULAR Y MOSTRAR LA DURACIÓN
        total_duration = end_time - start_time
        tds = pd.DataFrame(all_trades)
        #tds.to_parquet(f'trades/pipeline_{id}.parquet')
        utils_helpers.append_single_parquet(df=tds, path=f'trades/strategy_4x_tp_pipeline_{id}.parquet')
        print(f"⏰ Tiempo total de ejecución for pipeline {id}, with {len(df)} days: {total_duration:.2f} segundos. Total trades: {len(all_trades)}")
        utils_helpers.log(f"⏰ Tiempo total de ejecución for pipeline {id}: {total_duration:.2f} segundos. Total trades: {len(all_trades)}", f'./pipeline_logs/strategy_4x_tp_log_{id}.csv')
        
        print("==============================================")
       
        
    except Exception as e:
                print("pipeline:", e)   
                end_time = tm.perf_counter()
                utils_helpers.log(f'process fail for ticker {ticker} at {date_str}, error: {e}', f'./pipeline_logs/strategy_4x_tp_log_{id}.csv')
        
                # 6. CALCULAR Y MOSTRAR LA DURACIÓN
                total_duration = end_time - start_time

                print(f"⏰ Tiempo total de ejecución: {total_duration:.2f} segundos.")
                print("=================error=============================")
    
    
    
    return

@runner.pipeline
def pipeline_strategy_4x_tp_out_of_sample(df = pd.DataFrame([]), id=0):
    
    print(f'************** strategy_4x out_of_sample  pipeline {id} ******************')
    utils_helpers.log(f' ***** started strategy_4x pipeline {id}', file_path=f'./pipeline_logs/strategy_4x_tp_out_of_sample_log_{id}.csv')
    start_time = tm.perf_counter()
    all_trades = []
    
    date_str = None
    ticker = None
    
    try:
        
    
        for i in range(len(df)):
            row = df.iloc[i]
            previous_day_close = row['previous_close']
            date_str = row['date_str']
            ticker = row['ticker']
            (df_prep, indicators) = prepare_data_for_9sma_200sma_strategy(ticker, date_str, initial_equity=10000, slippage_pct=0.0005, previous_day_close = previous_day_close)
            (df_res, trades) = strategy_nx_tp(df_prep, ticker= ticker, initial_equity=10000,  previous_day_close=previous_day_close, factor_tp=4)
            all_trades = all_trades + trades
           
            
        end_time = tm.perf_counter()
            
        # 6. CALCULAR Y MOSTRAR LA DURACIÓN
        total_duration = end_time - start_time
        tds = pd.DataFrame(all_trades)
        #tds.to_parquet(f'trades/pipeline_{id}.parquet')
        utils_helpers.append_single_parquet(df=tds, path=f'trades/out_of_sample/strategy_4x_tp_pipeline_{id}.parquet')
        print(f"⏰ Tiempo total de ejecución for pipeline {id}, with {len(df)} days: {total_duration:.2f} segundos. Total trades: {len(all_trades)}")
        utils_helpers.log(f"⏰ Tiempo total de ejecución for pipeline {id}: {total_duration:.2f} segundos. Total trades: {len(all_trades)}", f'./pipeline_logs/strategy_4x_tp_out_of_sample_log_{id}.csv')
        
        print("==============================================")
       
        
    except Exception as e:
                print("pipeline:", e)   
                end_time = tm.perf_counter()
                utils_helpers.log(f'process fail for ticker {ticker} at {date_str}, error: {e}', f'./pipeline_logs/strategy_4x_tp_out_of_sample_log_{id}.csv')
        
                # 6. CALCULAR Y MOSTRAR LA DURACIÓN
                total_duration = end_time - start_time

                print(f"⏰ Tiempo total de ejecución: {total_duration:.2f} segundos.")
                print("=================error=============================")
    
    
    
    return

@runner.pipeline
def pipeline_strategy_5x_tp(df = pd.DataFrame([]), id=0):
    
    print(f'************** strategy_5x  pipeline {id} ******************')
    utils_helpers.log(f' ***** started strategy_5x pipeline {id}', file_path=f'./pipeline_logs/strategy_5x_tp_log_{id}.csv')
    start_time = tm.perf_counter()
    all_trades = []
    
    date_str = None
    ticker = None
    
    try:
        
    
        for i in range(len(df)):
            row = df.iloc[i]
            previous_day_close = row['previous_close']
            date_str = row['date_str']
            ticker = row['ticker']
            (df_prep, indicators) = prepare_data_for_9sma_200sma_strategy(ticker, date_str, initial_equity=10000, slippage_pct=0.0005, previous_day_close = previous_day_close)
            (df_res, trades) = strategy_nx_tp(df_prep, ticker= ticker, initial_equity=10000,  previous_day_close=previous_day_close, factor_tp=5)
            all_trades = all_trades + trades
           
            
        end_time = tm.perf_counter()
            
        # 6. CALCULAR Y MOSTRAR LA DURACIÓN
        total_duration = end_time - start_time
        tds = pd.DataFrame(all_trades)
        #tds.to_parquet(f'trades/pipeline_{id}.parquet')
        utils_helpers.append_single_parquet(df=tds, path=f'trades/strategy_5x_tp_pipeline_{id}.parquet')
        print(f"⏰ Tiempo total de ejecución for pipeline {id}, with {len(df)} days: {total_duration:.2f} segundos. Total trades: {len(all_trades)}")
        utils_helpers.log(f"⏰ Tiempo total de ejecución for pipeline {id}: {total_duration:.2f} segundos. Total trades: {len(all_trades)}", f'./pipeline_logs/strategy_5x_tp_log_{id}.csv')
        
        print("==============================================")
       
        
    except Exception as e:
                print("pipeline:", e)   
                end_time = tm.perf_counter()
                utils_helpers.log(f'process fail for ticker {ticker} at {date_str}, error: {e}', f'./pipeline_logs/strategy_5x_tp_log_{id}.csv')
        
                # 6. CALCULAR Y MOSTRAR LA DURACIÓN
                total_duration = end_time - start_time

                print(f"⏰ Tiempo total de ejecución: {total_duration:.2f} segundos.")
                print("=================error=============================")
    
    
    
    return

@runner.pipeline
def pipeline_strategy_5x_tp_out_of_sample(df = pd.DataFrame([]), id=0):
    
    print(f'************** strategy_5x  out_of_sample pipeline {id} ******************')
    utils_helpers.log(f' ***** started strategy_5x pipeline {id}', file_path=f'./pipeline_logs/strategy_5x_tp_out_of_sample_log_{id}.csv')
    start_time = tm.perf_counter()
    all_trades = []
    
    date_str = None
    ticker = None
    
    try:
        
    
        for i in range(len(df)):
            row = df.iloc[i]
            previous_day_close = row['previous_close']
            date_str = row['date_str']
            ticker = row['ticker']
            (df_prep, indicators) = prepare_data_for_9sma_200sma_strategy(ticker, date_str, initial_equity=10000, slippage_pct=0.0005, previous_day_close = previous_day_close)
            (df_res, trades) = strategy_nx_tp(df_prep, ticker= ticker, initial_equity=10000,  previous_day_close=previous_day_close, factor_tp=5)
            all_trades = all_trades + trades
           
            
        end_time = tm.perf_counter()
            
        # 6. CALCULAR Y MOSTRAR LA DURACIÓN
        total_duration = end_time - start_time
        tds = pd.DataFrame(all_trades)
        #tds.to_parquet(f'trades/pipeline_{id}.parquet')
        utils_helpers.append_single_parquet(df=tds, path=f'trades/out_of_sample/strategy_5x_tp_pipeline_{id}.parquet')
        print(f"⏰ Tiempo total de ejecución for pipeline {id}, with {len(df)} days: {total_duration:.2f} segundos. Total trades: {len(all_trades)}")
        utils_helpers.log(f"⏰ Tiempo total de ejecución for pipeline {id}: {total_duration:.2f} segundos. Total trades: {len(all_trades)}", f'./pipeline_logs/strategy_5x_tp_out_of_sample_log_{id}.csv')
        
        print("==============================================")
       
        
    except Exception as e:
                print("pipeline:", e)   
                end_time = tm.perf_counter()
                utils_helpers.log(f'process fail for ticker {ticker} at {date_str}, error: {e}', f'./pipeline_logs/strategy_5x_tp_out_of_sample_log_{id}.csv')
        
                # 6. CALCULAR Y MOSTRAR LA DURACIÓN
                total_duration = end_time - start_time

                print(f"⏰ Tiempo total de ejecución: {total_duration:.2f} segundos.")
                print("=================error=============================")
    
    
    
    return

#run_tests_strategy_2x_tp()

#runner.main(init_worker= runner.init_worker, strategy_func= pipeline_strategy_dynamic_tp_stp, n_cpus= 8, chunk_size= 1000)
#runner.main(init_worker= runner.init_worker, strategy_func= pipeline_strategy_1x_tp, n_cpus= 8, chunk_size= 1000)
#runner.main(init_worker= runner.init_worker, strategy_func= pipeline_strategy_2x_tp, n_cpus= 8, chunk_size= 1000)
#runner.main(init_worker= runner.init_worker, strategy_func= pipeline_strategy_2x_sma9_up, n_cpus= 8, chunk_size= 1000)
#runner.main(init_worker= runner.init_worker, strategy_func= pipeline_strategy_3x_tp, n_cpus= 8, chunk_size= 1000)
runner.main(init_worker= runner.init_worker, strategy_func= pipeline_strategy_4x_tp, n_cpus= 8, chunk_size= 1000)
#runner.main(init_worker= runner.init_worker, strategy_func= pipeline_strategy_5x_tp, n_cpus= 8, chunk_size= 1000)

# ====== out of sample runs ======

#runner.out_of_sample(init_worker= runner.init_worker, strategy_func= pipeline_strategy_1x_tp_out_of_sample, n_cpus= 8, chunk_size= 1000)
#runner.out_of_sample(init_worker= runner.init_worker, strategy_func= pipeline_strategy_2x_tp_out_of_sample, n_cpus= 8, chunk_size= 1000)
#runner.out_of_sample(init_worker= runner.init_worker, strategy_func= pipeline_strategy_2x_sma9_up_out_of_sample, n_cpus= 8, chunk_size= 1000)
#runner.out_of_sample(init_worker= runner.init_worker, strategy_func= pipeline_strategy_3x_tp_out_of_sample, n_cpus= 8, chunk_size= 1000)
#runner.out_of_sample(init_worker= runner.init_worker, strategy_func= pipeline_strategy_4x_tp_out_of_sample, n_cpus= 8, chunk_size= 1000)
#runner.out_of_sample(init_worker= runner.init_worker, strategy_func= pipeline_strategy_5x_tp_out_of_sample, n_cpus= 8, chunk_size= 1000)





