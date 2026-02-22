import os
import sys
sys.path.insert(0, os.path.abspath("."))
import vectorbt as vbt
import pandas as pd
pd.options.display.float_format = '{:.3f}'.format

import numpy as np
from utils import helpers as utils_helpers, trade_metrics as tm
from pprint import pprint
from small_caps_strategies import commons
from datetime import datetime
import yfinance as yf
import pandas_ta as ta
from numba import njit, prange

def get_asset_from_yf(ticker, date_from="2000-01-01", date_to=datetime.today().strftime('%Y-%m-%d')):
    
    asset = yf.download(ticker, start=date_from, end=date_to)
    asset.columns = [col[0] for col in asset.columns]
    asset.rename(columns={'Open': 'open', 'Close': 'close', 'High': 'high', 'Low': 'low', 'Volume': 'volume'}, inplace=True)


    asset["sma_200"] = ta.sma(asset["close"], length=200)
    asset["sma_9"]   = ta.sma(asset["close"], length=9)
    asset["sma_5"]   = ta.sma(asset["close"], length=5)
    asset["atr"] = ta.atr(asset["high"],asset["low"],asset["close"], length=14)
    
    return asset

def prepare_n_dim_arrays(f_dict, gap_list =[], tp_list = [], sl_list= []):
    
    # Verificar que todas tengan el mismo tamaño
    assert len(gap_list) == len(tp_list) == len(sl_list)
    
    # Creamos la lista de pares “paralelos”
    tp_sl_gap_pairs = list(zip(tp_list, sl_list, gap_list))
    n_params = len(tp_sl_gap_pairs) if  len(tp_sl_gap_pairs) > 0 else 1
    

    # --------------------------------------------------
    # 1. Construir índice maestro
    # --------------------------------------------------
    index_master = pd.DatetimeIndex([])

    for df in f_dict.values():
        index_master = index_master.union(df.index)

    index_master = pd.to_datetime(index_master.sort_values())

    tickers = list(f_dict.keys())
    n_tickers = len(tickers)
    n_cols = n_tickers * n_params
    n_bars = len(index_master)
    

    # --------------------------------------------------
    # 2. Crear arrays base
    # --------------------------------------------------
    open_arr  = np.full((n_bars, n_cols), np.nan)
    high_arr  = np.full_like(open_arr, np.nan)
    low_arr   = np.full_like(open_arr, np.nan)
    close_arr = np.full_like(open_arr, np.nan)
    
    atr_arr   = np.full_like(open_arr, np.nan)
    volume_arr = np.full_like(open_arr, np.nan)
    rvol_arr   = np.full_like(open_arr, np.nan)
    prev_day_close_arr = np.full_like(open_arr, np.nan)
    exhaustion_score_arr = np.full_like(open_arr, np.nan)
    sma_volume_20_5m_arr = np.full_like(open_arr, np.nan)
    vwap_arr = np.full_like(open_arr, np.nan)
    
    col = 0
    col_meta = []   # para mapear trades → parámetros
    
    
    for ticker in tickers:
        df = f_dict[ticker].reindex(index_master)
        
        # Si no hay combinaciones de parámetros, se crea 1 por ticker
        param_iter = tp_sl_gap_pairs if len(tp_sl_gap_pairs) > 0 else [(None, None, None)]

       
        idx = ~df['open'].isna()

        open_arr[idx, col]  = df.loc[idx, 'open'].values
        high_arr[idx, col]  = df.loc[idx, 'high'].values
        low_arr[idx, col]   = df.loc[idx, 'low'].values
        close_arr[idx, col] = df.loc[idx, 'close'].values
        #atr_arr[idx, col]   = df.loc[idx, 'atr'].values
        #volume_arr[idx, col]     = df.loc[idx, 'volume'].values
        #rvol_arr[idx, col]       = df.loc[idx, 'RVOL_daily'].values
        #prev_day_close_arr[idx, col] = df.loc[idx, 'previous_day_close'].values
        #sma_volume_20_5m_arr[idx, col] = df.loc[idx, 'SMA_VOLUME_20_5m'].values
        #vwap_arr[idx, col] = df.loc[idx, 'vwap'].values
        
        col_meta.append({
            'ticker': ticker,
            'column': col    
        })

        col += 1
    
    max_entry_time =  '15:45'
    # # -----------------------
    # # Filtro horario (<15:45)
    # # -----------------------
    # time_mask = np.array([
    #     t.strftime("%H:%M") < max_entry_time
    #     for t in index_master
    # ])
    
    hours   = np.array([t.hour for t in index_master])
    minutes = np.array([t.minute for t in index_master])

    time_mask = (
        ( (hours > 8) | ((hours == 8) & (minutes >= 15))) &
        ((hours < 15) | ((hours == 15) & (minutes <= 45)))
    )
    
    time_mask = (
        # 04:00 - 07:59
        (
            (hours >= 4) & (hours < 8)
        )
        |
        # 08:30 - 14:00
        (
            ((hours == 8) & (minutes >= 30)) |
            ((hours > 8) & (hours < 14)) |
            ((hours == 14) & (minutes == 0))
        )
    )
    
    
    all_params ={}
    all_params.update({
        "n_params":n_params,
        "tp_sl_gap_pairs":tp_sl_gap_pairs,
        "index_master":index_master,
        "n_tickers":n_tickers,
        "n_cols":n_cols,
        "n_bars": n_bars,
        "open_arr":open_arr,
        "high_arr":high_arr,
        "low_arr":low_arr,
        "close_arr":close_arr,
        "atr_arr": atr_arr,
        "volume_arr": volume_arr,
        "rvol_arr":rvol_arr,
        "prev_day_close_arr": prev_day_close_arr,
        "sma_volume_20_5m_arr":sma_volume_20_5m_arr,
        "vwap_arr": vwap_arr,
        "col":col,
        "col_meta":col_meta,
        "max_entry_time": max_entry_time,
        "time_mask":time_mask
    })
    
    return all_params

def walk_fordward_split(ticker="", df=pd.DataFrame(), train_days=252, test_days=160):
    
    n = len(df)
    splits = []
    df_diff = 150
    total_intervals = 4
    df_dict = {}
    
    if n > total_intervals * df_diff + train_days + test_days:
        
        for i in range(0, total_intervals):
            train = df[i*df_diff:i*df_diff + train_days]
            test = df[i*df_diff + train_days: i*df_diff + train_days + test_days]
            
            splits.append((train, test, f'{ticker}_{i}'))
            df_dict[f'{ticker}_{i}_train'] = train
            df_dict[f'{ticker}_{i}_test'] = test
        
             
    return df_dict
# ================ strategies ====================
#
# ================================================

def breakout_donchain(f_dict, initial_equity=10000):
    """
    Trend following strategy, good bullish assest like SPY, NASDAQ, GOLD, SLV. Timeframes >= 1h
    BIAS: Long only
    f_dict: dictionary with ticker name as key and data as value ex: {"TSLA": dataframe, "QQQ":dataframe2,....}
    
    Rules: Close above upper band of donchain channel -> buy
           Close bellow lower band -> sell
           Donchain channel : lookback=5 and offset=1
    """
    
    
    all_params = prepare_n_dim_arrays(f_dict)
    
    n_bars = all_params['n_bars']
    n_cols = all_params['n_cols']
    index_master = all_params['index_master']
    close_arr = all_params['close_arr']
    col_meta = all_params['col_meta']
    
    lookback = 5
    offset = 1
    donchain_upper_arr  = np.full((n_bars, n_cols), np.nan)
    donchain_lower_arr  = np.full((n_bars, n_cols), np.nan)
    atr_arr  = np.full((n_bars, n_cols), np.nan)
    
    col = 0
    
    tickers = list(f_dict.keys())
     
    for ticker in tickers:
        df = f_dict[ticker].reindex(index_master)
        
        donchain_upper_arr[:, col]  = (
        df['high']
        .rolling(window=lookback, min_periods=lookback)
        .max()
        .shift(offset))
        
        donchain_lower_arr[:, col]  = (
        df['low']
        .rolling(window=lookback, min_periods=lookback)
        .min()
        .shift(offset))
        
        atr_arr[:, col] = utils_helpers.compute_atr(df)

        col += 1
    
    # =============================
    # Entry Conditions
    # =============================
    long_entries = close_arr > donchain_upper_arr
  

    # =============================
    # Exit Conditions (reversal logic)
    # =============================
    long_exits = close_arr < donchain_lower_arr
   

   
    
    # =============================
    # Portfolio
    # =============================
    pf = vbt.Portfolio.from_signals(
        close=close_arr, 
        entries=long_entries,
        exits=long_exits,
        price=close_arr,
        direction='longonly',
        fees=0.001,
        size=1,
        #size_type="percent",  # % del equity
        init_cash= 0 ,#initial_equity,
        freq="1D"  # cambia si es intraday
    )
    
    # ============================
    # Trades
    # ============================
    
    trades = pf.trades.records_readable
    trades = trades = (
    trades
    .replace([np.inf, -np.inf], np.nan)
    .dropna()).copy()
    
    trades['entry_time'] = index_master[trades['Entry Timestamp'].values]
    trades['exit_time']  = index_master[trades['Exit Timestamp'].values]
    trades['day'] = trades['entry_time'].dt.normalize()
    entry_idx = trades['Entry Timestamp'].values
    
    entry_idx = trades['Entry Timestamp'].values
    col_idx   = trades['Column'].values

    trades['stop_loss_price'] = donchain_lower_arr[entry_idx, col_idx] 

    trades = trades.rename(columns={
        'Avg Entry Price': 'entry_price',
        'Avg Exit Price': 'exit_price',
        'PnL': 'pnl',
        'Direction':'type'
    })
    
    col_meta_df = pd.DataFrame(col_meta).set_index('column')
    trades = trades.join(col_meta_df, on='Column')
    
    trades = trades.replace([np.inf, -np.inf], np.nan).dropna()
    trades['strategy'] =  'breakout_donchain_multi_asset'
    trades['is_profit'] =  trades['pnl'] > 0
   
    grouped = trades.groupby('ticker')
    
    result = {
    }
    
    for ticker, group in grouped:
        result[ticker] = group
    
    
    return result
   
   
def mean_reversion_donchain(f_dict, initial_equity=10000):
    
    """
    Mean strategy, good bullish assest like SPY, NASDAQ, GOLD, SLV. Timeframes >= 1h
    BIAS: Long only
    f_dict: dictionary with ticker name as key and data as value ex: {"TSLA": dataframe, "QQQ":dataframe2,....}
    
    Rules: Close bellow lower band of donchain channel -> buy
           Close bellow upper band -> sell
           Donchain channel : lookback=5 and offset=1
    """
    
    
    all_params = prepare_n_dim_arrays(f_dict)
    
    n_bars = all_params['n_bars']
    n_cols = all_params['n_cols']
    index_master = all_params['index_master']
    close_arr = all_params['close_arr']
    open_arr = all_params['open_arr']
    col_meta = all_params['col_meta']
    
    lookback = 5
    offset = 1
    donchain_upper_arr  = np.full((n_bars, n_cols), np.nan)
    donchain_lower_arr  = np.full((n_bars, n_cols), np.nan)
    atr_arr  = np.full((n_bars, n_cols), np.nan)
    
    col = 0
    
    tickers = list(f_dict.keys())
     
    for ticker in tickers:
        df = f_dict[ticker].reindex(index_master)
        
        donchain_upper_arr[:, col]  = (
        df['high']
        .rolling(window=lookback, min_periods=lookback)
        .max()
        .shift(offset))
        
        donchain_lower_arr[:, col]  = (
        df['low']
        .rolling(window=lookback, min_periods=lookback)
        .min()
        .shift(offset))
        
        atr_arr[:, col] = utils_helpers.compute_atr(df)

        col += 1
    
    # =============================
    # Entry Conditions
    # =============================
    long_entries = close_arr < donchain_lower_arr
    
    tp_stop  = (close_arr - 3 * atr_arr) / close_arr
   

    # =============================
    # Exit Conditions (reversal logic)
    # =============================
    long_exits =  close_arr > donchain_upper_arr
   

   
    
    # =============================
    # Portfolio
    # =============================
    pf = vbt.Portfolio.from_signals(
        close=close_arr, 
        entries=long_entries,
        exits=long_exits,
        price=close_arr,
        direction='longonly',
        fees=0.001,
        tp_stop=tp_stop,
        size=0.2,
        size_type="percent",  # % del equity
        init_cash=initial_equity,

        freq="1D"  # cambia si es intraday
    )
    
    # ============================
    # Trades
    # ============================
    
    trades = pf.trades.records_readable
    trades = trades = (
    trades
    .replace([np.inf, -np.inf], np.nan)
    .dropna()).copy()
    
    trades['entry_time'] = index_master[trades['Entry Timestamp'].values]
    trades['exit_time']  = index_master[trades['Exit Timestamp'].values]
    trades['day'] = trades['entry_time'].dt.normalize()
    entry_idx = trades['Entry Timestamp'].values
    col_idx   = trades['Column'].values

    trades['stop_loss_price'] = open_arr[entry_idx, col_idx] - 3 * atr_arr[entry_idx, col_idx]

    trades = trades.rename(columns={
        'Avg Entry Price': 'entry_price',
        'Avg Exit Price': 'exit_price',
        'PnL': 'pnl',
        'Direction':'type'
    })
    
  
    
    col_meta_df = pd.DataFrame(col_meta).set_index('column')
    trades = trades.join(col_meta_df, on='Column')
    
    trades = trades.replace([np.inf, -np.inf], np.nan).dropna()
    trades['strategy'] =  'mean_reversion_donchain_multi_asset'
    trades['is_profit'] =  trades['pnl'] > 0
    
    
   
    
    grouped = trades.groupby('ticker')
    
    result = {
    }
    
    for ticker, group in grouped:
        result[ticker] = group
    
    
    return result
    
    

