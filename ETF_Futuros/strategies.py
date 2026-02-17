import os
import sys
sys.path.insert(0, os.path.abspath("."))
import vectorbt as vbt
import pandas as pd
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

# ================ strategies ====================
#
# ================================================
def breakout_donchain(df):
    
    don_channel_len = 5
    offset = 1
    
    # =============================
    # Donchian Channel
    # =============================
    high = df["high"]
    low = df["low"]
    close = df["close"]

    upper = high.rolling(don_channel_len).max()
    lower = low.rolling(don_channel_len).min()

    # Apply offset (same logic as Pine: [1 + offset])
    upper_shifted = upper.shift(offset)
    lower_shifted = lower.shift(offset)
    
    # =============================
    # Entry Conditions
    # =============================
    long_entries = close > upper_shifted
    short_entries = close < lower_shifted

    # =============================
    # Exit Conditions (reversal logic)
    # =============================
    long_exits = close < lower_shifted
    short_exits = close > upper_shifted

   
    
    # =============================
    # Portfolio
    # =============================
    pf = vbt.Portfolio.from_signals(
        close=close, 
        entries=long_entries,
        exits=long_exits,
        price=close,
        direction='longonly',
        fees=0.001,
        size=0.2,
        size_type="percent",  # % del equity
        init_cash=10_000,
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

    trades = trades.rename(columns={
        'Avg Entry Price': 'entry_price',
        'Avg Exit Price': 'exit_price',
        'PnL': 'pnl',
        'Direction':'type'
    })
    

    # =============================
    # Results
    # =============================
    print(pf.stats())
    #pf.plot().show()
    
    print(trades)
    
    
    return


def breakout_donchain_multi_asset(f_dict):
    
    
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
        #size=0.2,
        #size_type="percent",  # % del equity
        #init_cash=10_000,
         size=1,
        init_cash=0,
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
    
    for ticker, group in grouped:
        print(ticker)
        print(group)
    
    
    return grouped
   
   
def mean_reversion_donchain_multi_asset(f_dict):
    
    
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

        col += 1
    
    # =============================
    # Entry Conditions
    # =============================
    long_entries = close_arr < donchain_lower_arr
   

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
        #size=0.2,
        #size_type="percent",  # % del equity
        #init_cash=10_000,
         size=1,
        init_cash=0,
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
    
    for ticker, group in grouped:
        print(ticker)
        print(group)
        
    return grouped
    
    

