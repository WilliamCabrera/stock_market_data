import vectorbt as vbt
import pandas as pd
import numpy as np
from utils import helpers as utils_helpers, trade_metrics as tme
import itertools
import time as tm
from functools import wraps
from small_caps_strategies import commons
from pprint import pprint
from pathlib import Path

dt = {}
# ticker = "SIDU"
# date_str = '2025-12-26'
# _from = pd.to_datetime(date_str)
# _to = pd.to_datetime(date_str)
# (df1, _) = utils_helpers.get_data_5min_for_backtest(ticker, _from.strftime("%Y-%m-%d"), _to.strftime("%Y-%m-%d"))
# df1['ticker'] = ticker
# atr = utils_helpers.compute_atr(df1)
# df1['atr'] = atr
# df1 = df1.set_index('date').sort_index()


# ticker = "SIDU"
# date_str = '2025-12-27'
# _from = pd.to_datetime(date_str)
# _to = pd.to_datetime(date_str)
# (df3, _) = utils_helpers.get_data_5min_for_backtest(ticker, _from.strftime("%Y-%m-%d"), _to.strftime("%Y-%m-%d"))
# df3['ticker'] = ticker
# atr = utils_helpers.compute_atr(df3)
# df3['atr'] = atr
# df3 = df3.set_index('date').sort_index()


# ticker = "SOPA"
# date_str = '2025-12-26'
# _from = pd.to_datetime(date_str)
# _to = pd.to_datetime(date_str)
# (df2, _) = utils_helpers.get_data_5min_for_backtest(ticker, _from.strftime("%Y-%m-%d"), _to.strftime("%Y-%m-%d"))
# df2['ticker'] = ticker
# df2['atr'] = atr

# ====== helper functions ======

def prepare_params_and_vectors(f_dict, tp_list, sl_list ):
    
    all_params = {}
    assert len(tp_list) == len(sl_list)
    tp_sl_pairs = list(zip(tp_list, sl_list))
    n_params = len(tp_sl_pairs)
    

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

        for tp, sl in tp_sl_pairs:
            idx = ~df['open'].isna()

            open_arr[idx, col]  = df.loc[idx, 'open'].values
            high_arr[idx, col]  = df.loc[idx, 'high'].values
            low_arr[idx, col]   = df.loc[idx, 'low'].values
            close_arr[idx, col] = df.loc[idx, 'close'].values
            atr_arr[idx, col]   = df.loc[idx, 'atr'].values
            volume_arr[idx, col]     = df.loc[idx, 'volume'].values
            rvol_arr[idx, col]       = df.loc[idx, 'RVOL_daily'].values
            prev_day_close_arr[idx, col] = df.loc[idx, 'previous_day_close'].values
            sma_volume_20_5m_arr[idx, col] = df.loc[idx, 'SMA_VOLUME_20_5m'].values
            vwap_arr[idx, col] = df.loc[idx, 'vwap'].values
            
            col_meta.append({
                'ticker': ticker,
                'tp': tp,
                'sl': sl,
                'column': col    
            })

            col += 1
    all_params.update({
        "n_params":n_params,
        "tp_sl_pairs":tp_sl_pairs,
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
        "col_meta":col_meta
    })
    
    return all_params

def prepare_params_and_vectors_for_gappers(f_dict, gap_list =[], tp_list = [], sl_list= []):
    
    # Verificar que todas tengan el mismo tamaño
    assert len(gap_list) == len(tp_list) == len(sl_list)
    
    # Creamos la lista de pares “paralelos”
    tp_sl_gap_pairs = list(zip(tp_list, sl_list, gap_list))
    n_params = len(tp_sl_gap_pairs)
    

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

        for tp, sl, gap in tp_sl_gap_pairs:
            idx = ~df['open'].isna()

            open_arr[idx, col]  = df.loc[idx, 'open'].values
            high_arr[idx, col]  = df.loc[idx, 'high'].values
            low_arr[idx, col]   = df.loc[idx, 'low'].values
            close_arr[idx, col] = df.loc[idx, 'close'].values
            atr_arr[idx, col]   = df.loc[idx, 'atr'].values
            volume_arr[idx, col]     = df.loc[idx, 'volume'].values
            rvol_arr[idx, col]       = df.loc[idx, 'RVOL_daily'].values
            prev_day_close_arr[idx, col] = df.loc[idx, 'previous_day_close'].values
            sma_volume_20_5m_arr[idx, col] = df.loc[idx, 'SMA_VOLUME_20_5m'].values
            vwap_arr[idx, col] = df.loc[idx, 'vwap'].values
            
            col_meta.append({
                'ticker': ticker,
                'tp': tp,
                'sl': sl,
                'gap': gap,
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


def prepare_params_and_vectors_for_gappers_with_trailing(f_dict, gap_list =[], tp_list = []):
    
    # Verificar que todas tengan el mismo tamaño
    assert len(gap_list) == len(tp_list)
    
    # Creamos la lista de pares “paralelos”
    tp_sl_gap_pairs = list(zip(tp_list, gap_list))
    n_params = len(tp_sl_gap_pairs)
    

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
    donchian_upper_arr = np.full_like(open_arr, np.nan)
    donchian_basis_arr = np.full_like(open_arr, np.nan)
    donchian_lower_arr = np.full_like(open_arr, np.nan)
    
    col = 0
    col_meta = []   # para mapear trades → parámetros
    
    for ticker in tickers:
        df = f_dict[ticker].reindex(index_master)

        for tp, gap in tp_sl_gap_pairs:
            idx = ~df['open'].isna()

            open_arr[idx, col]  = df.loc[idx, 'open'].values
            high_arr[idx, col]  = df.loc[idx, 'high'].values
            low_arr[idx, col]   = df.loc[idx, 'low'].values
            close_arr[idx, col] = df.loc[idx, 'close'].values
            atr_arr[idx, col]   = df.loc[idx, 'atr'].values
            volume_arr[idx, col]     = df.loc[idx, 'volume'].values
            rvol_arr[idx, col]       = df.loc[idx, 'RVOL_daily'].values
            prev_day_close_arr[idx, col] = df.loc[idx, 'previous_day_close'].values
            sma_volume_20_5m_arr[idx, col] = df.loc[idx, 'SMA_VOLUME_20_5m'].values
            vwap_arr[idx, col] = df.loc[idx, 'vwap'].values
            donchian_upper_arr[idx, col] = df.loc[idx, 'donchian_upper'].values
            donchian_basis_arr[idx, col] = df.loc[idx, 'donchian_basis'].values
            donchian_lower_arr[idx, col] = df.loc[idx, 'donchian_lower'].values
            
            col_meta.append({
                'ticker': ticker,
                'tp': tp,
                'gap': gap,
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
        "time_mask":time_mask,
        "donchian_upper_arr": donchian_upper_arr,   
        "donchian_basis_arr": donchian_basis_arr,
        "donchian_lower_arr": donchian_lower_arr
    })
    
    return all_params

def modify_trades_columns(params=None, strategy_name_prefix = "strategy"):
    if params is None:
        return pd.DataFrame([])
    
    trades = params['trades']
    col_meta = params['col_meta']
    index_master = params['index_master']
    atr_arr = params['atr_arr']
    rvol_arr = params['rvol_arr']
    prev_day_close_arr = params['prev_day_close_arr']
    volume_arr = params['volume_arr']
    
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
    trades['strategy'] = f'{strategy_name_prefix}_'+ trades['tp'].astype(str) + "_" + trades['sl'].astype(str)+ "_" + trades['gap'].astype(str)
    trades['entry_time'] = index_master[trades['Entry Timestamp'].values]
    trades['exit_time']  = index_master[trades['Exit Timestamp'].values]
    entry_idx = trades['Entry Timestamp'].values
    col_idx   = trades['Column'].values
    
    atr_entry = atr_arr[entry_idx, col_idx]
    trades['stop_loss_price'] = (trades['entry_price'] + trades['sl'] * atr_entry)
    trades['rvol_daily'] = rvol_arr[entry_idx, col_idx]
    trades['previous_day_close'] = prev_day_close_arr[entry_idx, col_idx]
    trades['volume'] = volume_arr[entry_idx, col_idx]

    return trades[[ 'ticker', 'type','entry_price',
    'exit_price','stop_loss_price',  'pnl',
    'Return', 'rvol_daily',  'previous_day_close','volume','entry_time','exit_time','strategy']]
  
def modify_trades_columns_trailing(params=None, strategy_name_prefix = "strategy"):
    if params is None:
        return pd.DataFrame([])
    
    trades = params['trades']
    col_meta = params['col_meta']
    index_master = params['index_master']
    atr_arr = params['atr_arr']
    rvol_arr = params['rvol_arr']
    prev_day_close_arr = params['prev_day_close_arr']
    volume_arr = params['volume_arr']
    donchian_upper_arr = params['donchian_upper_arr']
    
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
    trades['strategy'] = f'{strategy_name_prefix}_'+ trades['tp'].astype(str) + "_"+ trades['gap'].astype(str)
    trades['entry_time'] = index_master[trades['Entry Timestamp'].values]
    trades['exit_time']  = index_master[trades['Exit Timestamp'].values]
    entry_idx = trades['Entry Timestamp'].values
    col_idx   = trades['Column'].values
    
    atr_entry = atr_arr[entry_idx, col_idx]
    trades['stop_loss_price'] = donchian_upper_arr[entry_idx, col_idx] 
    trades['rvol_daily'] = rvol_arr[entry_idx, col_idx]
    trades['previous_day_close'] = prev_day_close_arr[entry_idx, col_idx]
    trades['volume'] = volume_arr[entry_idx, col_idx]

    return trades[[ 'ticker', 'type','entry_price',
    'exit_price','stop_loss_price',  'pnl',
    'Return', 'rvol_daily',  'previous_day_close','volume','entry_time','exit_time','strategy']]
    
def reduce_trades_columns(trades):
    if trades is None or len(trades) == 0:
        return pd.DataFrame([])
    
    return trades[[ 'ticker', 'type','entry_price',
    'exit_price','stop_loss_price',  'pnl',
    'Return', 'rvol_daily',  'previous_day_close','volume','entry_time','exit_time','strategy']]
  
def generate_signal_to_force_close_EOD(forced_exit, index_master):
    
    dates = pd.to_datetime(index_master.date)  # fecha sin hora
    unique_dates = np.unique(dates)

    for d in unique_dates:
        # tomamos todas las barras del día
        mask_day = dates == d
        
        # todas las barras del día que sean <= 15:50
        mask_before_1550 = mask_day & (index_master.time <= pd.to_datetime("15:50").time())
        
        if np.any(mask_before_1550):
            # cerramos en la última barra antes de las 15:50
            last_bar_idx = np.where(mask_before_1550)[0][-1]
        else:
            # si no hay barra antes de 15:50, cerrar en la última del día
            last_bar_idx = np.where(mask_day)[0][-1]
            
        forced_exit[last_bar_idx, :] = True  # cerramos todas las posiciones abiertas

        
    return forced_exit

def save_trades(trades, path="vectorbt_trades", append=True):
    """
    trades: trades dataframe
    path: path to the folder where trades will be saved
    """
    if trades is None or isinstance(trades, pd.DataFrame) == False or len(trades) == 0 :
        return
    
    if 'strategy' in trades.columns:
        
        # Agrupar por la columna 'strategy'
        grouped = trades.groupby('strategy')

        # Iterar por cada grupo
        for strategy_name, group_df in grouped:
            #print("Strategy:", strategy_name)
            #print(group_df)  # Aquí tienes el DataFrame solo de esa estrategia
            if append:
                utils_helpers.append_single_parquet(df=group_df, path=f'{path}/{strategy_name}.parquet')
            else:  
                group_df.to_parquet(path=f'{path}/{strategy_name}.parquet')
        
    else:
        print("====== el dataframe trades no tiene la column: strategy , la cual contiene el nombre de la estrategia que se esta probando")
    
    
    
    return

def save_trades_to_file(trades, file_path="vectorbt_trades", append=True):
    """
    trades: trades dataframe
    path: path to the folder where trades will be saved
    """
    if trades is None or isinstance(trades, pd.DataFrame) == False or len(trades) == 0 :
        return
    
    if 'strategy' in trades.columns:
        
        # Agrupar por la columna 'strategy'
        grouped = trades.groupby('strategy')

        # Iterar por cada grupo
        for strategy_name, group_df in grouped:
            #print("Strategy:", strategy_name)
            #print(group_df)  # Aquí tienes el DataFrame solo de esa estrategia
            if append:
                utils_helpers.append_single_parquet(df=group_df, path=file_path)
            else:  
                group_df.to_parquet(path=file_path)
        
    else:
        print("====== el dataframe trades no tiene la column: strategy , la cual contiene el nombre de la estrategia que se esta probando")
    
    
    
    return


# ====== examples code ======= 

  
    
def backtest_multiple_dias_vars(ticker="SIDU"):
    
    dias = ['2025-12-26', '2025-12-29']
    tp_list = [0.15, 0.20]       # TP relativos
    sl_list = [3.5, 4.5]         # SL en múltiplos de ATR
    
    df_dict = {}
    for d in dias:
        _from = pd.to_datetime(d)
        _to = pd.to_datetime(d)
        (df3, _) = utils_helpers.get_data_5min_for_backtest(ticker, _from.strftime("%Y-%m-%d"), _to.strftime("%Y-%m-%d"))
        df3['ticker'] = ticker
        atr = utils_helpers.compute_atr(df3)  # ATR calculado
        df3['atr'] = atr
        df3 = df3.set_index('date').sort_index()  # poner date como índice
        df_dict[d] = df3

    # ------------------------------
    # 2️⃣ Crear índice maestro
    # ------------------------------
    index_master = pd.DatetimeIndex([])
    for d in dias:
        index_master = index_master.union(df_dict[d].index)
    index_master = index_master.sort_values()
    index_master = pd.to_datetime(index_master)

    n_bars = len(index_master)
    n_tickers = 1  # solo 1 ticker
    # ------------------------------
    # 3️⃣ Crear arrays base
    # ------------------------------
    open_arr  = np.full((n_bars, n_tickers), np.nan)
    high_arr  = np.full_like(open_arr, np.nan)
    low_arr   = np.full_like(open_arr, np.nan)
    close_arr = np.full_like(open_arr, np.nan)
    atr_arr   = np.full_like(open_arr, np.nan)

    # Rellenar arrays con datos por día
    for d in dias:
        df = df_dict[d].reindex(index_master)
        idx_valid = ~df['open'].isna()
        open_arr[idx_valid, 0]  = df.loc[idx_valid, 'open'].values
        high_arr[idx_valid, 0]  = df.loc[idx_valid, 'high'].values
        low_arr[idx_valid, 0]   = df.loc[idx_valid, 'low'].values
        close_arr[idx_valid, 0] = df.loc[idx_valid, 'close'].values
        atr_arr[idx_valid, 0]   = df.loc[idx_valid, 'atr'].values

    # ------------------------------
    # 4️⃣ Entradas a las 9:30
    # ------------------------------
    mask_930 = np.array([t.strftime("%H:%M")=="09:30" for t in index_master])
    entries = np.zeros((n_bars, n_tickers), dtype=bool)
    entries[mask_930, :] = ~np.isnan(open_arr[mask_930, :])

    # ------------------------------
    # 5️⃣ Crear combinaciones de TP y SL
    # ------------------------------
    #combinations = list(itertools.product(tp_list, sl_list))
    combinations = list(zip(tp_list, sl_list))
    n_combos = len(combinations)
    
    

    # Arrays 3D: (n_bars, n_tickers, n_combos)
    tp_stop_arr = np.full((n_bars, n_tickers, n_combos), np.nan)
    sl_stop_arr = np.full((n_bars, n_tickers, n_combos), np.nan)
    entries_arr = np.repeat(entries[:, :, np.newaxis], n_combos, axis=2)
    
    close_arr_flat = np.repeat(close_arr, n_combos, axis=1)  # shape (n_bars, n_combos)
    high_arr_flat  = np.repeat(high_arr, n_combos, axis=1)
    low_arr_flat   = np.repeat(low_arr, n_combos, axis=1)
    entries_arr_flat = entries_arr.reshape(n_bars, n_combos)
    tp_stop_flat = tp_stop_arr.reshape(n_bars, n_combos)
    sl_stop_flat = sl_stop_arr.reshape(n_bars, n_combos)

    # Rellenar TP/SL por combinación
    for i, (tp, sl) in enumerate(combinations):
        # TP relativo
        tp_price = open_arr * (1 - tp)
        tp_stop_arr[mask_930, 0, i] = (open_arr[mask_930, 0] - tp_price[mask_930, 0]) / open_arr[mask_930, 0]
        # SL relativo
        sl_price = open_arr + sl * atr_arr
        sl_stop_arr[mask_930, 0, i] = (sl_price[mask_930, 0] - open_arr[mask_930, 0]) / open_arr[mask_930, 0]

    # ------------------------------
    # 6️⃣ Crear Portfolio Vectorbt
    # ------------------------------
    pf = vbt.Portfolio.from_signals(
         close=close_arr_flat,
        high=high_arr_flat,
        low=low_arr_flat,
        entries=entries_arr_flat,
        
        exits=False,
        size=1,             # short 1 share
       tp_stop=tp_stop_flat,
    sl_stop=sl_stop_flat,
        init_cash=0,
        direction='shortonly',
        freq='5min'
    )
    
    trades =  pf.trades.records_readable
    trades['Ticker'] = ticker
    print(trades)
   
    return trades

# ==== short strategies =======

def gap_crap_strategy(f_dict):
    """
    short at the open 
    1 take profit at 15%
    1 stop loss at 3.5 ATR
    params: f_dict: {ticker: df_5m, ...}
    """
    
    print("Starting gap_crap_strategy backtest...")

    tp_list = [0.15, 0.20, 1, 0.20, 1]       # TP relativos
    sl_list = [3.5, 3.5, 3.5,3.5, 3.5]         # SL en múltiplos de ATR
    gap_list = [0.5, 0.5, 0.5, 0.8, 0.8]         # GAPs list to 
    
    all_params = prepare_params_and_vectors_for_gappers(f_dict,gap_list, tp_list, sl_list)
   
    tp_sl_gap_pairs = all_params['tp_sl_gap_pairs']
    n_params = all_params['n_params']
    index_master = all_params['index_master']
    n_tickers = all_params['n_tickers']
    n_cols = all_params['n_cols']
    n_bars = all_params['n_bars']
    open_arr  = all_params['open_arr']
    high_arr  = all_params['high_arr']
    low_arr   = all_params['low_arr']
    close_arr = all_params['close_arr']
    atr_arr   = all_params['atr_arr']
    volume_arr = all_params['volume_arr']
    rvol_arr   = all_params['rvol_arr']
    prev_day_close_arr = all_params['prev_day_close_arr']
    col = all_params['col']
    col_meta = all_params['col_meta']
    
    # --------------------------------------------------
    # Inicializar entradas
    # --------------------------------------------------
    entries = np.zeros((n_bars, n_cols), dtype=bool)

    # Usamos 9:25 para evitar lookahead bias (close 9:25 = open 9:30)
    mask_930 = np.array([t.strftime("%H:%M") == "09:25" for t in index_master])

    # --------------------------------------------------
    # Gap % vs previous_day_close
    # --------------------------------------------------
    gap_vals = np.array([m["gap"] for m in col_meta])

    gap_pct = np.divide(
        close_arr - prev_day_close_arr,
        prev_day_close_arr,
        out=np.zeros_like(close_arr, dtype=float),
        where=prev_day_close_arr > 0
    )

    # Condición de gap: cada columna usa su GAP específico
    gap_cond = (gap_pct >= gap_vals) & (gap_pct <= 5.0)

    # Opcionales: volumen y rvol
    vol_cond  = volume_arr > 40_000
    rvol_cond = rvol_arr >= 3

    # --------------------------------------------------
    # Entradas
    # --------------------------------------------------
    entries[mask_930, :] = (
        ~np.isnan(open_arr[mask_930, :]) &
        gap_cond[mask_930, :] &
        vol_cond[mask_930, :] &
        rvol_cond[mask_930, :]
    )

    # --------------------------------------------------
    # TAKE PROFIT y STOP LOSS
    # --------------------------------------------------
    # TP: ya definido por cada columna
    tp_vals = np.array([m["tp"] for m in col_meta])
    sl_vals = np.array([3.5 for m in col_meta])  # SL = 3.5*ATR para todos

    tp_price = open_arr * (1 - tp_vals)         # short
    sl_price = open_arr + sl_vals * atr_arr    # stop sobre open + 3.5*ATR

    # Stop en % para vectorbt
    tp_stop = np.full_like(open_arr, np.nan)
    sl_stop = np.full_like(open_arr, np.nan)

    tp_stop[mask_930, :] = (open_arr[mask_930, :] - tp_price[mask_930, :]) / open_arr[mask_930, :]
    sl_stop[mask_930, :] = (sl_price[mask_930, :] - open_arr[mask_930, :]) / open_arr[mask_930, :]

    # Por si quieres guardar los precios absolutos
    stop_l = sl_price[mask_930, :]
    
    forced_exit = np.zeros_like(entries, dtype=bool)
    
    forced_exit = generate_signal_to_force_close_EOD(forced_exit,index_master)

    pf = vbt.Portfolio.from_signals(
        close=close_arr,
        high=high_arr,
        low=low_arr,
        entries=entries,
        price=open_arr,
        exits=forced_exit,
        size=1,
        direction='shortonly',
        tp_stop=tp_stop,
        sl_stop=sl_stop,
        init_cash=0,
        freq='5min'
    )   
        
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
    trades['strategy'] = f'gap_and_crap_strategy_'+ trades['tp'].astype(str) + "_" + trades['sl'].astype(str)+ "_" + trades['gap'].astype(str)
    trades['entry_time'] = index_master[trades['Entry Timestamp'].values]
    trades['exit_time']  = index_master[trades['Exit Timestamp'].values]
    entry_idx = trades['Entry Timestamp'].values
    col_idx   = trades['Column'].values
    
    atr_entry = atr_arr[entry_idx, col_idx]
    trades['stop_loss_price'] = (trades['entry_price'] + trades['sl'] * atr_entry)
    trades['rvol_daily'] = rvol_arr[entry_idx, col_idx]
    trades['previous_day_close'] = prev_day_close_arr[entry_idx, col_idx]
    trades['volume'] = volume_arr[entry_idx, col_idx]
    
   
    return reduce_trades_columns(trades)   

def short_exhaustion_strategy(f_dict):
    
    """
    short when getting signal of exhaustion  (above vwap only)
    1 take profit at 15%
    1 stop loss at 3.5 ATR
    params: f_dict: {ticker: df_5m, ...}
    """
    
    print("Starting short_exhaustion backtest...")
    
    try:
       
        tp_list = [0.15, 0.15, 20]       # TP relativos
        sl_list = [3.5, 3.5, 3.5]         # SL en múltiplos de ATR
        gap_list = [0.5, 0.8,1]         # GAPs list 
        
        all_params = prepare_params_and_vectors_for_gappers(f_dict,gap_list, tp_list, sl_list)
   
        tp_sl_gap_pairs = all_params['tp_sl_gap_pairs']
        n_params = all_params['n_params']
        index_master = all_params['index_master']
        n_tickers = all_params['n_tickers']
        n_cols = all_params['n_cols']
        n_bars = all_params['n_bars']
        open_arr  = all_params['open_arr']
        high_arr  = all_params['high_arr']
        low_arr   = all_params['low_arr']
        close_arr = all_params['close_arr']
        atr_arr   = all_params['atr_arr']
        volume_arr = all_params['volume_arr']
        rvol_arr   = all_params['rvol_arr']
        prev_day_close_arr = all_params['prev_day_close_arr']
        sma_volume_20_5m_arr = all_params['sma_volume_20_5m_arr']
        vwap_arr = all_params['vwap_arr']
        col = all_params['col']
        col_meta = all_params['col_meta']
        time_mask = all_params['time_mask']
       
        
        # --------------------------------------------------
        # Gap % vs previous_day_close
        # --------------------------------------------------
        gap_vals = np.array([m["gap"] for m in col_meta])

        gap_pct = np.divide(
            close_arr - prev_day_close_arr,
            prev_day_close_arr,
            out=np.zeros_like(close_arr, dtype=float),
            where=prev_day_close_arr > 0
        )

        # Condición de gap: cada columna usa su GAP específico
        gap_cond = (gap_pct >= gap_vals) & (gap_pct <= 5.0)   
         
        vwap_pct = (close_arr - vwap_arr) / vwap_arr
        body_size = np.abs(close_arr - open_arr)
        body_size_avg_10 = vbt.nb.rolling_mean_nb(body_size,window=10)
        
        # --- equivalente a  df_5min["high"].shift(1)
        high_prev = np.roll(high_arr, 1, axis=0)
        high_prev[0, :] = np.nan
        
        close_prev = np.roll(close_arr, 1, axis=0)
        close_prev[0, :] = np.nan
        
        #  el ultimo high es una vela pequenha, se esta perdiendo fuerza
        weak_progress = (
            (high_arr > high_prev) &
            (body_size < body_size_avg_10)
        )
           
        # 
        range_hl = high_arr - low_arr
        close_pos = np.divide(
            close_arr - low_arr,
            range_hl,
            out=np.zeros_like(close_arr, dtype=float),
            where=range_hl > 0
        )

        weak_close = close_pos < 0.35
        
        #
        vwap_extension_ratio = np.divide(
            close_arr - vwap_arr,
            vwap_arr,
             out=np.zeros_like(close_arr, dtype=float),  # en posiciones inválidas se pone 0
            where=vwap_arr != 0             # solo divide donde vwap_arr != 0
        )
        vwap_extension = vwap_extension_ratio > 0.25
        
        #
        no_follow_through =  (
            (high_arr > high_prev) &
            (close_arr < close_prev) 
        )
        
        #
        vol_ratio = np.divide(
        volume_arr,
        sma_volume_20_5m_arr,
        out=np.zeros_like(volume_arr,  dtype=float),
        where=sma_volume_20_5m_arr > 0
        )

        climactic_volume = vol_ratio >= 2
        
        rvol_cond = rvol_arr  >= 3
        
        is_red = close_arr < open_arr
        is_above_vwap = high_arr > vwap_arr
        
        exhaustion_score = (
            climactic_volume.astype(np.int8) +
            weak_progress.astype(np.int8) +
            weak_close.astype(np.int8) +
            vwap_extension.astype(np.int8) +
            no_follow_through.astype(np.int8)
        )
        
        vol_cond  = volume_arr > 10_000
        
        red_exhaustion_signal = (exhaustion_score >= 3) & is_red & vol_cond & rvol_cond &is_above_vwap
        
        entries = np.zeros((n_bars, n_cols), dtype=bool)
        
        entries[:] = red_exhaustion_signal
        # filtrar las entradas
        entries = entries & time_mask[:, None]
        
        tp_price = open_arr * (1 - np.array([m['tp'] for m in col_meta]))
        sl_price = open_arr + np.array([m['sl'] for m in col_meta]) * atr_arr

        tp_stop = np.full_like(open_arr, np.nan)
        sl_stop = np.full_like(open_arr, np.nan)
        
        tp_stop[entries] = np.where(
            open_arr[entries] > 0,
            (open_arr[entries] - tp_price[entries]) / open_arr[entries],
            False
        )

        sl_stop[entries] = np.where(
            open_arr[entries] > 0,
            (sl_price[entries] - open_arr[entries]) / open_arr[entries],
            False
        )
        
        # Agrupar por día
        forced_exit = np.zeros_like(entries, dtype=bool)
        
        forced_exit = generate_signal_to_force_close_EOD(forced_exit, index_master)
    
        pf = vbt.Portfolio.from_signals(
            close=close_arr,
            high=high_arr,
            low=low_arr,
            entries=entries,
            exits=forced_exit,
            size=1,
            direction='shortonly',
            tp_stop=tp_stop,
            sl_stop=sl_stop,
            init_cash=0,
            freq='5min'
        )   
        
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
        trades['strategy'] = f'short_exhaustion_strategy_'+ trades['tp'].astype(str) + "_" + trades['sl'].astype(str)+ "_" + trades['gap'].astype(str)
        trades['entry_time'] = index_master[trades['Entry Timestamp'].values]
        trades['exit_time']  = index_master[trades['Exit Timestamp'].values]
        entry_idx = trades['Entry Timestamp'].values
        col_idx   = trades['Column'].values
        
        atr_entry = atr_arr[entry_idx, col_idx]
        trades['stop_loss_price'] = (trades['entry_price'] + trades['sl'] * atr_entry)
        trades['rvol_daily'] = rvol_arr[entry_idx, col_idx]
        trades['previous_day_close'] = prev_day_close_arr[entry_idx, col_idx]
        trades['volume'] = volume_arr[entry_idx, col_idx]
        
        return reduce_trades_columns(trades) 
    
    except Exception as e:
        print(" error found in   ---short_exhaustion_strategy--- ")
        print(e)
        return pd.DataFrame([])


def short_vwap_pop_strategy(f_dict):
    """
    Short pops to vwap after the price was bellow vwap at least 25% 
    1 take profit at 15%
    1 stop loss at 3.5 ATR
    params: f_dict: {ticker: df_5m, ...}
    """
    
    print("Starting short_vwap_pop backtest...")
    
    try:
        
        tp_list = [0.15, 0.10]       # TP relativos
        sl_list = [3.5, 3.5]         # SL en múltiplos de ATR
        gap_list = [0.5, 0.5]         # GAPs list 
        
        all_params = prepare_params_and_vectors_for_gappers(f_dict,gap_list, tp_list, sl_list)
        
        tp_sl_gap_pairs = all_params['tp_sl_gap_pairs']
        n_params = all_params['n_params']
        index_master = all_params['index_master']
        n_tickers = all_params['n_tickers']
        n_cols = all_params['n_cols']
        n_bars = all_params['n_bars']
        open_arr  = all_params['open_arr']
        high_arr  = all_params['high_arr']
        low_arr   = all_params['low_arr']
        close_arr = all_params['close_arr']
        atr_arr   = all_params['atr_arr']
        volume_arr = all_params['volume_arr']
        rvol_arr   = all_params['rvol_arr']
        prev_day_close_arr = all_params['prev_day_close_arr']
        sma_volume_20_5m_arr = all_params['sma_volume_20_5m_arr']
        vwap_arr = all_params['vwap_arr']
        col = all_params['col']
        col_meta = all_params['col_meta']
        time_mask = all_params['time_mask']
       
        # --------------------------------------------------
        # Gap % vs previous_day_close
        # --------------------------------------------------
        gap_vals = np.array([m["gap"] for m in col_meta])
        
        # boolean array de horario válido
        hours = np.array([t.hour for t in index_master])
        minutes = np.array([t.minute for t in index_master])
        time_mask = ((hours >= 4) & (hours < 15)) | ((hours == 15) & (minutes <= 30))
        window_low = 5
        window_close = 4
        
        n_bars, n_cols = open_arr.shape
        
        # ------------------------
        # IDENTIFY DAYS
        # ------------------------
        dates = np.array([t.date() for t in index_master])
        unique_dates = np.unique(dates)
        
        # ================================================
        
        # -----------------------
        # CONDITION 1: WAS ABOVE VWAP DURING DAY
        # -----------------------
        above_vwap = high_arr > vwap_arr
        was_above_vwap = np.zeros_like(high_arr, dtype=bool)

        for day in unique_dates:
            day_mask = (dates == day) & time_mask
            idxs = np.where(day_mask)[0]
            for col in range(n_cols):
                if np.any(above_vwap[idxs, col]):
                    was_above_vwap[idxs, col] = True

        # -----------------------
        # CONDITION 2: HIGH >= GAP % PREVIOUS DAY CLOSE
        # -----------------------
        high_cond = np.zeros_like(high_arr, dtype=bool)
        gap_vals = np.array([m["gap"] for m in col_meta])

        for day in unique_dates:
            day_mask = (dates == day) & time_mask
            idxs = np.where(day_mask)[0]
            for col in range(n_cols):
                if np.any(high_arr[idxs, col] >= prev_day_close_arr[idxs, col] * (1 + gap_vals[col])):
                    high_cond[idxs, col] = True

        # -----------------------
        # CONDITION 3: LOW last 4 bars >= 90% VWAP, Highs below VWAP
        # -----------------------
        low_cond = np.zeros_like(low_arr, dtype=bool)
        window = 4

        for day in unique_dates:
            day_mask = (dates == day) & time_mask
            idxs = np.where(day_mask)[0]
            if len(idxs) <= window:
                continue

            for col in range(n_cols):
                lows  = low_arr[idxs, col]
                highs = high_arr[idxs, col]
                vwap_day = vwap_arr[idxs, col]

                for i in range(window, len(idxs)):
                    # Barra actual: High > VWAP, Low < VWAP
                    entry_high = highs[i] > vwap_day[i]
                    entry_low  = lows[i] < vwap_day[i]

                    # 4 velas previas
                    prev_lows  = lows[i-window:i]
                    prev_highs = highs[i-window:i]

                    if prev_lows.size == 0 or prev_highs.size == 0:
                        continue

                    prev_highs_below_vwap = np.all(prev_highs < vwap_day[i])
                    lowest_low_ok = np.min(prev_lows) <= vwap_day[i] * 0.90

                    if entry_high and entry_low and prev_highs_below_vwap and lowest_low_ok:
                        low_cond[idxs[i], col] = True
            
        # ================================================

        # ------------------------
        # CONDITION 5: CURRENT BAR ENTRY (High>VWAP & Low<VWAP)
        # ------------------------
        entry_cond = (high_arr > vwap_arr) & (low_arr < vwap_arr)
        
        rvol_cond = rvol_arr  >= 2

        # ------------------------
        # COMBINE CONDITIONS
        # ------------------------
        entries = was_above_vwap & high_cond  & entry_cond  & rvol_cond & low_cond 
        entries = entries & time_mask[:, None]  # ensure only 4:00-16:00
        
        tp_price = open_arr * (1 - np.array([m['tp'] for m in col_meta]))
        sl_price = open_arr + np.array([m['sl'] for m in col_meta]) * atr_arr

        tp_stop = np.full_like(open_arr, np.nan)
        sl_stop = np.full_like(open_arr, np.nan)
        
        tp_stop[entries] = np.where(
            open_arr[entries] > 0,
            (open_arr[entries] - tp_price[entries]) / open_arr[entries],
            False
        )

        sl_stop[entries] = np.where(
            open_arr[entries] > 0,
            (sl_price[entries] - open_arr[entries]) / open_arr[entries],
            False
        )
        
        # Agrupar por día
        forced_exit = np.zeros_like(entries, dtype=bool)
        
        forced_exit = generate_signal_to_force_close_EOD(forced_exit, index_master)
    
        pf = vbt.Portfolio.from_signals(
            close=close_arr,
            high=high_arr,
            low=low_arr,
            entries=entries,
            exits=forced_exit,
            size=1,
            direction='shortonly',
            tp_stop=tp_stop,
            sl_stop=sl_stop,
            init_cash=0,
            freq='5min'
        )   
        
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
        trades['strategy'] = f'short_vwap_pop_strategy_'+ trades['tp'].astype(str) + "_" + trades['sl'].astype(str)+ "_" + trades['gap'].astype(str)
        trades['entry_time'] = index_master[trades['Entry Timestamp'].values]
        trades['exit_time']  = index_master[trades['Exit Timestamp'].values]
        entry_idx = trades['Entry Timestamp'].values
        col_idx   = trades['Column'].values
        
        atr_entry = atr_arr[entry_idx, col_idx]
        trades['stop_loss_price'] = (trades['entry_price'] + trades['sl'] * atr_entry)
        trades['rvol_daily'] = rvol_arr[entry_idx, col_idx]
        trades['previous_day_close'] = prev_day_close_arr[entry_idx, col_idx]
        trades['volume'] = volume_arr[entry_idx, col_idx]
        
        return reduce_trades_columns(trades) 
    
    except Exception as e:
        print(" error found in   ---short_vwap_pop_strategy--- ")
        print(e)
        return pd.DataFrame([])
 

def short_explosives_pops(f_dict):
    
    tp_list  = [0.15, 0.20, 0.25]
    sl_list  = [0.02, 0.02, 0.03] # porciento por encima de la vela de entrada
    gap_list = [0.40, 0.50, 0.60]
    
    
    try:

        all_params = prepare_params_and_vectors_for_gappers(f_dict, gap_list=gap_list, tp_list=tp_list, sl_list=sl_list)
        
        tp_sl_pairs = all_params['tp_sl_gap_pairs']
        n_params = all_params['n_params']
        index_master = all_params['index_master']
        n_tickers = all_params['n_tickers']
        n_cols = all_params['n_cols']
        n_bars = all_params['n_bars']
        open_arr  = all_params['open_arr']
        high_arr  = all_params['high_arr']
        low_arr   = all_params['low_arr']
        close_arr = all_params['close_arr']
        atr_arr   = all_params['atr_arr']
        volume_arr = all_params['volume_arr']
        rvol_arr   = all_params['rvol_arr']
        prev_day_close_arr = all_params['prev_day_close_arr']
        col = all_params['col']
        col_meta = all_params['col_meta']
        time_mask = all_params['time_mask']
        
            
        # -----------------------
        # Gap % vs previous_day_close
        # -----------------------
        gap_vals = np.array([m["gap"] for m in col_meta])

        gap_pct = np.divide(
            close_arr - prev_day_close_arr,
            prev_day_close_arr,
            out=np.zeros_like(close_arr, dtype=float),
            where=prev_day_close_arr > 0
        )
        gap_cond = gap_pct >= gap_vals  # ahora cada columna compara su GAP respectivo
        
        # -----------------------
        # Máximo 3 velas desde primer toque
        # -----------------------
        first_touch = gap_cond & ~np.roll(gap_cond, 1, axis=0)
        entry_window = np.zeros_like(first_touch, dtype=bool)

        for col in range(n_cols):
            for i in range(2, n_bars):  # empezamos desde la tercera fila
                # Revisar secuencia de hasta 3 velas
                for k in range(3):  # k=0 -> current, k=1 -> prev, k=2 -> prev2
                    if i - k < 0:
                        continue

                    # Secuencia de tres velas
                    valid_seq = True
                    for j in range(k, 0, -1):
                        if open_arr[i-j, col] < close_arr[i-j-1, col]:
                            valid_seq = False
                            break

                    if valid_seq and first_touch[i, col]:
                        entry_window[i, col] = True
        
        # -----------------------
        # Evitar velas sin tail (full body)
        # -----------------------
        range_hl = high_arr - low_arr

        close_pos = np.divide(
            close_arr - low_arr,
            range_hl,
            out=np.zeros_like(close_arr, dtype=float),
            where=range_hl > 0
        )

        has_tail = close_pos < 0.85

        # -----------------------
        # Entradas
        # -----------------------
        entries = (
            entry_window &
            has_tail &
            time_mask[:, None] &
            ~np.isnan(open_arr)
        )
        
        tp_vals = np.array([m["tp"] for m in col_meta])
        sl_vals = np.array([m["sl"] for m in col_meta])
        
        tp_price = open_arr * (1 - tp_vals)
        sl_price = high_arr * (1 + sl_vals)

        tp_stop = (open_arr - tp_price) / open_arr
        sl_stop = (sl_price - open_arr) / open_arr
        
        # -----------------------
        # TP / SL
        # -----------------------
        tp_vals = np.array([m["tp"] for m in col_meta])
        sl_vals = np.array([m["sl"] for m in col_meta])

        tp_price = open_arr * (1 - tp_vals)
        sl_price = high_arr * (1 + sl_vals)

        tp_stop = (open_arr - tp_price) / open_arr
        sl_stop = (sl_price - open_arr) / open_arr

        # solo válidos en entradas
        tp_stop[~entries] = np.nan
        sl_stop[~entries] = np.nan
        
        # Agrupar por día
        forced_exit = np.zeros_like(entries, dtype=bool)
        forced_exit = generate_signal_to_force_close_EOD(forced_exit, index_master)

        pf = vbt.Portfolio.from_signals(
            close=close_arr,
            high=high_arr,
            low=low_arr,
            entries=entries,
            exits=forced_exit,
            size=1,
            direction='shortonly',
            tp_stop=tp_stop,
            sl_stop=sl_stop,
            init_cash=0,
            freq='5min'
        )   
        
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
        trades['strategy'] = f'short_explosives_pops_strategy_'+ trades['tp'].astype(str) + "_" + trades['sl'].astype(str)+ "_" + trades['gap'].astype(str)
        trades['entry_time'] = index_master[trades['Entry Timestamp'].values]
        trades['exit_time']  = index_master[trades['Exit Timestamp'].values]
        entry_idx = trades['Entry Timestamp'].values
        col_idx   = trades['Column'].values
        
        atr_entry = atr_arr[entry_idx, col_idx]
        trades['stop_loss_price'] = (trades['entry_price'] + trades['sl'] * atr_entry)
        trades['rvol_daily'] = rvol_arr[entry_idx, col_idx]
        trades['previous_day_close'] = prev_day_close_arr[entry_idx, col_idx]
        trades['volume'] = volume_arr[entry_idx, col_idx]
        
        return reduce_trades_columns(trades) 

    except Exception as e:
        print(" error found in   --- short_explosives_pops --- ")
        print(e)
        return pd.DataFrame([])
       

def backside_short(f_dict):
    """
    Short cuando el momentum comienza a fallar (backside)
    condiciones:
    - vela previa verde
    - vela actual roja (cuerpo mayor cuerpo de la vela previa)
    - lower low (cierre actual < mínimo previo) o upper tail grande (vela roja)
    - gap mínimo vs previous day close
    1 take profit at 15%
    1 stop loss at 3.5 ATR
    params: f_dict: {ticker: df_5m, ...}
    
    """
    
    try:
        # ==================================================
        # PARAMETROS
        # ==================================================
        tp_list  = [0.10, 0.10, 0.15, 0.15, 0.20, 0.20, 1.00, 1.00]        # TP relativos (short)
        sl_list  = [2.00, 3.50, 2.00, 3.50, 2.00, 3.50, 2.00, 3.50]          # SL en múltiplos de ATR
        gap_list = [0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50]          # Gap mínimo vs prev day close

        # ==================================================
        # PREPARAR VECTORES
        # ==================================================
        all_params = prepare_params_and_vectors_for_gappers(
            f_dict,
            gap_list,
            tp_list,
            sl_list
        )

        tp_sl_gap_pairs     = all_params['tp_sl_gap_pairs']
        n_params            = all_params['n_params']
        index_master        = all_params['index_master']
        n_tickers           = all_params['n_tickers']
        n_cols              = all_params['n_cols']
        n_bars              = all_params['n_bars']

        open_arr            = all_params['open_arr']
        high_arr            = all_params['high_arr']
        low_arr             = all_params['low_arr']
        close_arr           = all_params['close_arr']
        atr_arr             = all_params['atr_arr']
        volume_arr          = all_params['volume_arr']
        rvol_arr            = all_params['rvol_arr']
        prev_day_close_arr  = all_params['prev_day_close_arr']

        col_meta            = all_params['col_meta']
        vwap_arr  = all_params['vwap_arr']
        time_mask = all_params['time_mask']

        # ==================================================
        # INICIALIZAR ENTRADAS
        # ==================================================
        entries = np.zeros((n_bars, n_cols), dtype=bool)

        # ==================================================
        # VELAS (VECTORIAL)
        # ==================================================
        red_curr = close_arr < open_arr

        prev_open  = np.roll(open_arr, 1, axis=0)
        prev_close = np.roll(close_arr, 1, axis=0)
        prev_high  = np.roll(high_arr, 1, axis=0)
        prev_low   = np.roll(low_arr, 1, axis=0)

        green_prev = prev_close > prev_open
        
        # ================================================
        # VELA ACTUAL MAYOR QUE VELA PREVIA         
        # =================================================
        prev_open_2  = np.roll(open_arr, 2, axis=0)
        prev_close_2 = np.roll(close_arr, 2, axis=0)
        prev_high_2  = np.roll(high_arr, 2, axis=0)
        prev_low_2   = np.roll(low_arr, 2, axis=0)

        green_prev_2 = prev_close_2 > prev_open_2

        # ==================================================
        # LOWER LOW (definición exacta)
        # ==================================================
        lower_low_1 = (
            green_prev &
            red_curr &
            (close_arr < prev_low)
        )
        
        lower_low_2 = (
            green_prev_2 &
            red_curr &
            (close_arr < prev_low_2)
        )

        lower_low = lower_low_1 | lower_low_2
        
        # ==================================================
        # GAP % VS PREVIOUS DAY CLOSE (POR COLUMNA)
        # ==================================================
        gap_vals = np.array([m["gap"] for m in col_meta])

        gap_pct = np.divide(
            close_arr - prev_day_close_arr,
            prev_day_close_arr,
            out=np.zeros_like(close_arr, dtype=float),
            where=prev_day_close_arr > 0
        )

        gap_cond = (gap_pct >= gap_vals) & (gap_pct <= 5.0)
        
        gap_pct_with_high = np.divide(
            high_arr - prev_day_close_arr,
            prev_day_close_arr,
            out=np.zeros_like(high_arr, dtype=float),
            where=prev_day_close_arr > 0
        )
        
        gap_cond_for_tail = (gap_pct_with_high >= gap_vals) & (gap_pct_with_high <= 5.0)
        
        #print("Gap condition calculated")   
        #print(gap_cond)
        
      
        
        # ==================================================
        # UPPER TAIL GRANDE (vela roja)
        # ==================================================
        body = np.abs(close_arr - open_arr)
        upper_tail = high_arr - np.maximum(open_arr, close_arr)

        big_upper_tail = (
            red_curr &
            (upper_tail >= 5.0 * body) & 
            (prev_high < high_arr)
        ) 

        gap_filter = np.where(
            big_upper_tail,
            gap_cond_for_tail,   # cuando hay upper tail
            gap_cond             # caso contrario
        )
        
        # ==================================================
        # CONDICION: PRECIO > VWAP
        # ==================================================
        
        is_above_vwap = close_arr > vwap_arr
        
        #print("is_above_vwap condition calculated")   
        #print(is_above_vwap)

        # ==================================================
        # ENTRADAS SHORT
        # ==================================================
        entries = (
            gap_filter &
            is_above_vwap &
            time_mask[:, None] &
            (
                lower_low |
                big_upper_tail
            )
        )

        # ==================================================
        # TAKE PROFIT / STOP LOSS (POR COLUMNA)
        # ==================================================
        tp_vals = np.array([m["tp"] for m in col_meta])
        sl_vals = np.array([m["sl"] for m in col_meta])

        tp_price = open_arr * (1 - tp_vals)        # short
        sl_price = open_arr + sl_vals * atr_arr    # SL ATR

        tp_stop = np.full_like(open_arr, np.nan)
        sl_stop = np.full_like(open_arr, np.nan)

        tp_stop[:] = (open_arr - tp_price) / open_arr
        sl_stop[:] = (sl_price - open_arr) / open_arr

        # ==================================================
        # FORCED EXIT EOD
        # ==================================================
        forced_exit = np.zeros_like(entries, dtype=bool)
        forced_exit = generate_signal_to_force_close_EOD(
            forced_exit,
            index_master
        )

        # ==================================================
        # PORTFOLIO
        # ==================================================
        pf = vbt.Portfolio.from_signals(
            close=close_arr,
            high=high_arr,
            low=low_arr,
            entries=entries,
            exits=forced_exit,
            #price=open_arr,
            direction='shortonly',
            tp_stop=tp_stop,
            sl_stop=sl_stop,
            size=1,
            init_cash=0,
            freq='5min'
        )
        
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
        trades['strategy'] = f'backside_short_strategy_'+ trades['tp'].astype(str) + "_" + trades['sl'].astype(str)+ "_" + trades['gap'].astype(str)
        trades['entry_time'] = index_master[trades['Entry Timestamp'].values]
        trades['exit_time']  = index_master[trades['Exit Timestamp'].values]
        entry_idx = trades['Entry Timestamp'].values
        col_idx   = trades['Column'].values
        
        atr_entry = atr_arr[entry_idx, col_idx]
        trades['stop_loss_price'] = (trades['entry_price'] + trades['sl'] * atr_entry)
        trades['rvol_daily'] = rvol_arr[entry_idx, col_idx]
        trades['previous_day_close'] = prev_day_close_arr[entry_idx, col_idx]
        trades['volume'] = volume_arr[entry_idx, col_idx]
        
        return reduce_trades_columns(trades) 
        
    except Exception as e:
        print(" error found in   --- backside_short --- ")
        print(e)
        return pd.DataFrame([])
    

def backside_short_tp_dchain_stop(f_dict):
    """
    Short cuando el momentum comienza a fallar (backside)
    condiciones:
    - vela previa verde
    - vela actual roja (cuerpo mayor cuerpo de la vela previa)
    - lower low (cierre actual < mínimo previo) o upper tail grande (vela roja)
    - gap mínimo vs previous day close
    1 take profit at 15%
    1 stop loss at upper band of Donchian Channel 
    params: f_dict: {ticker: df_5m, ...}
    
    """
    
    try:
        # ==================================================
        # PARAMETROS
        # ==================================================
        tp_list  = [1, 1]        # TP relativos (short)
        gap_list = [0.5, 0.8]          # Gap mínimo vs prev day close

        # ==================================================
        # PREPARAR VECTORES
        # ==================================================
        all_params = prepare_params_and_vectors_for_gappers_with_trailing(
            f_dict,
            gap_list,
            tp_list
        )

        tp_sl_gap_pairs     = all_params['tp_sl_gap_pairs']
        n_params            = all_params['n_params']
        index_master        = all_params['index_master']
        n_tickers           = all_params['n_tickers']
        n_cols              = all_params['n_cols']
        n_bars              = all_params['n_bars']

        open_arr            = all_params['open_arr']
        high_arr            = all_params['high_arr']
        low_arr             = all_params['low_arr']
        close_arr           = all_params['close_arr']
        atr_arr             = all_params['atr_arr']
        volume_arr          = all_params['volume_arr']
        rvol_arr            = all_params['rvol_arr']
        prev_day_close_arr  = all_params['prev_day_close_arr']

        col_meta            = all_params['col_meta']
        vwap_arr  = all_params['vwap_arr']
        time_mask = all_params['time_mask']
        
        donchian_upper_arr =  all_params['donchian_upper_arr']
        donchian_basis_arr =  all_params['donchian_basis_arr']
        donchian_lower_arr = all_params['donchian_lower_arr']

        # ==================================================
        # INICIALIZAR ENTRADAS
        # ==================================================
        entries = np.zeros((n_bars, n_cols), dtype=bool)
        
        # ==================================================
        # DONCHIAN CHANNEL UPPER BAND       
        # ==================================================
        
                

        # ==================================================
        # VELAS (VECTORIAL)
        # ==================================================
        red_curr = close_arr < open_arr
        green_curr = close_arr > open_arr

        prev_open  = np.roll(open_arr, 1, axis=0)
        prev_close = np.roll(close_arr, 1, axis=0)
        prev_high  = np.roll(high_arr, 1, axis=0)
        prev_low   = np.roll(low_arr, 1, axis=0)

        green_prev = prev_close > prev_open
        
        # ================================================
        # VELA ACTUAL MAYOR QUE VELA PREVIA         
        # =================================================
        prev_open_2  = np.roll(open_arr, 2, axis=0)
        prev_close_2 = np.roll(close_arr, 2, axis=0)
        prev_high_2  = np.roll(high_arr, 2, axis=0)
        prev_low_2   = np.roll(low_arr, 2, axis=0)

        green_prev_2 = prev_close_2 > prev_open_2

        # ==================================================
        # LOWER LOW (definición exacta)
        # ==================================================
        lower_low_1 = (
            green_prev &
            red_curr &
            (close_arr < prev_low)
        )
        
        lower_low_2 = (
            green_prev_2 &
            red_curr &
            (close_arr < prev_low_2)
        )

        lower_low = lower_low_1 | lower_low_2
        
        # ==================================================
        # GAP % VS PREVIOUS DAY CLOSE (POR COLUMNA)
        # ==================================================
        gap_vals = np.array([m["gap"] for m in col_meta])

        gap_pct = np.divide(
            close_arr - prev_day_close_arr,
            prev_day_close_arr,
            out=np.zeros_like(close_arr, dtype=float),
            where=prev_day_close_arr > 0
        )

        gap_cond = (gap_pct >= gap_vals) & (gap_pct <= 5.0)
        
        gap_pct_with_high = np.divide(
            high_arr - prev_day_close_arr,
            prev_day_close_arr,
            out=np.zeros_like(high_arr, dtype=float),
            where=prev_day_close_arr > 0
        )
        
        gap_cond_for_tail = (gap_pct_with_high >= gap_vals) & (gap_pct_with_high <= 5.0)
        
        #print("Gap condition calculated")   
        #print(gap_cond)
        
      
        
        # ==================================================
        # UPPER TAIL GRANDE (vela roja)
        # ==================================================
        body = np.abs(close_arr - open_arr)
        upper_tail = high_arr - np.maximum(open_arr, close_arr)

        big_upper_tail = (
            red_curr &
            (upper_tail >= 5.0 * body) & 
            (prev_high < high_arr)
        ) 

        gap_filter = np.where(
            big_upper_tail,
            gap_cond_for_tail,   # cuando hay upper tail
            gap_cond             # caso contrario
        )
        
        # ==================================================
        # CONDICION: PRECIO > VWAP
        # ==================================================
        
        is_above_vwap = close_arr > vwap_arr
        
        #print("is_above_vwap condition calculated")   
        #print(is_above_vwap)

        # ==================================================
        # ENTRADAS SHORT
        # ==================================================
        entries = (
            gap_filter &
            is_above_vwap &
            time_mask[:, None] &
            (
                lower_low |
                big_upper_tail
            )
        )

        # ==================================================
        # TAKE PROFIT / STOP LOSS (POR COLUMNA)
        # ==================================================
        tp_vals = np.array([m["tp"] for m in col_meta])
        
        tp_price = open_arr * (1 - tp_vals)   # short
        tp_stop  = (open_arr - tp_price) / open_arr

        # ==================================================
        # DONCHIAN TRAILING STOP (SHORT)
        # vela verde + close > upper band
        # ==================================================
        donchian_exit = (
            green_curr &
            (close_arr > donchian_upper_arr)
        )

        # donchian_exit = np.roll(donchian_exit, 1, axis=0)
        # donchian_exit[0, :] = False
        
        exit_price = np.roll(open_arr, -1, axis=0)
        exit_price[-1, :] = close_arr[-1, :] 

        # ==================================================
        # FORCED EXIT EOD
        # ==================================================
        forced_exit = np.zeros_like(entries, dtype=bool)
        forced_exit = generate_signal_to_force_close_EOD(
            forced_exit,
            index_master
        )
        
        # ==================================================
        # EXITS FINALES
        # ==================================================
        exits = (
            donchian_exit |
            forced_exit
        )

        # ==================================================
        # PORTFOLIO
        # ==================================================
        pf = vbt.Portfolio.from_signals(
            close=close_arr,
            high=high_arr,
            low=low_arr,
            entries=entries,
            exits=exits,
            price=exit_price,
            direction='shortonly',
            tp_stop=tp_stop,
            size=1,
            init_cash=0,
            freq='5min'
        )
        
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
        trades['strategy'] = f'backside_short_tp_dchain_stop_'+ trades['tp'].astype(str) + "_"+ trades['gap'].astype(str)
        trades['entry_time'] = index_master[trades['Entry Timestamp'].values]
        trades['exit_time']  = index_master[trades['Exit Timestamp'].values]
        entry_idx = trades['Entry Timestamp'].values
        col_idx   = trades['Column'].values
        
        atr_entry = atr_arr[entry_idx, col_idx]
        trades['stop_loss_price'] = donchian_upper_arr[entry_idx, col_idx] 
        trades['rvol_daily'] = rvol_arr[entry_idx, col_idx]
        trades['previous_day_close'] = prev_day_close_arr[entry_idx, col_idx]
        trades['volume'] = volume_arr[entry_idx, col_idx]
        
        return reduce_trades_columns(trades)   
        
    except Exception as e:
        print(" error found in   --- backside_short --- ")
        print(e)
        return pd.DataFrame([])
   
# ======== long strategies ===========


# Notas!!!!!!!:   
#  * probar salidas en red candles con big upper tail
#  * evitar entradas despues de 5 velas verdes consecutivas (parabolic push)

def small_range_breakout_long_strategy(f_dict):
    """
    Long breakout de rango pequeño
    Timeframe: 5 minutos
    condiciones:
    - rompe upper band (donchain channel) de rango de 5 velas, offset 1 vela
    - volumen de la vela que rompe de al menos 10k acciones
    - volumen de la vela que rompe mayor a 2 veces el volumen promedio de 5 velas
    - take profit: No hay (usar trailing stop)
    - stop loss: Donchian lower band de rango de 5 velas, offset 1 vela
    params: f_dict: {ticker: df_5m, ...}
    """
    
    try:
        # ==================================================
        # PARAMETROS
        # ==================================================
        tp_list  = [9]        # TP relativos (short)
        gap_list = [0.1]          # Gap mínimo vs prev day close

        # ==================================================
        # PREPARAR VECTORES
        # ==================================================
        all_params = prepare_params_and_vectors_for_gappers_with_trailing(
            f_dict,
            gap_list,
            tp_list
        )

        tp_sl_gap_pairs     = all_params['tp_sl_gap_pairs']
        n_params            = all_params['n_params']
        index_master        = all_params['index_master']
        n_tickers           = all_params['n_tickers']
        n_cols              = all_params['n_cols']
        n_bars              = all_params['n_bars']

        open_arr            = all_params['open_arr']
        high_arr            = all_params['high_arr']
        low_arr             = all_params['low_arr']
        close_arr           = all_params['close_arr']
        atr_arr             = all_params['atr_arr']
        volume_arr          = all_params['volume_arr']
        rvol_arr            = all_params['rvol_arr']
        prev_day_close_arr  = all_params['prev_day_close_arr']

        col_meta            = all_params['col_meta']
        vwap_arr  = all_params['vwap_arr']
        time_mask = all_params['time_mask']
        
        donchian_upper_arr =  all_params['donchian_upper_arr']
        donchian_basis_arr =  all_params['donchian_basis_arr']
        donchian_lower_arr = all_params['donchian_lower_arr']

        # ==================================================
        # INICIALIZAR ENTRADAS
        # ==================================================
        entries = np.zeros((n_bars, n_cols), dtype=bool)
        
        # ==================================================
        # VELAS (VECTORIAL)
        # ==================================================
        red_curr = close_arr < open_arr
        green_curr = close_arr > open_arr

        # ==================================================
        # GAP % VS PREVIOUS DAY CLOSE (POR COLUMNA)
        # ==================================================
        gap_vals = np.array([m["gap"] for m in col_meta])

        gap_pct = np.divide(
            close_arr - prev_day_close_arr,
            prev_day_close_arr,
            out=np.zeros_like(close_arr, dtype=float),
            where=prev_day_close_arr > 0
        )

        gap_cond = (gap_pct >= gap_vals) & (gap_pct <= 5.0)
        
        # ==================================================
        # CONDICION: VOLUMEN
        # ==================================================
        
        vol_1 = np.roll(volume_arr, 1, axis=0)
        vol_2 = np.roll(volume_arr, 2, axis=0)
        vol_3 = np.roll(volume_arr, 3, axis=0)
        vol_4 = np.roll(volume_arr, 4, axis=0)
        vol_5 = np.roll(volume_arr, 5, axis=0)

        avg_vol_5 = (vol_1 + vol_2 + vol_3 + vol_4 + vol_5) / 5.0

        # evitar basura en las primeras filas
        avg_vol_5[:5, :] = np.nan
        
        volume_cond = (volume_arr >= 10000) & (volume_arr >= 2 * avg_vol_5) 
        
        # ==================================================
        # CONDICION: BREAKOUT DONCHIAN UPPER BAND
        # ==================================================
        breakout_cond =  close_arr > donchian_upper_arr
        
        # ==================================================
        # UPPER TAIL GRANDE 
        # ==================================================
        body = np.abs(close_arr - open_arr)
        upper_tail = high_arr - np.maximum(open_arr, close_arr)
        
        not_big_upper_tail = (upper_tail <= 1.4 * body) 
        
        not_parabolic_push =  ((body/open_arr) < 0.20) & (green_curr) # evitar entrar en grandes spikes (velas verdes parabolicas, +30% en una sola vela)
         
     
        # ==================================================
        # ENTRADAS SHORT
        # ==================================================
        entries = (
            gap_cond &
            green_curr &
            not_big_upper_tail &
            not_parabolic_push &
            breakout_cond &
            volume_cond &
            time_mask[:, None] 
        )

        # ==================================================
        # TAKE PROFIT / STOP LOSS (POR COLUMNA)
        # ==================================================
        tp_vals = np.array([m["tp"] for m in col_meta])
        
        tp_price = open_arr * (1 + tp_vals)   # short
        tp_stop  = (tp_price - open_arr) / open_arr

        # ==================================================
        # DONCHIAN TRAILING STOP (SHORT)
        # vela verde + close > upper band
        # ==================================================
        donchian_exit = (
            red_curr &
            (close_arr < donchian_lower_arr)
        )

        # donchian_exit = np.roll(donchian_exit, 1, axis=0)
        # donchian_exit[0, :] = False
        
        exit_price = np.roll(close_arr, -1, axis=0)
        exit_price[-1, :] = close_arr[-1, :] 
        

        # ==================================================
        # FORCED EXIT EOD
        # ==================================================
        forced_exit = np.zeros_like(entries, dtype=bool)
        forced_exit = generate_signal_to_force_close_EOD(
            forced_exit,
            index_master
        )
        
        # ==================================================
        # EXTRA EXIT FILTER: RED CANDLES WITH BIG UPPER TAIL
        
        # red candles with big upper tail for exit filter
        prev_high  = np.roll(high_arr, 1, axis=0)
        big_upper_tail = (
            red_curr &
            (upper_tail >= 3.0 * body) & 
            (prev_high < high_arr)
        ) 
        
        # ==================================================
        # EXITS FINALES
        # ==================================================
        
        
        exits = (
            donchian_exit |
            #big_upper_tail |
            forced_exit
        )

        # ==================================================
        # PORTFOLIO
        # ==================================================
        pf = vbt.Portfolio.from_signals(
            close=close_arr,
            high=high_arr,
            low=low_arr,
            entries=entries,
            exits=exits,
            price=exit_price,
            direction='longonly',
            tp_stop=tp_stop,
            size=1,
            init_cash=0,
            freq='5min'
        )
        
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
        trades['strategy'] = f'small_range_breakout_long_strategy_'+ trades['tp'].astype(str) + "_"+ trades['gap'].astype(str)
        trades['entry_time'] = index_master[trades['Entry Timestamp'].values]
        trades['exit_time']  = index_master[trades['Exit Timestamp'].values]
        entry_idx = trades['Entry Timestamp'].values
        col_idx   = trades['Column'].values
        
        atr_entry = atr_arr[entry_idx, col_idx]
        trades['stop_loss_price'] = donchian_lower_arr[entry_idx, col_idx] 
        trades['rvol_daily'] = rvol_arr[entry_idx, col_idx]
        trades['previous_day_close'] = prev_day_close_arr[entry_idx, col_idx]
        trades['volume'] = volume_arr[entry_idx, col_idx]
        
        
        return reduce_trades_columns(trades)   
        
    except Exception as e:
        print(" error found in   --- small_range_breakout_long_strategy  --- ")
        print(e)
        return pd.DataFrame([])
 
def small_range_breakout_long_strategy_with_tp_factor(f_dict):
    """
    Long breakout de rango pequeño
    Timeframe: 5 minutos
    condiciones:
    - rompe upper band (donchain channel) de rango de 5 velas, offset 1 vela
    - volumen de la vela que rompe de al menos 10k acciones
    - volumen de la vela que rompe mayor a 2 veces el volumen promedio de 5 velas
    - take profit: RR base, ex: 2:1, 3:1, 4:1
    - stop loss: Donchian lower band de rango de 5 velas, offset 1 vela
    params: f_dict: {ticker: df_5m, ...}
    """
    
    try:
        # ==================================================
        # PARAMETROS
        # ==================================================
        tp_list  = [1, 2, 3, 4, 5, 6, 8, 10]        # R-multiple
        gap_list = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]      # Gap mínimo

        # ==================================================
        # PREPARAR VECTORES
        # ==================================================
        all_params = prepare_params_and_vectors_for_gappers_with_trailing(
            f_dict,
            gap_list,
            tp_list
        )

        tp_sl_gap_pairs     = all_params['tp_sl_gap_pairs']
        n_params            = all_params['n_params']
        index_master        = all_params['index_master']
        n_tickers           = all_params['n_tickers']
        n_cols              = all_params['n_cols']
        n_bars              = all_params['n_bars']

        open_arr            = all_params['open_arr']
        high_arr            = all_params['high_arr']
        low_arr             = all_params['low_arr']
        close_arr           = all_params['close_arr']
        atr_arr             = all_params['atr_arr']
        volume_arr          = all_params['volume_arr']
        rvol_arr            = all_params['rvol_arr']
        prev_day_close_arr  = all_params['prev_day_close_arr']

        col_meta            = all_params['col_meta']
        vwap_arr  = all_params['vwap_arr']
        time_mask = all_params['time_mask']
        
        donchian_upper_arr =  all_params['donchian_upper_arr']
        donchian_basis_arr =  all_params['donchian_basis_arr']
        donchian_lower_arr = all_params['donchian_lower_arr']

        # ==================================================
        # VELAS
        # ==================================================
        green_curr = close_arr > open_arr

        # ==================================================
        # GAP %
        # ==================================================
        gap_vals = np.array([m["gap"] for m in col_meta])

        gap_pct = np.divide(
            close_arr - prev_day_close_arr,
            prev_day_close_arr,
            out=np.zeros_like(close_arr, dtype=float),
            where=prev_day_close_arr > 0
        )

        gap_cond = (gap_pct >= gap_vals) & (gap_pct <= 5.0)

        # ==================================================
        # VOLUMEN
        # ==================================================
        vol_1 = np.roll(volume_arr, 1, axis=0)
        vol_2 = np.roll(volume_arr, 2, axis=0)
        vol_3 = np.roll(volume_arr, 3, axis=0)
        vol_4 = np.roll(volume_arr, 4, axis=0)
        vol_5 = np.roll(volume_arr, 5, axis=0)

        avg_vol_5 = (vol_1 + vol_2 + vol_3 + vol_4 + vol_5) / 5.0
        avg_vol_5[:5, :] = np.nan

        volume_cond = (volume_arr >= 10_000) & (volume_arr >= 2 * avg_vol_5)

        # ==================================================
        # FILTROS DE VELA
        # ==================================================
        body = np.abs(close_arr - open_arr)
        upper_tail = high_arr - np.maximum(open_arr, close_arr)

        not_parabolic_push = (body / open_arr) < 0.20

        # ==================================================
        # BREAKOUT DONCHIAN
        # ==================================================
        breakout_cond = close_arr > donchian_upper_arr

        # ==================================================
        # ENTRADAS LONG
        # ==================================================
        entries = (
            gap_cond &
            green_curr &
            not_parabolic_push &
            volume_cond &
            breakout_cond &
            time_mask[:, None]
        )

        # ==================================================
        # PRECIO DE EJECUCION (OPEN SIGUIENTE VELA)
        # ==================================================
        entry_price = np.roll(open_arr, -1, axis=0)
        entry_price[-1, :] = np.nan

        entry_low = np.roll(low_arr, -1, axis=0)
        #entry_low[-1, :] = np.nan
        
        entry_low = low_arr.copy()

        # ==================================================
        # RIESGO
        # ==================================================
        risk = entry_price - entry_low
        risk = np.where(risk > 0, risk, np.nan)

        # ==================================================
        # TP / SL (SOLO DONDE HAY ENTRADA)
        # ==================================================
        tp_factor = np.array([m["tp"] for m in col_meta])

        tp_price = entry_price + tp_factor * risk

        sl_stop = np.full_like(entry_price, np.nan, dtype=float)
        tp_stop = np.full_like(entry_price, np.nan, dtype=float)

        valid_entries = entries & np.isfinite(risk)

        sl_stop[valid_entries] = (entry_price[valid_entries] - entry_low[valid_entries]) / entry_price[valid_entries]
        tp_stop[valid_entries] = (tp_price[valid_entries] - entry_price[valid_entries]) / entry_price[valid_entries]

        sl_stop = np.where(sl_stop > 0, sl_stop, np.nan)
        tp_stop = np.where(tp_stop > 0, tp_stop, np.nan)

        # ==================================================
        # FORCED EXIT EOD
        # ==================================================
        forced_exit = np.zeros_like(entries, dtype=bool)
        forced_exit = generate_signal_to_force_close_EOD(
            forced_exit,
            index_master
        )

        exits = forced_exit

        # ==================================================
        # PORTFOLIO
        # ==================================================
        pf = vbt.Portfolio.from_signals(
            close=close_arr,
            high=high_arr,
            low=low_arr,
            entries=entries,
            exits=exits,
            price=entry_price,
            direction='longonly',
            sl_stop=sl_stop,
            tp_stop=tp_stop,
            size=1,
            init_cash=0,
            freq='5min'
        )

        
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
        trades['strategy'] = f'small_range_breakout_long_strategy_with_tp_factor_'+ trades['tp'].astype(str) + "_" + trades['gap'].astype(str)
        trades['entry_time'] = index_master[trades['Entry Timestamp'].values]
        trades['exit_time']  = index_master[trades['Exit Timestamp'].values]
        entry_idx = trades['Entry Timestamp'].values
        col_idx   = trades['Column'].values
        
        atr_entry = atr_arr[entry_idx, col_idx]
        lows_stop = low_arr[entry_idx, col_idx]
        trades['stop_loss_price'] = lows_stop
        trades['rvol_daily'] = rvol_arr[entry_idx, col_idx]
        trades['previous_day_close'] = prev_day_close_arr[entry_idx, col_idx]
        trades['volume'] = volume_arr[entry_idx, col_idx]
        
        return  reduce_trades_columns(trades)    
        
    except Exception as e:
        print(" error found in   --- small_range_breakout_long_strategy_with_tp_factor --- ")
        print(e)
        return pd.DataFrame([])
    
# ========  examples with data =========   

def running_examples():
    
        

    #df = pd.read_parquet('small_caps_strategies/datasets/gappers_backtest_dataset_5min.parquet')
    #gap_crap_strategy_pipeline(df)


    # df['date'] = pd.to_datetime(df['date'])
    # df = df.set_index('date')
    # df1 = df[df['ticker']=='HIVE'] 
    # df2 = df[df['ticker']=='AVDL'] 

    # df_ditc = {
    #     'HIVE': df1,
    #     'AVDL': df2,
    # }

    #trades  = gap_crap_strategy(df_ditc)

    #print(trades.columns)
    # print(trades[[ 'ticker', 'entry_price',
    #        'exit_price','stop_loss_price',  'pnl',
    #        'Return', 'rvol_daily',  'previous_day_close','volume','entry_time','exit_time','strategy']])



    # date_str =  "2023-10-11"
    # previous_day_close = 3.12
    # ticker='TPST'

    # date_str =  "2024-12-19"
    # previous_day_close = 6.4
    # ticker='NVNI'


    date_str0 =  "2026-01-05"
    previous_day_close_0 = 5.82
    ticker='MNTS'

    (df0,_) = commons.prepare_data_parabolic_short_5min_v1(ticker, date_str0)
    df0['ticker'] = ticker
    df0['previous_day_close'] = previous_day_close_0
    df_day_0 = df0[df0["date"].dt.date == pd.to_datetime(date_str0).date()]
    df_day_0['date'] = pd.to_datetime(df_day_0['date'])
    #df_day_0 = df_day_0.set_index('date')


    date_str1 =  "2026-01-06"
    previous_day_close_1 = 8.74
    ticker='MNTS'

    (df,_) = commons.prepare_data_parabolic_short_5min_v1(ticker, date_str1)
    df['ticker'] = ticker
    df['previous_day_close'] = previous_day_close_1
    df_day = df[df["date"].dt.date == pd.to_datetime(date_str1).date()]
    df_day['date'] = pd.to_datetime(df_day['date'])
    #df_day = df_day.set_index('date')

    date_str2 =  "2026-01-07"
    previous_day_close_2 = 9.76
    ticker='MNTS'

    (df1,_) = commons.prepare_data_parabolic_short_5min_v1(ticker, date_str2)
    df1['ticker'] = ticker
    df1['previous_day_close'] = previous_day_close_2
    df_day_1 = df1[df1["date"].dt.date == pd.to_datetime(date_str2).date()]
    df_day_1['date'] = pd.to_datetime(df_day_1['date'])
    #df_day_1 = df_day_1.set_index('date')

    all_days = pd.concat([df_day_0, df_day, df_day_1], ignore_index=True)
    all_days_x = all_days.copy()
    all_days = all_days.set_index('date')

    #all_days = all_days[all_days['date_str'] == date_str1]
    #all_days_x = all_days[all_days['date_str'] == date_str1]
    print(all_days[['ticker','date_str']])
    f_dict ={ticker:all_days}
    trades = extension_backside_short_strategy(f_dict)
    print(trades)
    markers = commons.create_marker_from_signals_from_trades(trades)
    commons.plot_trades(all_days_x, markers )


# hace llamadas a polygon api para obtener datos
def exemple_with_api_data():
    
    date_str0 =  "2026-01-08"
    previous_day_close_0 = 3.36
    ticker='FLYX'
    
    # date_str0 =  "2025-12-10"
    # previous_day_close_0 = 0.79
    # ticker='BEAT'
    
    # date_str0 =  "2022-05-04"
    # previous_day_close_0 = 0.95
    # ticker='CETXP'
    
    date_str0 =  "2026-02-02"
    previous_day_close_0 = 1.56
    ticker='FUSE'
    
    date_str0 =  "2026-02-02"
    previous_day_close_0 = 0.40
    ticker='DKI'
    
    date_str0 =  "2026-02-02"
    previous_day_close_0 = 1.8
    ticker='CISS'
    
    date_str0 =  "2026-01-30"
    previous_day_close_0 = 0.23
    ticker='FAT'
    
    date_str0 =  "2026-02-02"
    previous_day_close_0 = 1.06
    ticker='DOGZ'
    
    date_str0 =  "2026-02-02"
    previous_day_close_0 = 1.18
    ticker='MRNO'
    
    # ejemplo con big tail roja 6:50am para usar como condicion de salida
    date_str0 =  "2026-01-28"
    previous_day_close_0 = 1.18
    ticker='MRNO'
    
    date_str0 =  "2022-07-01"
    previous_day_close_0 = 1.19
    ticker='ADXN'
    
    date_str0 =  "2021-12-22"
    previous_day_close_0 = 1.27
    ticker='ALJJ'

    (df0,_) = commons.prepare_data_parabolic_short_5min_v1(ticker, date_str0)
    df0['ticker'] = ticker
    df0['previous_day_close'] = previous_day_close_0
    df0 =  utils_helpers.donchainChannel(df0, lookback=5, offset=1)
    df_day_0 = df0[df0["date"].dt.date == pd.to_datetime(date_str0).date()]
    df_day_0['date'] = pd.to_datetime(df_day_0['date'])
    
    df1 =  df_day_0.copy()
    donchian_upper =  pd.DataFrame({
        'time': df1['date'],
        'donchian_upper': df1['donchian_upper']
    }).dropna()
    donchian_lower =  pd.DataFrame({
        'time': df1['date'],
        'donchian_lower': df1['donchian_lower']
    }).dropna()
    donchian_indicator =  ('donchian_upper', donchian_upper,'blue')
    donchian_indicator_lower =  ('donchian_lower', donchian_lower,'blue')
    indicators = [donchian_indicator, donchian_indicator_lower]
    df1 = df1.drop(columns=["time"])
    
    df_day_0 = df_day_0.set_index('date')
    #print(df_day_0.between_time("10:30","16:00"))
    f_dict ={ticker:df_day_0}
    
    # trades = gap_crap_strategy(f_dict)
    # trades = short_exhaustion_strategy(f_dict)
    # trades = short_vwap_pop_strategy(f_dict)
    # trades = short_explosives_pops(f_dict)
    # trades = backside_short(f_dict)
    # trades = backside_short_tp_dchain_stop(f_dict)
    # trades = small_range_breakout_long_strategy(f_dict)
    trades = small_range_breakout_long_strategy_with_tp_factor(f_dict)
    
    
    
    #save_trades(trades)
    print(trades)
    if len(trades) > 0:
        markers = commons.create_marker_from_signals_from_trades(trades)
        utils_helpers.plot_trades_indicators(df1[df1["date"].dt.date == pd.to_datetime(date_str0).date()], markers, indicators=indicators )
        
    return


# usa datos del parquet local
def exemple_with_local_data():
    
    
    date_str0 =  "2022-07-21"
    ticker =  'ADXN'
    
    date_str0 =  "2022-06-28"
    ticker =  'AGRX'
    
    # date_str0 =  "2025-12-31"
    # ticker =  'INBS'
    
    # date_str0 =  "2026-01-05"
    # ticker =  'INBS'
    
    df0  = pd.read_parquet('backtest_dataset/in_sample/gappers_backtest_dataset_5min_in_sample.parquet')
    df0 = df0.dropna(
        subset=["donchian_upper", "donchian_lower", "donchian_basis"]
    )
    
    # df0["date"].dt.date == pd.to_datetime(date_str0).date()
    df_day_0 = df0[(df0['ticker'] == ticker) & (df0["date"].dt.date == pd.to_datetime(date_str0).date()) ]
    df_day_0['date'] = pd.to_datetime(df_day_0['date'])
    df1 =  df_day_0.copy()
    donchian_upper =  pd.DataFrame({
        'time': df1['date'],
        'donchian_upper': df1['donchian_upper']
    }).dropna()
    donchian_lower =  pd.DataFrame({
        'time': df1['date'],
        'donchian_lower': df1['donchian_lower']
    }).dropna()
    donchian_indicator =  ('donchian_upper', donchian_upper,'blue')
    donchian_indicator_lower =  ('donchian_lower', donchian_lower,'blue')
    indicators = [donchian_indicator, donchian_indicator_lower]
    df1 = df1.drop(columns=["time"])
    
    df_day_0 = df_day_0.set_index('date')
   
    df =  df_day_0[['ticker','date_str','open','high','low','close', 'volume', 'donchian_upper', 'donchian_lower', 'donchian_basis', 'previous_day_close']]
    print(df)
    #print(df_day_0.between_time("9:30","16:00"))
    f_dict ={ticker:df_day_0}
    
    
    # trades = gap_crap_strategy(f_dict)
    # trades = short_exhaustion_strategy(f_dict)
    # trades = short_vwap_pop_strategy(f_dict)
    # trades = short_explosives_pops(f_dict)
    # trades = backside_short(f_dict)
    # trades = backside_short_tp_dchain_stop(f_dict)
    # trades = small_range_breakout_long_strategy(f_dict)
    trades = small_range_breakout_long_strategy_with_tp_factor(f_dict)
    
    # save_trades(trades)
    print(trades[['ticker', 'type','entry_price',
    'exit_price','stop_loss_price',  'pnl',
    'Return','entry_time','exit_time','strategy']])
    markers = commons.create_marker_from_signals_from_trades(trades)
    utils_helpers.plot_trades_indicators(df1,markers, indicators= indicators )
    
    return 

def check_db():
    df = pd.read_parquet('small_caps_strategies/datasets/gappers_backtest_dataset_5min.parquet')
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    
    # gappers 50%
    df_50  = df.copy()
    df_50['gap_prct'] =  (df_50['open'] - df_50['previous_day_close'])/df_50['previous_day_close']
    
    mask_930 = (df.index.hour == 9) & (df.index.minute == 30)

    df_50.loc[~mask_930, 'gap_prct'] = np.nan
    
    filtered = df_50[np.isnan(df_50['gap_prct']) == False ]
    
    filtered = filtered[(filtered['gap_prct']>= 5)  & (filtered['volume']> 20000) & (filtered['RVOL_daily']>= 0.5)  ]  #  & (filtered['gap_prct'] <= 5)
    
    print(filtered[['ticker','open', 'previous_day_close', 'gap_prct','date_str',"volume", "cummulative_vol","RVOL_daily"]])
    

# ============ stats =============

def run_stats_on_trades(sub_path='in_sample'):
    
    # =========== gap and craps trades ============
    #trades = pd.read_parquet(f'vectorbt_trades/{sub_path}/gap_and_crap_strategy_0.15_3.5_0.5.parquet')
    #trades = pd.read_parquet(f'vectorbt_trades/{sub_path}/gap_and_crap_strategy_0.2_3.5_0.5.parquet')
    #trades = pd.read_parquet(f'vectorbt_trades/{sub_path}/gap_and_crap_strategy_1.0_3.5_0.5.parquet')
    #trades = pd.read_parquet(f'vectorbt_trades/{sub_path}/gap_and_crap_strategy_0.2_3.5_0.8.parquet')
    #trades = pd.read_parquet(f'vectorbt_trades/{sub_path}/gap_and_crap_strategy_1.0_3.5_0.8.parquet')
    
    
    # =========== exhaustion trades ============
    # trades = pd.read_parquet(f'vectorbt_trades/{sub_path}/short_exhaustion_strategy_0.15_3.5_0.8.parquet')
    # trades = pd.read_parquet(f'vectorbt_trades/{sub_path}/short_exhaustion_strategy_0.15_3.5_0.5.parquet')
    # trades = pd.read_parquet(f'vectorbt_trades/{sub_path}/short_exhaustion_strategy_20.0_3.5_1.0.parquet')
    
    # =========== backside trades ============
    #trades = pd.read_parquet(f'vectorbt_trades/{sub_path}/short_backside_strategy_0.15_3.5_0.5.parquet')
    #trades = pd.read_parquet(f'vectorbt_trades/{sub_path}/short_backside_strategy_0.2_3.5_0.5.parquet')
    #trades = pd.read_parquet(f'vectorbt_trades/{sub_path}/short_backside_strategy_1.0_3.5_0.5.parquet')
    #trades = pd.read_parquet(f'vectorbt_trades/{sub_path}/short_backside_strategy_0.2_3.5_0.8.parquet')
    #trades = pd.read_parquet(f'vectorbt_trades/{sub_path}/short_backside_strategy_1.0_3.5_0.8.parquet')
    
    # =========== backside trailing trades ============
    #trades = pd.read_parquet(f'vectorbt_trades/{sub_path}/short_backside_trailing_strategy_1_0.5.parquet')
    #trades = pd.read_parquet(f'vectorbt_trades/{sub_path}/short_backside_trailing_strategy_1_0.8.parquet')
    
    # =========== small range break long trailing trades ============
    #trades = pd.read_parquet(f'vectorbt_trades/{sub_path}/small_range_breakout_long_strategy_10_0.1.parquet')
    trades = pd.read_parquet(f'vectorbt_trades/{sub_path}/small_range_breakout_long_strategy_9_0.1.parquet')
    
    
    
    
    #print(trades.columns)
    trades['is_profit'] = trades['pnl'] > 0
    trades['gap_prct'] = (trades['entry_price'] - trades['previous_day_close']) / trades['previous_day_close']
    
    #trades = trades[(trades['gap_prct'] >= 0.5) & (trades['gap_prct'] <= 0.8) ]
    #print(trades)
    
    (_stats, df )= utils_helpers.stats(trades_df=trades)
    print("===================================")
    pprint(_stats)
   
    #print(trades)
  
def run_stats_on_first_stage_trades(path_to_trades="backtest_dataset", sample_type="in_sample", strategy_fn=backside_short_tp_dchain_stop):
    
    folder_path = Path(f'{path_to_trades}/{sample_type}/trades/{strategy_fn.__name__}')
    file_path = Path(f'{folder_path}/{strategy_fn.__name__}_{sample_type}_trades.parquet')
    
    if file_path.exists():
        trades = pd.read_parquet(file_path)
    else:
        print(f"File {file_path} does not exist.")
        return
    
    
    trades['is_profit'] = trades['pnl'] > 0
    trades['gap_prct'] = (trades['entry_price'] - trades['previous_day_close']) / trades['previous_day_close']
    
    groupedtrades = trades.groupby('strategy')
    
    trade_stats_list = []
    
    for strategy_name, group in groupedtrades:     
        (_stats, df )= utils_helpers.stats(trades_df=group)
        _stats['strategy'] = strategy_name
        trade_stats_list.append(_stats)
        #print(f"Stats for strategy: {strategy_name}")
        #print("===================================")
        #pprint(_stats)
        #print("\n")
        
    table = pd.DataFrame(trade_stats_list)
    table.to_parquet(f'{folder_path}/{strategy_fn.__name__}_{sample_type}_trade_stats.parquet', index=False)
    print(table)

def  run_stats_on_walk_fordward_trades(path_to_trades="backtest_dataset/walk_fordward", sample_type="in_sample", strategy_fn=backside_short_tp_dchain_stop):
    
    folder_path = Path(f'{path_to_trades}/trades/{strategy_fn.__name__}')
    
    all_trades = pd.DataFrame()
    
    for i in range(1,4):
        file_path = Path(f'{folder_path}/walk_fordward_{sample_type}_{i}_trades.parquet')
        
        if file_path.exists():
            trades = pd.read_parquet(file_path)
            all_trades = pd.concat([all_trades, trades], ignore_index=True)
        else:
            print(f"File {file_path} does not exist.")
            return
        
        all_trades['is_profit'] = all_trades['pnl'] > 0
        all_trades['gap_prct'] = (all_trades['entry_price'] - all_trades['previous_day_close']) / all_trades['previous_day_close']
        
        groupedtrades = all_trades.groupby('strategy')
        trade_stats_list = []
    
        for strategy_name, group in groupedtrades:     
            (_stats, df )= utils_helpers.stats(trades_df=group)
            _stats['strategy'] = strategy_name
            trade_stats_list.append(_stats)
        
        table = pd.DataFrame(trade_stats_list)
        table.to_parquet(f'{folder_path}/walk_fordward_{strategy_fn.__name__}_{sample_type}_{i}_trade_stats.parquet', index=False)   
        print(table)
    #print(trades)
    
# ============ first stage test =============
  
def run_first_stage_test(path="backtest_dataset",sample_type="in_sample", strategy_fn=backside_short_tp_dchain_stop, append_trades=True):
    
    file_path = Path(f'{path}/{sample_type}/gappers_backtest_dataset_5min_{sample_type}.parquet')
    
            
    if file_path.exists():
        df = pd.read_parquet(file_path)
    else:
        print(f"File {file_path} does not exist.")
        return
        
    
    if len(df) >0:
        folder_path = Path(f'{path}/{sample_type}/trades/{strategy_fn.__name__}')
        folder_path.mkdir(parents=True, exist_ok=True)

        df['date'] = pd.to_datetime(df['date'])
        groups = df.groupby(['ticker','date_str'])

        print(f'Total of groups: {len(groups)}')

        counter = 0
        df_dict = {}
        index = 0
        total_trades = 0

        start_time = tm.perf_counter()

        # -----------------------------
        # MAIN LOOP
        # -----------------------------
        for (ticker,date_str), group in groups:
            
            group = group.set_index('date')
            len_group = len(group)
            

            if counter >= 100_000:
                index += 1
                print(
                    f'Processing backtest for {len(df_dict)} tickers '
                    f'at iteration {index}...'
                )

                trades = strategy_fn(df_dict)
                save_trades_to_file(trades,file_path=f'{folder_path}/{strategy_fn.__name__}_{sample_type}_trades.parquet', append=append_trades) 
                total_trades += len(trades)
                print(f'Trades generated in iteration {index}: {len(trades)}')

                counter = 0
                df_dict = {}

            if len_group > 50:
                counter += len_group
                if ticker in df_dict:
                    gp = df_dict[ticker]
                    new_group = pd.concat([gp, group], ignore_index=False)
                    new_group.sort_index()
                    df_dict[ticker] = new_group
                    
                else:
                    df_dict[ticker] = group
                    
                    

        # -----------------------------
        # FINAL FLUSH
        # -----------------------------
        if index == 0 and counter > 0 and counter <= 100_000:
            trades = strategy_fn(df_dict)
            #print(trades)
            save_trades_to_file(trades,file_path=f'{folder_path}/{strategy_fn.__name__}_{sample_type}_trades.parquet', append=append_trades)
            total_trades += len(trades)
            print(f'Trades generated in iteration {index}: {len(trades)}')

        end_time = tm.perf_counter()

        print(
            f"⏰ Tiempo total {sample_type} ({strategy_fn.__name__}): "
            f"{end_time - start_time:.2f}s | "
            f"Total trades: {total_trades}"
        )

        print(f'Finalizing with {index} iterations')
    
# ============ walk fordward test =============

def run_walk_forward_test(path="backtest_dataset/walk_fordward", sample_type="in_sample",strategy_fn=backside_short_tp_dchain_stop, append_trades=True):
    
    folder_path = Path(f'{path}/trades/{strategy_fn.__name__}')
    folder_path.mkdir(parents=True, exist_ok=True)
   
    for i in range(1,4):
        
        file_path = Path(f'{path}/walk_fordward_{sample_type}_{i}.parquet')
                
        if file_path.exists():
            df = pd.read_parquet(file_path)
            if len(df) > 0:
                df['date'] = pd.to_datetime(df['date'])
                groups = df.groupby(['ticker','date_str'])

                print(f'Total of groups: {len(groups)}')

                counter = 0
                df_dict = {}
                index = 0
                total_trades = 0

                start_time = tm.perf_counter()

                # -----------------------------
                # MAIN LOOP
                # -----------------------------
                for (ticker,date_str), group in groups:
                    
                    group = group.set_index('date')
                    len_group = len(group)
                    

                    if counter >= 100_000:
                        index += 1
                        print(
                            f'Processing backtest for {len(df_dict)} tickers '
                            f'at iteration {index}...'
                        )
                        
                        
                        trades = strategy_fn(df_dict)
                        save_trades_to_file(trades,file_path=f'{folder_path}/walk_fordward_{sample_type}_{i}_trades.parquet', append=append_trades)
                        total_trades += len(trades)
                        print(f'Trades generated in iteration {index}: {len(trades)}')

                        counter = 0
                        df_dict = {}

                    if len_group > 50:
                        counter += len_group
                        if ticker in df_dict:
                            gp = df_dict[ticker]
                            new_group = pd.concat([gp, group], ignore_index=False)
                            new_group.sort_index()
                            df_dict[ticker] = new_group
                            
                        else:
                            df_dict[ticker] = group
                            
                        
                # -----------------------------
                # FINAL FLUSH
                # -----------------------------
                if index == 0 and counter > 0 and counter <= 100_000:
                    trades = strategy_fn(df_dict)
                    save_trades_to_file(trades, file_path=f'{folder_path}/walk_fordward_{sample_type}_{i}_trades.parquet', append=append_trades)
                    total_trades += len(trades)
                    print(trades)
                    print(f'Trades generated in iteration {index}: {len(trades)}')

                end_time = tm.perf_counter()

                print(
                    f"⏰ Tiempo total {sample_type} ({strategy_fn.__name__}): "
                    f"{end_time - start_time:.2f}s | "
                    f"Total trades: {total_trades}"
                )

                print(f'Finalizing {strategy_fn.__name__} with  {index} iterations')
        
        else:
            print(f"File {file_path} does not exist.")
    
    
    pass

# ============ full backtest =============
def run_full_backtest():
    
    # ============ first stage test =============
    
    run_first_stage_test(sample_type="in_sample", strategy_fn=backside_short_tp_dchain_stop)
    run_first_stage_test(sample_type="out_of_sample", strategy_fn=backside_short_tp_dchain_stop)

    run_first_stage_test(sample_type="in_sample", strategy_fn=gap_crap_strategy)
    run_first_stage_test(sample_type="out_of_sample", strategy_fn=gap_crap_strategy)

    run_first_stage_test(sample_type="in_sample", strategy_fn=backside_short)
    run_first_stage_test(sample_type="out_of_sample", strategy_fn=backside_short)

    run_first_stage_test(sample_type="in_sample", strategy_fn=short_vwap_pop_strategy)
    run_first_stage_test(sample_type="out_of_sample", strategy_fn=short_vwap_pop_strategy)

    run_first_stage_test(sample_type="in_sample", strategy_fn=short_explosives_pops)
    run_first_stage_test(sample_type="out_of_sample", strategy_fn=short_explosives_pops)
    
    # ====== longs =====
    run_first_stage_test(sample_type="in_sample", strategy_fn=small_range_breakout_long_strategy)
    run_first_stage_test(sample_type="out_of_sample", strategy_fn=small_range_breakout_long_strategy)
    
    run_first_stage_test(sample_type="in_sample", strategy_fn=small_range_breakout_long_strategy_with_tp_factor)
    run_first_stage_test(sample_type="out_of_sample", strategy_fn=small_range_breakout_long_strategy_with_tp_factor)
    
    # ============ walk fordward test =============

    run_walk_forward_test(sample_type="in_sample", strategy_fn=backside_short_tp_dchain_stop)
    run_walk_forward_test(sample_type="out_sample", strategy_fn=backside_short_tp_dchain_stop)

    run_walk_forward_test(sample_type="in_sample", strategy_fn=gap_crap_strategy)
    run_walk_forward_test(sample_type="out_sample", strategy_fn=gap_crap_strategy)

    run_walk_forward_test(sample_type="in_sample", strategy_fn=backside_short)
    run_walk_forward_test(sample_type="out_sample", strategy_fn=backside_short)

    run_walk_forward_test(sample_type="in_sample", strategy_fn=short_vwap_pop_strategy)
    run_walk_forward_test(sample_type="out_sample", strategy_fn=short_vwap_pop_strategy)

    run_walk_forward_test(sample_type="in_sample", strategy_fn=short_explosives_pops)
    run_walk_forward_test(sample_type="out_sample", strategy_fn=short_explosives_pops)
    
    # ====== longs =====
    run_walk_forward_test(sample_type="in_sample", strategy_fn=small_range_breakout_long_strategy)
    run_walk_forward_test(sample_type="out_sample", strategy_fn=small_range_breakout_long_strategy)
    
    run_walk_forward_test(sample_type="in_sample", strategy_fn=small_range_breakout_long_strategy_with_tp_factor)
    run_walk_forward_test(sample_type="out_sample", strategy_fn=small_range_breakout_long_strategy_with_tp_factor)
    
    
#check_db()
#exemple_with_api_data()
#exemple_with_local_data()
#running_examples()
#run_backest()
#run_stats_on_trades()
#run_stats_on_trades(sub_path='out_of_sample')

#run_full_backtest()

#run_walk_forward_test(strategy_fn=small_range_breakout_long_strategy)
#run_walk_forward_test(strategy_fn=small_range_breakout_long_strategy, sample_type='out_of_sample')

#run_first_stage_test(sample_type="in_sample", strategy_fn=small_range_breakout_long_strategy_with_tp_factor)
#run_first_stage_test(sample_type="out_of_sample", strategy_fn=small_range_breakout_long_strategy_with_tp_factor)
#
#run_walk_forward_test(sample_type="in_sample", strategy_fn=small_range_breakout_long_strategy_with_tp_factor)
#run_walk_forward_test(sample_type="out_sample", strategy_fn=small_range_breakout_long_strategy_with_tp_factor)


# run_first_stage_test(strategy_fn=backside_short)
# run_first_stage_test(strategy_fn=backside_short, sample_type='out_of_sample')

# run_walk_forward_test(strategy_fn=backside_short)
# run_walk_forward_test(strategy_fn=backside_short, sample_type='out_sample')

# run_stats_on_first_stage_trades(strategy_fn=backside_short)
# run_stats_on_first_stage_trades(strategy_fn=backside_short, sample_type='out_of_sample')

# run_stats_on_walk_fordward_trades(strategy_fn=backside_short)
# run_stats_on_walk_fordward_trades(strategy_fn=backside_short, sample_type='out_sample')

# in_sample = pd.read_parquet("backtest_dataset/walk_fordward/trades/small_range_breakout_long_strategy_with_tp_factor/walk_fordward_in_sample_1_trades.parquet")
# in_sample['is_profit'] = in_sample['pnl'] >0
# trades1=  in_sample[(in_sample['strategy'] == 'small_range_breakout_long_strategy_with_tp_factor_1_0.1') & (in_sample['rvol_daily'] > 20)]
# trades2=  in_sample[(in_sample['strategy'] == 'small_range_breakout_long_strategy_with_tp_factor_2_0.1') & (in_sample['rvol_daily'] > 20)]
# trades3=  in_sample[(in_sample['strategy'] == 'small_range_breakout_long_strategy_with_tp_factor_3_0.1') & (in_sample['rvol_daily'] > 20)]
# trades4=  in_sample[(in_sample['strategy'] == 'small_range_breakout_long_strategy_with_tp_factor_4_0.1') & (in_sample['rvol_daily'] > 20)]
# trades5=  in_sample[(in_sample['strategy'] == 'small_range_breakout_long_strategy_with_tp_factor_5_0.1') & (in_sample['rvol_daily'] > 20)]
# trades6=  in_sample[(in_sample['strategy'] == 'small_range_breakout_long_strategy_with_tp_factor_6_0.1') & (in_sample['rvol_daily'] > 20)]
# trades7=  in_sample[(in_sample['strategy'] == 'small_range_breakout_long_strategy_with_tp_factor_7_0.1') & (in_sample['rvol_daily'] > 20)]
# trades8=  in_sample[(in_sample['strategy'] == 'small_range_breakout_long_strategy_with_tp_factor_8_0.1') & (in_sample['rvol_daily'] > 20)]


# groups = in_sample.groupby('strategy')
# for strategy, group in groups:
#     print(strategy)
   
   

# filter = ['ticker', 'type', 'stop_loss_price', 'pnl','is_profit', 'rvol_daily', 'previous_day_close', 'entry_time', 'volume', 'strategy']
# print("========= trades1 =======") 
# print(trades1[filter])

# (_stats, df )= utils_helpers.stats(trades_df=trades1)
# print("===================================")
# pprint(_stats)

# print("========= trades1 =======") 
# print(trades8[filter])

# (_stats, df )= utils_helpers.stats(trades_df=trades8)
# print("===================================")
# pprint(_stats)

# print(len(trades1))
# print(len(trades2))
# print(len(trades3))
# print(len(trades4))

# exemple_with_api_data()
# print("======== trades2 ========")
# print(trades2[filter])
# print("======== trades3 ========")
# print(trades3[filter])
# print("======== trades4 ========")
# print(trades4[filter])
# print("======== trades5 ========")
# print(trades5[filter])

trades_path = 'backtest_dataset/in_sample/trades/backside_short/backside_short_in_sample_trades.parquet'
data_path = 'backtest_dataset/in_sample/gappers_backtest_dataset_5min_in_sample.parquet'

trades=  pd.read_parquet(trades_path)
data =  pd.read_parquet(data_path)
res = tme.get_mae_mfe(trades, data)

print(res)


 

   









