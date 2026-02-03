import vectorbt as vbt
import pandas as pd
import numpy as np
from utils import utils, helpers, trade_metrics as tm
from pprint import pprint
from small_caps_strategies import commons



qqq =  pd.read_parquet('./nasdaq/QQQ_v1.parquet')
qqq.rename(columns={'o': 'open', 'c': 'close', 'h': 'high', 'l': 'low', 'v': 'volume', 't': 'time', 'vw':'vwap'}, inplace=True)

qqq['date'] = pd.to_datetime(qqq['date']) 
qqq['day'] = pd.to_datetime(qqq['date']) 
qqq = qqq.set_index('date')
qqq = qqq.drop(columns=['time'])


backtest_dataset = qqq.loc["2021-01-01":"2022-12-31"]
OFSample_dataset = qqq.loc["2023-01-01":"2024-12-31"]
OFSample_dataset = qqq.loc["2026-01-01":"2026-01-14"]
dataset = qqq.loc["2025-01-01":"2025-12-31"]


tqqq =  pd.read_parquet('./nasdaq/TQQQ_v1.parquet')
tqqq.rename(columns={'o': 'open', 'c': 'close', 'h': 'high', 'l': 'low', 'v': 'volume', 't': 'time', 'vw':'vwap'}, inplace=True)

tqqq['date'] = pd.to_datetime(tqqq['date']) 
tqqq['day'] = pd.to_datetime(tqqq['date']) 
tqqq = tqqq.set_index('date')
tqqq = tqqq.drop(columns=['time'])



backtest_dataset_tqqq = tqqq.loc["2021-01-01":"2022-12-31"]
OFSample_dataset_tqqq = tqqq.loc["2023-01-01":"2024-12-31"]
OFSample_dataset_tqqq = tqqq.loc["2026-01-01":"2026-01-14"]
dataset_tqqq = tqqq.loc["2025-01-01":"2025-12-31"]


tqqq1 =  pd.read_parquet('./nasdaq/TQQQ_v1.parquet')
tqqq1.rename(columns={'o': 'open', 'c': 'close', 'h': 'high', 'l': 'low', 'v': 'volume', 't': 'time', 'vw':'vwap'}, inplace=True)
tqqq1['date'] = pd.to_datetime(tqqq1['date']) 

#tqqq_test = tqqq1.loc["2024-05-20":"2024-05-21"]
tqqq_test = tqqq1[(tqqq1['date'].dt.date >= pd.to_datetime('2024-05-20').date()) & (tqqq1['date'].dt.date <= pd.to_datetime('2024-05-21').date()) ]
ff = tqqq_test.copy()
ff = ff.set_index('date')



print(tqqq_test)
#print(backtest_dataset)
#print(backtest_dataset_tqqq)



def qqq_IB(f_dict, strategy_name_prefix = ""):
   

    IB_START = "09:30"
    IB_END = "10:25"
    FIRST_ENTRY = "10:30"
    LAST_ENTRY = "13:30"
    FORCE_EXIT = "15:30"
    IB_MIN_PCT = 0.0055
    
    
    tp_list = [0.6]  # ejemplo
    sl_list = [0.4]  # ejemplo

    assert len(tp_list) == len(sl_list), "tp_list y sl_list deben tener el mismo largo"
    param_pairs = list(zip(sl_list, tp_list))
    n_params = len(param_pairs)

    tickers = list(f_dict.keys())
    n_tickers = len(tickers)

    # --------------------------------------------------
    # Índice maestro
    # --------------------------------------------------
    index_master = pd.DatetimeIndex([])
    for df in f_dict.values():
        index_master = index_master.union(df.index)
    index_master = pd.to_datetime(index_master.sort_values())
    n_bars = len(index_master)
    n_cols = n_tickers * n_params

    # --------------------------------------------------
    # Arrays base
    # --------------------------------------------------
    open_arr  = np.full((n_bars, n_cols), np.nan)
    high_arr  = np.full_like(open_arr, np.nan)
    low_arr   = np.full_like(open_arr, np.nan)
    close_arr = np.full_like(open_arr, np.nan)

    ib_high_arr  = np.full_like(close_arr, np.nan)
    ib_low_arr   = np.full_like(close_arr, np.nan)
    ib_range_arr = np.full_like(close_arr, np.nan)
    valid_ib_arr = np.full_like(close_arr, False, dtype=bool)

    col_meta = []
    col = 0

    # --------------------------------------------------
    # Poblar arrays por ticker x parámetros
    # --------------------------------------------------
    for ticker in tickers:
        df = f_dict[ticker].copy()
        df = df.reindex(index_master)
        df['day'] = df.index.normalize()

        # IB diario
        ib = (
            df
            .between_time(IB_START, IB_END)
            .groupby('day')
            .agg(
                IB_high=('high', 'max'),
                IB_low=('low', 'min')
            )
        )
        ib['IB_range'] = ib['IB_high'] - ib['IB_low']
        ib['IB_pct'] = ib['IB_range'] / ib['IB_low']
        ib['valid'] = ib['IB_pct'] >= IB_MIN_PCT

        df = df.join(ib, on='day')
        idx = ~df['close'].isna()

        for sl_factor, tp_factor in param_pairs:
            open_arr[idx, col]   = df.loc[idx, 'open'].values
            high_arr[idx, col]   = df.loc[idx, 'high'].values
            low_arr[idx, col]    = df.loc[idx, 'low'].values
            close_arr[idx, col]  = df.loc[idx, 'close'].values

            ib_high_arr[idx, col]  = df.loc[idx, 'IB_high'].values
            ib_low_arr[idx, col]   = df.loc[idx, 'IB_low'].values
            ib_range_arr[idx, col] = df.loc[idx, 'IB_range'].values
            valid_ib_arr[idx, col] = df.loc[idx, 'valid'].fillna(False).values

            col_meta.append({
                'ticker': ticker,
                'sl_factor': sl_factor,
                'tp_factor': tp_factor,
                'column': col
            })
            col += 1

    # --------------------------------------------------
    # Ventana horaria
    # --------------------------------------------------
    time_idx = index_master.time
    entry_time_ok = (
        (time_idx >= pd.to_datetime(FIRST_ENTRY).time()) &
        (time_idx <= pd.to_datetime(LAST_ENTRY).time())
    )[:, None]

    force_exit = (time_idx == pd.to_datetime(FORCE_EXIT).time())[:, None]

    # --------------------------------------------------
    # Señales crudas
    # --------------------------------------------------
    long_raw  = valid_ib_arr & entry_time_ok & (close_arr > ib_high_arr)
    short_raw = valid_ib_arr & entry_time_ok & (close_arr < ib_low_arr)

    raw_entries = long_raw | short_raw

    # --------------------------------------------------
    # Una sola entrada por día
    # --------------------------------------------------
    days = index_master.normalize().values
    first_entry = np.zeros_like(raw_entries, dtype=bool)

    for c in range(n_cols):
        seen = {}
        for i in range(n_bars):
            if raw_entries[i, c]:
                d = days[i]
                if d not in seen:
                    first_entry[i, c] = True
                    seen[d] = True

    long_entries  = long_raw & first_entry
    short_entries = short_raw & first_entry

    # --------------------------------------------------
    # SL/TP dinámicos convertidos a porcentaje
    # --------------------------------------------------
    sl_factors = np.array([m['sl_factor'] for m in col_meta])
    tp_factors = np.array([m['tp_factor'] for m in col_meta])

    # ---------------- Long ----------------
    tp_price_long = ib_high_arr + ib_range_arr * tp_factors
    sl_price_long = ib_low_arr

    tp_stop_long = np.full_like(close_arr, np.nan)
    sl_stop_long = np.full_like(close_arr, np.nan)

    tp_stop_long[long_entries] = (tp_price_long[long_entries] - open_arr[long_entries]) / open_arr[long_entries]
    sl_stop_long[long_entries] = (open_arr[long_entries] - sl_price_long[long_entries]) / open_arr[long_entries]

    # ---------------- Short ----------------
    tp_price_short = ib_low_arr - ib_range_arr * tp_factors
    sl_price_short = ib_high_arr

    tp_stop_short = np.full_like(close_arr, np.nan)
    sl_stop_short = np.full_like(close_arr, np.nan)

    tp_stop_short[short_entries] = (open_arr[short_entries] - tp_price_short[short_entries]) / open_arr[short_entries]
    sl_stop_short[short_entries] = (sl_price_short[short_entries] - open_arr[short_entries]) / open_arr[short_entries]

    # --------------------------------------------------
    # Portfolio long
    # --------------------------------------------------
    pf_long = vbt.Portfolio.from_signals(
        close=close_arr,
        high=high_arr,
        low=low_arr,
        price=open_arr,
        entries=long_entries,
        exits=force_exit,
        direction='longonly',
        tp_stop=tp_stop_long,
        sl_stop=sl_stop_long,
        size=1,
        init_cash=0,
        freq='5min'
    )

    # --------------------------------------------------
    # Portfolio short
    # --------------------------------------------------
    pf_short = vbt.Portfolio.from_signals(
        close=close_arr,
        high=high_arr,
        low=low_arr,
        price=open_arr,
        entries=short_entries,
        exits=force_exit,
        direction='shortonly',
        tp_stop=tp_stop_short,
        sl_stop=sl_stop_short,
        size=1,
        init_cash=0,
        freq='5min'
    )
    
    
    trades_long = (
        pf_long.trades.records_readable
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    
        
    trades_long['entry_time'] = index_master[trades_long['Entry Timestamp'].values]
    trades_long['exit_time']  = index_master[trades_long['Exit Timestamp'].values]
    trades_long['day'] = trades_long['entry_time'].dt.normalize()
    entry_idx = trades_long['Entry Timestamp'].values
    col_idx   = trades_long['Column'].values
    trades_long['stop_loss_price'] = ib_low_arr[entry_idx, col_idx]
    
    trades_long = trades_long.rename(columns={
        'Avg Entry Price': 'entry_price',
        'Avg Exit Price': 'exit_price',
        'PnL': 'pnl',
        'Direction':'type',
    })
    
    meta_df = pd.DataFrame(col_meta)

    trades_long = trades_long.merge(
        meta_df,
        left_on='Column',
        right_on='column',
        how='left'
    )
    
    trades_short = (
        pf_short.trades.records_readable
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    trades_short['entry_time'] = index_master[trades_short['Entry Timestamp'].values]
    trades_short['exit_time']  = index_master[trades_short['Exit Timestamp'].values]
    trades_short['day'] = trades_short['entry_time'].dt.normalize()
    entry_idx = trades_short['Entry Timestamp'].values
    col_idx   = trades_short['Column'].values
    trades_short['stop_loss_price'] = ib_high_arr[entry_idx, col_idx]
    
    trades_short = trades_short.merge(
        meta_df,
        left_on='Column',
        right_on='column',
        how='left'
    )
     
    trades_short = trades_short.rename(columns={
        'Avg Entry Price': 'entry_price',
        'Avg Exit Price': 'exit_price',
        'PnL': 'pnl',
        'Direction':'type',
    })
    
    trades =  pd.concat([trades_long, trades_short], ignore_index=True)
    len_tp_list = len(tp_list)
    
    # obtener el index dentro de la lista de tp_list
    tp_idx = trades['Column'] % len_tp_list
    trades['tp_factor'] = np.array(tp_list)[tp_idx]
    trades['sl_factor'] = np.array(sl_list)[tp_idx]
    trades['strategy'] = f'{strategy_name_prefix}_' + trades['tp_factor'].astype(str)+"_" + trades['sl_factor'].astype(str)
    trades['is_profit'] = trades['pnl'] > 0
    
    return trades.groupby(['ticker', 'strategy'])
    
    
def qqq_orb_5min(f_dict, strategy_name_prefix=""):

    import numpy as np
    import pandas as pd
    import vectorbt as vbt

    ORB_TIME    = "09:30"
    FIRST_ENTRY = "09:30"
    LAST_ENTRY  = "13:30"
    FORCE_EXIT  = "15:30"
    
    tp_list = [8]

       # --------------------------------------------------
    # 1. Índice maestro
    # --------------------------------------------------
    index_master = pd.DatetimeIndex([])
    for df in f_dict.values():
        index_master = index_master.union(df.index)
    index_master = pd.to_datetime(index_master.sort_values())
    n_bars = len(index_master)

    # --------------------------------------------------
    # 2. Columnas (ticker × TP)
    # --------------------------------------------------
    tickers = list(f_dict.keys())
    n_tickers = len(tickers)
    n_tp = len(tp_list)
    n_cols = n_tickers * n_tp

    # --------------------------------------------------
    # 3. Arrays base
    # --------------------------------------------------
    open_arr  = np.full((n_bars, n_cols), np.nan)
    high_arr  = np.full_like(open_arr, np.nan)
    low_arr   = np.full_like(open_arr, np.nan)
    close_arr = np.full_like(open_arr, np.nan)

    orb_high_arr  = np.full_like(open_arr, np.nan)
    orb_low_arr   = np.full_like(open_arr, np.nan)
    orb_range_arr = np.full_like(open_arr, np.nan)
    orb_close_arr = np.full_like(open_arr, np.nan)

    orb_green_arr = np.zeros_like(open_arr, dtype=bool)
    orb_red_arr   = np.zeros_like(open_arr, dtype=bool)

    col_meta = []
    col = 0

    # --------------------------------------------------
    # 4. Poblar arrays
    # --------------------------------------------------
    for ticker in tickers:
        df = f_dict[ticker].copy()
        df = df.reindex(index_master)
        df['day'] = df.index.normalize()

        open_bar = df[df.index.time == pd.to_datetime(ORB_TIME).time()]
        orb = (
            open_bar
            .groupby('day')
            .agg(
                orb_high=('high', 'max'),
                orb_low=('low', 'min'),
                orb_open=('open', 'first'),
                orb_close=('close', 'first')
            )
        )
        orb['orb_range'] = orb['orb_high'] - orb['orb_low']
        orb['orb_green'] = orb['orb_close'] > orb['orb_open']
        orb['orb_red']   = orb['orb_close'] < orb['orb_open']

        df = df.join(orb, on='day')
        valid = ~df['close'].isna()

        for tp_factor in tp_list:
            open_arr[valid, col]  = df.loc[valid, 'open'].values
            high_arr[valid, col]  = df.loc[valid, 'high'].values
            low_arr[valid, col]   = df.loc[valid, 'low'].values
            close_arr[valid, col] = df.loc[valid, 'close'].values

            orb_high_arr[valid, col]  = df.loc[valid, 'orb_high'].values
            orb_low_arr[valid, col]   = df.loc[valid, 'orb_low'].values
            orb_range_arr[valid, col] = df.loc[valid, 'orb_range'].values
            orb_close_arr[valid, col] = df.loc[valid, 'orb_close'].values

            orb_green_arr[valid, col] = df.loc[valid, 'orb_green'].fillna(False).values
            orb_red_arr[valid, col]   = df.loc[valid, 'orb_red'].fillna(False).values

            col_meta.append({'ticker': ticker, 'tp_factor': tp_factor, 'column': col})
            col += 1

    # --------------------------------------------------
    # 5. Ventanas horarias
    # --------------------------------------------------
    t = index_master.time
    entry_time_ok = ((t >= pd.to_datetime(FIRST_ENTRY).time()) &
                     (t <= pd.to_datetime(LAST_ENTRY).time()))[:, None]
    forced_exit = (t == pd.to_datetime(FORCE_EXIT).time())[:, None]

    # --------------------------------------------------
    # 6. Señales ORB
    # --------------------------------------------------
    long_raw  = entry_time_ok & orb_green_arr & (close_arr > orb_high_arr)
    short_raw = entry_time_ok & orb_red_arr   & (close_arr < orb_low_arr)
    
    
    #print(orb)

    # --------------------------------------------------
    # 7. Una sola entrada por día
    # --------------------------------------------------
    days = index_master.normalize().values
    def first_trade_per_day(entries):
        out = np.zeros_like(entries, dtype=bool)
        for c in range(entries.shape[1]):
            seen = set()
            for i in range(entries.shape[0]):
                if entries[i, c]:
                    d = days[i]
                    if d not in seen:
                        out[i, c] = True
                        seen.add(d)
        return out

    long_entries  = first_trade_per_day(long_raw)
    short_entries = first_trade_per_day(short_raw)

    # --------------------------------------------------
    # 8. TP / SL convertidos a porcentaje para tp_stop / sl_stop
    # --------------------------------------------------
    tp_factors = np.array([m['tp_factor'] for m in col_meta])

    # ----- Long -----
    tp_price_long = orb_high_arr + orb_range_arr * tp_factors
    sl_price_long = orb_low_arr

    tp_stop_long = np.full_like(close_arr, np.nan)
    sl_stop_long = np.full_like(close_arr, np.nan)

    tp_stop_long[long_entries] = (tp_price_long[long_entries] - orb_high_arr[long_entries]) / orb_high_arr[long_entries]
    sl_stop_long[long_entries] = (orb_high_arr[long_entries] - sl_price_long[long_entries]) / orb_high_arr[long_entries]

    # ----- Short -----
    tp_price_short = orb_low_arr - orb_range_arr * tp_factors
    sl_price_short = orb_high_arr

    tp_stop_short = np.full_like(close_arr, np.nan)
    sl_stop_short = np.full_like(close_arr, np.nan)

    tp_stop_short[short_entries] = (orb_low_arr[short_entries] - tp_price_short[short_entries]) / orb_low_arr[short_entries]
    sl_stop_short[short_entries] = (sl_price_short[short_entries] - orb_low_arr[short_entries]) / orb_low_arr[short_entries]

    # --------------------------------------------------
    # 9. Portfolio Long Only
    # --------------------------------------------------
    pf_long = vbt.Portfolio.from_signals(
        close=close_arr,
        high=high_arr,
        low=low_arr,
        price=close_arr,
        entries=long_entries,
        exits=forced_exit,
        direction='longonly',
        tp_stop=tp_stop_long,
        sl_stop=sl_stop_long,
        size=1,
        init_cash=0,
        freq='5min'
    )

    # --------------------------------------------------
    # 10. Portfolio Short Only
    # --------------------------------------------------
    pf_short = vbt.Portfolio.from_signals(
        close=close_arr,
        high=high_arr,
        low=low_arr,
        price=close_arr,
        entries=short_entries,
        exits=forced_exit,
        direction='shortonly',
        tp_stop=tp_stop_short,
        sl_stop=sl_stop_short,
        size=1,
        init_cash=0,
        freq='5min'
    )
    
    trades_long = (
        pf_long.trades.records_readable
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    
    
    trades_long['entry_time'] = index_master[trades_long['Entry Timestamp'].values]
    trades_long['exit_time']  = index_master[trades_long['Exit Timestamp'].values]
    trades_long['day'] = trades_long['entry_time'].dt.normalize()
    entry_idx = trades_long['Entry Timestamp'].values
    col_idx   = trades_long['Column'].values
    trades_long['stop_loss_price'] = orb_low_arr[entry_idx, col_idx]
    
    

    trades_long = trades_long.rename(columns={
        'Avg Entry Price': 'entry_price',
        'Avg Exit Price': 'exit_price',
        'PnL': 'pnl',
        'Direction':'type',
    })
    
    meta_df = pd.DataFrame(col_meta)

    trades_long = trades_long.merge(
        meta_df,
        left_on='Column',
        right_on='column',
        how='left'
    )
    
    trades_short = (
        pf_short.trades.records_readable
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    trades_short['entry_time'] = index_master[trades_short['Entry Timestamp'].values]
    trades_short['exit_time']  = index_master[trades_short['Exit Timestamp'].values]
    trades_short['day'] = trades_short['entry_time'].dt.normalize()
    entry_idx = trades_short['Entry Timestamp'].values
    col_idx   = trades_short['Column'].values
    trades_short['stop_loss_price'] = orb_high_arr[entry_idx, col_idx]
    
    trades_short = trades_short.merge(
        meta_df,
        left_on='Column',
        right_on='column',
        how='left'
    )
     
    trades_short = trades_short.rename(columns={
        'Avg Entry Price': 'entry_price',
        'Avg Exit Price': 'exit_price',
        'PnL': 'pnl',
        'Direction':'type',
    })
    
    trades =  pd.concat([trades_long, trades_short], ignore_index=True)
    
    len_tp_list = len(tp_list)
    
    # obtener el index dentro de la lista de tp_list
    tp_idx = trades['Column'] % len_tp_list
    trades['tp_factor'] = np.array(tp_list)[tp_idx]
    trades['strategy'] = f'{strategy_name_prefix}_' + trades['tp_factor'].astype(str)+"x"
    trades['is_profit'] = trades['pnl'] > 0
    
    return trades.groupby(['ticker', 'strategy'])
    
    
# data1 = {'QQQ':backtest_dataset}
# trades = qqq_IB(data1, strategy_name_prefix="IB_breakout")
# print(trades)

data2 = {'QQQ':backtest_dataset, "TQQQ": backtest_dataset_tqqq}
data3 = {'QQQ':qqq, "TQQQ": tqqq}
data4 = {"TQQQ": ff}
#trades_grouped = qqq_IB(data2, strategy_name_prefix="IB_breakout")
#trades_grouped = qqq_IB(data3, strategy_name_prefix="IB_breakout_all_data")
#trades_grouped = qqq_orb_5min(data3, strategy_name_prefix="qqq_orb_5min_all_data")
trades_grouped = qqq_orb_5min(data4, strategy_name_prefix="qqq_orb_5min_test")
#print(trades)


def save_groups(trades_grouped):
    
    for (ticker, strategy), df_group in trades_grouped:
        print(ticker, strategy)
        trades = df_group[['entry_time','exit_time', "Column",'ticker', 'type','entry_price',
        'exit_price','stop_loss_price','pnl','is_profit','strategy']]
        helpers.save_trades(f'trades/FUTURES_ETF/{ticker}_{strategy}.parquet', trades)
        print(trades)
    
save_groups(trades_grouped)

risk_pct = 0.002

qqq_trades = helpers.load_trades('trades/FUTURES_ETF/TQQQ_qqq_orb_5min_test_8x.parquet')

markers = commons.create_marker_from_signals_from_trades(qqq_trades)
commons.plot_trades_v1(tqqq_test, markers )

# tm.analysis_and_plot_with_benchmark( 
#                                     qqq_trades,
#                                     initial_capital=100_000,
#                                     risk_pct=risk_pct , # 0.5% por R,
#                                     benchmark_ticker = "QQQ"
#                                     )



# (_stats, df )= helpers.stats(trades_df=trades)
# pprint(_stats)
# print(trades)




