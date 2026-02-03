import os
import sys
sys.path.insert(0, os.path.abspath("."))
import pandas as pd
from utils import utils, helpers
from pprint import pprint
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, time, date

print("====== mariel lang ideas =======")

def download_QQQ_data(file_path="."):
    
    today = datetime.today()
    start = datetime(today.year - 4, today.month, today.day).strftime('%Y-%m-%d')
    #_dates = utils.generate_date_interval_to_fetch(start)
    _dates = utils.month_ranges(start)
    parameters = []
    df = pd.DataFrame([])
    
    for item in _dates:
        (date1, date2) = item
        parameters.append(("QQQ", date1, date2))
        
    for item in parameters:
        (ticker, d1, d2) = item
        
        df = utils.fetch_ticker_data_5min(ticker, d1,d2)
        df = pd.DataFrame(df)
        print(df)
        helpers.append_single_parquet(df, f'{file_path}/QQQ.parquet')
    
  
def stats(trades_df):
    
    #print(trades_df)
    
    trades_df = trades_df.copy()

    #trades_df["risk"] = 0.4 * trades_df["range"]
    trades_df["risk"] = (trades_df["entry_price"] - trades_df["stop_loss_price"]).abs().round(3)
    trades_df["R"] = trades_df["pnl"] / trades_df["risk"]
    
    stats = {}
    stats["trades"] = len(trades_df)
    stats["wins"] = (trades_df["pnl"] > 0).sum()
    stats["losses"] = (trades_df["pnl"] < 0).sum()
    stats["win_rate"] = stats["wins"] / stats["trades"] if stats["trades"] > 0 else 0

    stats["avg_win"] = trades_df.loc[trades_df["pnl"] > 0, "pnl"].mean()
    stats["avg_loss"] = trades_df.loc[trades_df["pnl"] < 0, "pnl"].mean()

    stats["avg_R"] = trades_df["R"].mean()
    stats["total_R"] = trades_df["R"].sum()
    
    win_R = trades_df.loc[trades_df["R"] > 0, "R"].mean()
    loss_R = trades_df.loc[trades_df["R"] < 0, "R"].mean()

    expectancy_R = (
        stats["win_rate"] * win_R +
        (1 - stats["win_rate"]) * loss_R
    )
    
    trades_df["equity_R"] = trades_df["R"].cumsum()
    trades_df["equity_peak"] = trades_df["equity_R"].cummax()
    trades_df["drawdown_R"] = trades_df["equity_R"] - trades_df["equity_peak"]

    max_dd_R = trades_df["drawdown_R"].min()
    
    # exit_stats = (
    # trades_df
    # .groupby("exit_reason")
    # .agg(
    #     trades=("pnl", "count"),
    #     avg_R=("R", "mean"),
    #     total_R=("R", "sum")
    #     )
    # )
    # trades_df["month"] = trades_df["day"].dt.to_period("M")

    # monthly_stats = (
    #     trades_df
    #     .groupby("month")
    #     .agg(
    #         trades=("R", "count"),
    #         win_rate=("R", lambda x: (x > 0).mean()),
    #         total_R=("R", "sum"),
    #         avg_R=("R", "mean")
    #     )
    # )
    
    print("==== STRATEGY STATS ====")
    print(f"Trades        : {stats['trades']}")
    print(f"Win rate      : {stats['win_rate']:.2%}")
    print(f"Avg R / trade : {stats['avg_R']:.2f}")
    print(f"Expectancy R  : {expectancy_R:.2f}")
    print(f"Total R       : {stats['total_R']:.2f}")
    print(f"Max DD (R)    : {max_dd_R:.2f}")



    stats_df = pd.DataFrame(stats, index=["value"])

    #print(stats_df)    
    return  stats_df    

def qqq_IB_BO(file_path=None):
    
    if file_path is None:
        print('No data file path was passed')
        
    
    df = pd.read_parquet(file_path)
    df.rename(columns={'o': 'open', 'c': 'close', 'h': 'high', 'l': 'low', 'v': 'volume', 't': 'time'}, inplace=True)
    df["date"] =  pd.to_datetime(df["time"], unit='ms', utc=True) # pd.to_datetime(df["time"], unit='ms') - pd.Timedelta(hours=5) # -5 means New York timezone 
    df["date"] = df["date"].dt.tz_convert("America/New_York")
    df["day"] =  df["date"].dt.date #pd.to_datetime(pd.to_datetime(df["time"], unit='ms').dt.date).dt.strftime('%Y-%m-%d') 
    #df["time"] = df["date"].dt.time
    
    df = df.copy()

    # Ensure datetime types
    df["date"] = pd.to_datetime(df["date"])
    df["day"] = pd.to_datetime(df["day"])

    # Helper: convert time column to datetime.time
    df["t"] = df["date"].dt.time

    trades = []

    grouped_by_day = df.groupby("day")
    
   
    # -------------------------
    # Process day by day
    # -------------------------
    for day, d in grouped_by_day:

        
        d = d.reset_index(drop=True)
        trade_taken = False

        # -------------------------
        # Opening range 9:30–10:30
        # -------------------------
        or_df = d[(d["t"] >= time(9,30)) & (d["t"] <= time(10,30))]
        if or_df.empty:
            continue

        range_high = or_df["high"].max()
        range_low  = or_df["low"].min()
        range_val  = range_high - range_low

        # Volatility filter
        if range_val / range_low < 0.0055:
            continue

        # -------------------------
        # Scan bars for breakout signal
        # -------------------------
        for i in range(len(d) - 1):

            if trade_taken:
                break

            bar = d.iloc[i]
            next_bar = d.iloc[i + 1]

            # Entry time constraint
            if not (time(10,30) < bar["t"] <= time(13,30)):
                continue

            # Signal detection
            if bar["high"] > range_high:
                direction = "long"
            elif bar["low"] < range_low:
                direction = "short"
            else:
                continue

            # -------------------------
            # ENTRY (next bar open)
            # -------------------------
            entry_time = next_bar["date"]
            entry_price = next_bar["open"]

            # Targets & stops
            if direction == "long":
                target = entry_price + 0.6 * range_val
                stop   = entry_price - 0.4 * range_val
            else:
                target = entry_price - 0.6 * range_val
                stop   = entry_price + 0.4 * range_val

            trade_taken = True

            # -------------------------
            # Trade management
            # -------------------------
            exit_price = None
            exit_time = None
            exit_reason = None

            for j in range(i + 1, len(d) - 1):

                bar = d.iloc[j]
                next_bar = d.iloc[j + 1]

                if bar["t"] > time(15,30):
                    break

                hit = False

                if direction == "long":
                    if bar["low"] <= stop:
                        hit = True
                        exit_reason = "stop"
                    elif bar["high"] >= target:
                        hit = True
                        exit_reason = "target"
                else:
                    if bar["high"] >= stop:
                        hit = True
                        exit_reason = "stop"
                    elif bar["low"] <= target:
                        hit = True
                        exit_reason = "target"

                if hit:
                    exit_time = next_bar["date"]
                    exit_price = next_bar["open"]
                    break

            # -------------------------
            # Forced exit at 15:30
            # -------------------------
            if exit_price is None:
                last_idx = d[d["t"] <= time(15,30)].index.max()

                if last_idx is not None and last_idx < len(d) - 1:
                    exit_time = d.iloc[last_idx + 1]["date"]
                    exit_price = d.iloc[last_idx + 1]["open"]
                else:
                    exit_time = d.iloc[last_idx]["date"]
                    exit_price = d.iloc[last_idx]["close"]

                exit_reason = "time_exit"

            # -------------------------
            # PnL
            # -------------------------
            pnl = (
                exit_price - entry_price
                if direction == "long"
                else entry_price - exit_price
            )

            trades.append({
                "day": day,
                "type": direction,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "stop_loss_price": stop,
                "pnl": pnl,
                "is_profit":pnl > 0,
                "range": range_val,
                "exit_reason": exit_reason
            })


    # Resulting trades DataFrame
    trades_df = pd.DataFrame(trades)

   
    return trades_df
    

trades_df = qqq_IB_BO(file_path='./strategies_ideas/QQQ.parquet')
print(trades_df)
helpers.stats(trades_df)
helpers.stats_per_year(trades_df)