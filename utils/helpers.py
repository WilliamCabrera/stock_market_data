import os
import sys
sys.path.insert(0, os.path.abspath("."))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import utils
from lightweight_charts import Chart
from datetime import datetime, timedelta, time, date
import time as tm
import requests
from pprint import pprint

def vwap(df, time_col='date'):
    df = df.copy()

    # asegurarse que es datetime
    df[time_col] = pd.to_datetime(df[time_col])

    tp = (df["high"] + df["low"] + df["close"]) / 3

    df["vwap"] = (
        (tp * df["volume"])
        .groupby(df[time_col].dt.date)
        .cumsum()
        /
        df["volume"]
        .groupby(df[time_col].dt.date)
        .cumsum()
    )

    return df["vwap"]


def calculate_sma(df, period: int = 50):
    return pd.DataFrame({
        'time': df['date'],
        f'SMA {period}': df['close'].rolling(window=period).mean()
    }).dropna()
    
def compute_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr

def raw_data_to_dataframe(data):
    df = pd.DataFrame(data)

    try:
        df.rename(columns={'o': 'open', 'c': 'close', 'h': 'high', 'l': 'low', 'v': 'volume', 't': 'time'}, inplace=True)
        # Convert timestamp to datetime
        df["date"] = pd.to_datetime(df["time"], unit='ms', utc=True)  # -5 means New York timezone
        #df["date"] = df["date"].dt.tz_convert("America/New_York")
        df['date'] = df['date'] - pd.Timedelta(hours=5)
        df = df.drop(columns=['time'])
        
    except  Exception as e:
       print(f' error: {e}')
    
    return df

def donchainChannel(df_5min, lookback=5,  offset=1):

    df = df_5min.copy()

    # Donchian channel
    df['donchian_upper'] = (
        df['high']
        .rolling(window=lookback, min_periods=lookback)
        .max()
        .shift(offset)
    )

    df['donchian_lower'] = (
        df['low']
        .rolling(window=lookback, min_periods=lookback)
        .min()
        .shift(offset)
    )
    
     # Rellenar NaN iniciales con el low de la vela
    df['donchian_upper'] = df['donchian_upper'].fillna(df['low'])
    df['donchian_lower'] = df['donchian_lower'].fillna(df['low'])

    # Línea media (basis)
    df['donchian_basis'] = (
        (df['donchian_upper'] + df['donchian_lower']) / 2
    )
    
    return df

def get_data_for_backtest(ticker="IRBT", start_date="2025-12-01", end_date="2025-12-10", adjusted=True):
    
    df_1min = utils.fetch_ticker_data_1min(ticker, start_date, end_date, adjusted=adjusted)
    df_1min = raw_data_to_dataframe(df_1min)
    #df_5min = utils.fetch_ticker_data_5min(ticker, start_date, end_date)
    #df_5min = raw_data_to_dataframe(df_5min)
   
    return (df_1min, None)

def get_data_5min_for_backtest(ticker="IRBT", start_date="2025-12-01", end_date="2025-12-10"):
    
    #df_1min = utils.fetch_ticker_data_1min(ticker, start_date, end_date)
    #df_1min = raw_data_to_dataframe(df_1min)
    df_5min = utils.fetch_ticker_data_5min(ticker, start_date, end_date)
    df_5min = raw_data_to_dataframe(df_5min)
   
    return (df_5min, None)

def get_data_daily_for_backtest(ticker="IRBT", start_date="2025-12-01", end_date="2025-12-10"):
    
    #df_1min = utils.fetch_ticker_data_1min(ticker, start_date, end_date)
    #df_1min = raw_data_to_dataframe(df_1min)
    df_daily = utils.fetch_ticker_data_daily(ticker, start_date, end_date)
    df_daily = raw_data_to_dataframe(df_daily)
   
    return (df_daily, None)


def fetch_all_data_from_gappers(connectionParams):
    
    
    df = pd.DataFrame()
    if connectionParams != None:
            try:


                POSTGREST_H =  connectionParams['POSTGREST_H']  # os.getenv("POSTGREST_H", "none")
                POSTGREST_P =  connectionParams['POSTGREST_P']  # os.getenv("POSTGREST_P", "none")
                POSTGREST_TOKEN = connectionParams['POSTGREST_TOKEN']  # os.getenv("POSTGREST_TOKEN", "none")
                
                #connect_to_DB_1(user=DB_USER, password=DB_PASSWORD, host=POSTGREST_H, port=POSTGREST_P, database=DB_NAME)
                
                # Headers with Authorization
                headers = {
                    "Authorization": f"Bearer {POSTGREST_TOKEN}",
                    "Content-Type": "application/json",
                    'Accept':'application/json'
                }

                # The API endpoint
                url = f'http://{POSTGREST_H}:{POSTGREST_P}/gappers_pm_and_market_hours'

                response = requests.get(url)
              
                if response.status_code != 200:
                    print(f"error fetching data: get_latest_ticker_date: {response.status_code}")
                    return pd.DataFrame()
                
                data = response.json()
                df = pd.DataFrame(data)
                
               

            except Exception as e:
                print("Error:", e)

    return df

def stats_type_long(df_trades = pd.DataFrame([]), limit_price = 20, initial_equity=0, risk_pct=0):
   
    
    # the equity curve is express in R by default
    df = df_trades.copy() 
    df = df[df['entry_price'] < limit_price] 
    #df = df[0:50]
    df["pnl"] = df["exit_price"] - df["entry_price"]
    df['is_profit'] = df['pnl'] > 0
    df["risk"] = (df["entry_price"] - df["stop_loss_price"]).abs().round(3)
    df = df[df['risk'] > 0]
    
    df = df.sort_values("entry_time").reset_index(drop=True)
    df["duration_min"] = (
    df["exit_time"] - df["entry_time"]
    ).dt.total_seconds() / 60
    
    
    df["pnl_pct"] = (df["exit_price"] / df["entry_price"] - 1) * 100

    df["R"] = (df["pnl"] / df["risk"]).round(3)
    df["stop_distance_pct"] = ((
        (df["exit_price"] - df["stop_loss_price"])
        / df["entry_price"]
    ) * 100).round(2)
    

    df["equity_R"] = df["R"].cumsum().round(2)
    df["equity_peak"] = df["equity_R"].cummax()
    df["drawdown"] = df["equity_R"] - df["equity_peak"]
    df["drawdown_pct"] = (df["drawdown"] / df["equity_peak"])
    
    # ============= dollar equity curve =============
    
    if initial_equity > 0 and risk_pct > 0:
        dollar_equity = [initial_equity]

        for r in df["R"]:
            prev_equity = dollar_equity[-1]
            pnl = prev_equity * risk_pct * r
            dollar_equity.append(prev_equity + pnl)

        df["equity_$"] = dollar_equity[1:]
        
       
    # === expectancy =================

    # Winners / losers
    wins = df[df["R"] > 0]
    losses = df[df["R"] <= 0]

    win_rate = len(wins) / len(df)
    loss_rate = 1 - win_rate

    avg_win_R = wins["R"].mean() if len(wins) else 0
    avg_loss_R = losses["R"].mean() if len(losses) else 0
    
    # ======= drawdowns  =======
    max_dd = df["drawdown"].min()
    max_dd_pct = df["drawdown_pct"].min()
    underwater = df["drawdown"] < 0
    df['underwater'] = underwater

    # Mark each drawdown period
    dd_groups = (underwater != underwater.shift()).cumsum()
    
    #print(dd_groups[10:50])
    # Count trades in each drawdown period
    dd_durations = df[underwater].groupby(dd_groups).size()
 
    
    # Longest drawdown streak (in trades)
    max_dd_duration = dd_durations.max() if not dd_durations.empty else 0
    
    # ======= Expected Value ======
    expectancy = win_rate * avg_win_R + loss_rate * avg_loss_R

    summary = {
        "trades": len(df),
        "win_rate_%": round(win_rate * 100,2),
        "avg_win_R": round(avg_win_R,2),
        "avg_loss_R": round(abs(avg_loss_R),2),
        "expectancy_R": round(expectancy,2),
        "max_drawdown_R": round(abs(max_dd),2),
        "max_dd_trades": round(max_dd_duration,2),
        "max_dd_pct": round(max_dd_pct,2),
        "total_profit": round(wins["R"].sum(),2),
        "total_loss": round(abs(losses["R"].sum()),2),
        "net_profit": round((wins["R"].sum() - abs(losses["R"].sum())), 2),
        "profit_factor":  round(wins["R"].sum()/abs(losses["R"].sum()),2),
        "final_equity_$": df["equity_$"].iloc[-1] if initial_equity > 0 and risk_pct > 0 else np.nan,
        "initial_equity":initial_equity,
        "account_risk_per_trade_pct": risk_pct
         
    }
    
    stats = {
    k: v.item() if hasattr(v, "item") else v
    for k, v in summary.items()
    }
    
    return  (stats, df)

# create markers to show in the charts
def generate_markers_from_trade_list(trades=[]):
    
    markers =[]
    for trade in trades:
        entry_marker = {"time":  trade['entry_time'],
                        "position": "below",
                        "shape": "arrow_up" ,
                        "color": "yellow",
                        "text":""
                        }
        
        position =  'above'
        
        if trade['type'] == 'LONG':
            if trade['is_profit']:
                position =  'above'
            else: 
                position =  'below'
                
        if trade['type'] == 'SHORT':
            if trade['is_profit']:
                position =  'below'
            else: 
                position =  'above'
                
        
        exit_marker = {"time": trade['exit_time'],
                        "position": position,
                        "shape": "arrow_down" ,
                        "color": "red",
                        "text":""}
        markers.append(entry_marker)
        markers.append(exit_marker)
        
    
    return markers

def split_df_by_size(df, chunk_size=10):
    return [
        df.iloc[i:i + chunk_size]
        for i in range(0, len(df), chunk_size)
    ]

def append_single_parquet(df, path):
    if os.path.exists(path):
        old = pd.read_parquet(path)
        df = pd.concat([old, df], ignore_index=True)

    df.to_parquet(path)

def log(text, file_path="logs.csv"):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    df = pd.DataFrame([[now, text]], columns=["Date", "message"])

    df.to_csv(
        file_path,
        mode="a",                          # append
        header=not os.path.exists(file_path),
        index=False
    )

def stats(trades_df,  limit_price = 20):
    
    trades_df = trades_df.copy()
    
    col = trades_df.get("pnl")
    if col is None:
        trades_df['pnl'] = np.where(
            trades_df['type'].astype(str).str.lower() == 'long',
            trades_df['exit_price'] - trades_df['entry_price'],  # long
            trades_df['entry_price'] - trades_df['exit_price']   # short
        )
    
    trades_df = trades_df.sort_values(by=['entry_time']).reset_index(drop=True)
    trades_df.reset_index(drop=True, inplace=True)
    #trades_df = trades_df[trades_df['entry_price'] < limit_price] 
    valid_short_trades = (  trades_df['type'].astype(str).str.lower() == 'short') & (trades_df['entry_price'] < trades_df['stop_loss_price'])
    valid_long_trades = (  trades_df['type'].astype(str).str.lower() == 'long') & (trades_df['entry_price'] > trades_df['stop_loss_price'])
    
    trades_df =  trades_df[ valid_short_trades | valid_long_trades ]

    trades_df["risk"] = (trades_df["entry_price"] - trades_df["stop_loss_price"]).abs().round(3)
    trades_df["R"] = trades_df["pnl"] / trades_df["risk"]
   
    trades_df = trades_df[ trades_df["R"].notna() & ~np.isinf(trades_df["R"])]
    total_trades =  len(trades_df)
    
    
    stats = {}
    stats["trades"] = len(trades_df)
    
    wins = trades_df[trades_df["R"] > 0]
    losses = trades_df[trades_df["R"] <= 0]
    
    win_rate =  len(wins) / total_trades if total_trades > 0 else 0
    loss_rate = 1 - win_rate
    
    avg_win_R = wins["R"].mean() if len(wins) else 0
    avg_loss_R = losses["R"].mean() if len(losses) else 0
    
    avg_win = trades_df.loc[trades_df["pnl"] > 0, "pnl"].mean()
   
    win_R = trades_df.loc[trades_df["R"] > 0, "R"].mean()
    loss_R = trades_df.loc[trades_df["R"] < 0, "R"].mean()
    
    expectancy_R = (
        win_rate * win_R +
        (1 - win_rate) * loss_R
    )
    
    # trades_df["equity_R"] = trades_df["R"].cumsum()
    # trades_df["equity_peak"] = trades_df["equity_R"].cummax()
    # trades_df["drawdown_R"] = trades_df["equity_R"] - trades_df["equity_peak"]
    # trades_df["drawdown"] = trades_df["equity_R"] - trades_df["equity_peak"]
    # trades_df["drawdown_pct"] = (trades_df["drawdown_R"] / trades_df["equity_peak"])
    
    
    
    # --- Drawdown en unidades de R (NO monetario) ---

    # 1) Equity en R
    trades_df["equity_R"] = trades_df["R"].cumsum()

    # 2) Peak de R acumulado
    trades_df["equity_R_peak"] = trades_df["equity_R"].cummax()

    # 3) Drawdown en R (siempre <= 0)
    trades_df["drawdown_R"] = trades_df["equity_R"] - trades_df["equity_R_peak"]

    # 4) Blindaje numérico
    trades_df["drawdown_R"] = trades_df["drawdown_R"].clip(upper=0)
    
    
    max_dd_R = trades_df["drawdown_R"].min()
    
    # ======= drawdowns monentario  =======
    # max_dd = trades_df["drawdown"].min()
    # max_dd_pct = trades_df["drawdown_pct"].min()
    # underwater = trades_df["drawdown"] < 0
    # trades_df['underwater'] = underwater
    
    # ======= drawdowns expresado en unidades de riesgo (R) =======
    max_dd = trades_df["drawdown_R"].min()
    underwater = trades_df["drawdown_R"] < 0
    trades_df['underwater'] = underwater

    # Mark each drawdown period
    dd_groups = (underwater != underwater.shift()).cumsum()
    
    #print(dd_groups[10:50])
    # Count trades in each drawdown period
    dd_durations = trades_df[underwater].groupby(dd_groups).size()
 
    
    # Longest drawdown streak (in trades)
    max_dd_duration = dd_durations.max() if not dd_durations.empty else 0
    
    avg_win_R = wins["R"].mean() if len(wins) else 0
    avg_loss_R = losses["R"].mean() if len(losses) else 0
    
    # ======= wins/losses streaks =======
    s = trades_df['is_profit']

    # Identifica cambios True ↔ False
    groups = (s != s.shift()).cumsum()
    
    streaks = s.groupby(groups).agg(
    value='first',
    length='size'
    )
    
    max_positive_streak = streaks.loc[streaks['value'] == True, 'length'].max()
    max_negative_streak = streaks.loc[streaks['value'] == False, 'length'].max()

    #==== STRATEGY STATS ====
  
    
    summary = {
        "trades": total_trades,
        "win_rate_%": round(win_rate * 100,2),
        "avg_win_R": round(avg_win_R,2),
        "avg_loss_R": round(abs(avg_loss_R),2),
        "expectancy_R": round(expectancy_R,2),
        "max_drawdown_R": round(abs(max_dd),2),
        "max_dd_trades": round(max_dd_duration,2),
        #"max_dd_pct": round(max_dd_pct,2),
        "total_profit": round(wins["R"].sum(),2),
        "total_loss": round(abs(losses["R"].sum()),2),
        "net_profit": round((wins["R"].sum() - abs(losses["R"].sum())), 2),
        "profit_factor":  round(wins["R"].sum()/abs(losses["R"].sum()),2),
        "max_positive_streak": max_positive_streak,
        "max_negative_streak": max_negative_streak,
        #"final_equity_$": df["equity_$"].iloc[-1] if initial_equity > 0 and risk_pct > 0 else np.nan,
        #"initial_equity":initial_equity,
        #"account_risk_per_trade_pct": risk_pct
         
    }
    
    stats = {
    k: v.item() if hasattr(v, "item") else v
    for k, v in summary.items()
    }


    #print(stats_df)    
    return  (stats, trades_df)    

def stats_per_year(trades_df):
    
    trades_df['year'] =  pd.to_datetime(trades_df['exit_time']).dt.year
    trades_grouped = trades_df.groupby('year')

    for year, d in trades_grouped:
        print(f'==== year: {year} =====')
        stats(d)
 
def plot_stats(df):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # Equity
    ax1.plot(df["equity_R"], label="Equity R")
    ax1.fill_between(df.index, df["equity_R"], df["equity_R_peak"], alpha=0.3)
    ax1.legend()
    ax1.set_title("Equity Curve")

    # Drawdown
    ax2.plot(df["drawdown_R"], label="Drawdown")
    ax2.fill_between(df.index, df["drawdown_R"], 0, alpha=0.3)
    ax2.axhline(0)
    ax2.legend()
    ax2.set_title("Drawdown")

    plt.show()

def plot_trades_indicators(df, markers=[], indicators = []):
    
    chart = Chart()
    chart.legend(visible=True)
    chart.set(df)
    
    for item in indicators:
        (name, data, color) = item
        line = chart.create_line(name=name,color=color)
        line.set(data)
    
    chart.marker_list(markers)
    chart.show(block=True)
    return

 
    
def trades_stats(func):
    def wrapper(*args, **kwargs):
        
        
        _df = func(*args, **kwargs)
        _df = _df.sort_values(by=['entry_time']).reset_index(drop=True)
        (_stats, df )= stats(_df)
        pprint(_stats)

        plot_stats(df)
        
        return _df
    return wrapper

        
def is_split_day(df, previous_day_close):
    
    first_open = df.iloc[0]['open']
    perc_change = ((first_open - previous_day_close) / previous_day_close) * 100
    
    return perc_change <= -500 or perc_change >= 500


def create_marker_from_signals_from_trades(trades):
    
    df = pd.DataFrame(trades)
    if df is None or df.empty:
        return
    
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df['exit_time'] = pd.to_datetime(df['exit_time'])
    df_len = len(df)
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

# save trades to parquet files
def save_trades(filepath, trades):
    
    trades.to_parquet(filepath, index=False)
    
    return 

# save trades to parquet files
def load_trades(filepath):
    
    trades = pd.read_parquet(filepath)
    trades['entry_time'] = pd.to_datetime(trades['entry_time'])
    trades['exit_time'] = pd.to_datetime(trades['exit_time'])
    
    return trades