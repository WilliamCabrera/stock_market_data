import os
import sys
sys.path.insert(0, os.path.abspath("."))
import pandas as pd
from utils import utils, helpers
from pprint import pprint
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, time, date
import yfinance as yf


def three_red_days(df):
    
    df["ATR"]  = helpers.compute_atr(df) 
    df = df.copy()
    
    # =============================
    # 1. Señal de entrada (3 closes bajistas consecutivos)
    # =============================
    df['down1'] = df['close'] < df['close'].shift(1)
    df['down2'] = df['close'].shift(1) < df['close'].shift(2)
    df['down3'] = df['close'].shift(2) < df['close'].shift(3)

    df['entry_signal'] = df['down1'] & df['down2'] & df['down3']
    
    # =============================
    # 2. Inicialización
    # =============================
   
    in_position = False
    entry_idx = None
    
    trades = []
    current_trade = {}

    # =============================
    # 3. Loop principal
    # =============================
    for i in range(len(df) - 1):  # -1 porque usamos open del día siguiente

        entry_idx = i + 1  # open del día siguiente
        bar = df.iloc[i]
        next_bar = df.iloc[i + 1]
        
        # -------- ENTRADA --------
        if  in_position == False and df['entry_signal'].iloc[i] and  pd.notna(bar['ATR']):
            
            direction = 'long'
           
            entry_time = next_bar["date"]
            entry_price = next_bar["open"]
            
            atr = bar['ATR']
            stop_loss = entry_price - 2 * bar['ATR']
            exit_price = entry_price + 2 * bar['ATR']
            day =  bar['date']

            current_trade.update({
                "day": day,
                "type": direction,
                "entry_time": entry_time,
                "entry_price": entry_price,
                "stop_loss_price": stop_loss,
            }) 

            in_position = True
            continue

        # -------- GESTIÓN DE POSICIÓN --------
        if in_position:
           
          
            low = bar['close']
            high = bar['high']

            exit_signal = False
            reason = None

            if low <= stop_loss:
                exit_signal = True
                reason = 'SL'
            elif high >= exit_price:
                exit_signal = True
                reason = 'TP'
                
            #print(f'==== exit {i}, {exit_signal}, low: {low}, stop_loss: {stop_loss}=====')

            # -------- SALIDA (open siguiente día) --------
            if exit_signal:
                
                exit_price = next_bar['open']
                pnl = exit_price - current_trade['entry_price'] if direction == 'long' else   current_trade['entry_price'] - exit_price 
                exit_time = next_bar["date"]
                current_trade.update({
                "exit_time": exit_time,
                "exit_price": exit_price,
                "pnl": pnl,
                "is_profit":pnl > 0,
                "exit_reason": reason
                }) 
                
                trades.append(current_trade)
                current_trade = {}

                in_position = False
                entry_idx = None
                
    return trades



activo = 'QQQ'
# =========== data from yfinance ==========
today = datetime.today()
start = datetime(today.year - 4, today.month, today.day).strftime('%Y-%m-%d')
present = datetime.today().strftime('%Y-%m-%d')
spy = yf.download(activo, start=start, end=present)
spy.columns = [col[0] for col in spy.columns]
spy = spy.reset_index()
spy.rename(columns={'Open': 'open', 'Close': 'close', 'High': 'high', 'Low': 'low', 'Volumen': 'volume', 'Date':'date'}, inplace=True)

print('======= data from yfinance =======')
trades = three_red_days(spy)
trades = pd.DataFrame(trades)
helpers.stats(trades)

print(trades)








# =========== data from polygon ==========
today = datetime.today()
start = datetime(today.year - 4, today.month, today.day).strftime('%Y-%m-%d')
df = utils.fetch_ticker_data_daily(activo, start, today.strftime('%Y-%m-%d'))
df = pd.DataFrame(df)
df.rename(columns={'o': 'open', 'c': 'close', 'h': 'high', 'l': 'low', 'v': 'volume', 't': 'time'}, inplace=True)
df["date"] =  pd.to_datetime(df["time"], unit='ms', utc=True) # pd.to_datetime(df["time"], unit='ms') - pd.Timedelta(hours=5) # -5 means New York timezone 
df["date"] = df["date"].dt.tz_convert("America/New_York")
df["day"] =  df["date"].dt.date 

print('======= data from polygon =======')
trades = three_red_days(df)
trades = pd.DataFrame(trades)

helpers.stats(trades)
helpers.stats_per_year(trades)

