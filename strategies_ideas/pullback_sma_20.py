import os
import sys
sys.path.insert(0, os.path.abspath("."))
import pandas as pd
from utils import utils, helpers
from pprint import pprint
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, time, date
import yfinance as yf
import pandas_ta as ta

print('======= pullback sma 20 =======')


def pullback_sma_20_trailing_stp(df):
    
    df = df.copy()
    df["SMA_200"] = ta.sma(df["close"], length=200)
    df["SMA_20"] = ta.sma(df["close"], length=20)
    
    df['sma_200_condition'] = df['close'] > df["SMA_200"] 
    df['sma20_above_200'] = df["SMA_20"] > df["SMA_200"]
    

    in_position = False
    entry_idx = None
    
    trades = []
    current_trade = {}
    
    for i in range(len(df) - 1):  # -1 porque usamos open del día siguiente
        
        if i < 2:
            continue
        
        entry_idx = i + 1  # open del día siguiente
        bar = df.iloc[i]
        previous_bar = df.iloc[i - 1]
        next_bar = df.iloc[i + 1]
        
        sma_20_cond =  bar['close'] >  bar['open'] and  bar['close'] > bar['SMA_20'] and  ( bar['low'] <  bar['SMA_20'] or previous_bar['low'] < bar['SMA_20'])
        
        # -------- ENTRADA --------
        if  in_position == False and bar['sma_200_condition'] and sma_20_cond and bar['sma20_above_200']:
           
            direction = 'long'
            entry_time = next_bar["date"]
            entry_price = next_bar["open"]
            day =  next_bar['date']
            trailing_stop =  bar['low']
            
            current_trade.update({
                "day": day,
                "type": direction,
                "entry_time": entry_time,
                "entry_price": entry_price,
                "stop_loss_price": trailing_stop,
            }) 
            
            in_position = True
            continue
        
        if in_position:
            _close = bar['close']
            
            if _close < trailing_stop:
                exit_signal = True
                
                exit_price = next_bar['open']
                pnl = exit_price - current_trade['entry_price'] if direction == 'long' else   current_trade['entry_price'] - exit_price 
                exit_time = next_bar["date"]
                current_trade.update({
                "exit_time": exit_time,
                "exit_price": exit_price,
                "pnl": pnl,
                "is_profit":pnl > 0,
                "exit_reason": None
                }) 
                
                trades.append(current_trade)
                current_trade = {}

                in_position = False
                entry_idx = None
                
            else:
                trailing_stop =  bar['low']
                continue
                
        
    
    return pd.DataFrame(trades)

activo = 'QQQ'
# =========== data from yfinance ==========
today = datetime.today()
start = datetime(today.year - 4, today.month, today.day).strftime('%Y-%m-%d')
present = datetime.today().strftime('%Y-%m-%d')
spy = yf.download(activo, start="2001-01-01", end=present)
spy.columns = [col[0] for col in spy.columns]
spy = spy.reset_index()
spy.rename(columns={'Open': 'open', 'Close': 'close', 'High': 'high', 'Low': 'low', 'Volumen': 'volume', 'Date':'date'}, inplace=True)

trades =  pullback_sma_20_trailing_stp(spy)
helpers.stats(trades)
print(trades)
#helpers.stats_per_year(trades)

print('====== other ======')
other_test = pd.read_csv('trades_pullback_qqq.csv')
other_test = other_test[["Size","EntryBar","ExitBar","EntryPrice","ExitPrice","PnL","EntryTime","ExitTime"]]
#other_test.rename(columns={'EntryPrice':'entry_price','ExitPrice':'exit_price', 'EntryTime':'entry_time', 'ExitTime':'exit_time', "PnL":'pnl'}, inplace=True)
print(other_test)

print('======= joined =======')

# 1. Asegúrate de que las columnas de tiempo sean objetos datetime
trades['entry_time'] = pd.to_datetime(trades['entry_time'])
other_test['EntryTime'] = pd.to_datetime(other_test['EntryTime'])

# 2. Realiza el join (merge)
# Usamos left_on y right_on porque los nombres de las columnas son distintos
# Usamos how='inner' para mantener solo las filas que coinciden en ambos
df_final = pd.merge(
    trades, 
    other_test, 
    left_on='entry_time', 
    right_on='EntryTime', 
    how='inner'
)

df_final = df_final.drop(columns=['Size','EntryBar','ExitBar','exit_reason','type','is_profit','day'])
print(df_final)


print('======== only in strategy 1 ========')

# 1. Unir con indicador
merged = pd.merge(trades, other_test, left_on='entry_time', right_on='EntryTime', how='left', indicator=True)

# 2. Filtrar solo los que están en la izquierda (df1) pero no en la derecha (df2)
df1_only = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])
df1_only = df1_only.drop(columns=['Size','EntryBar','ExitBar','exit_reason','type','is_profit','day'])
print(df1_only)



