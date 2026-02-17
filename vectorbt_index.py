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


present = datetime.today().strftime('%Y-%m-%d')

asset = yf.download("NVDA", start="2000-01-01", end=present)
asset.columns = [col[0] for col in asset.columns]
asset.rename(columns={'Open': 'open', 'Close': 'close', 'High': 'high', 'Low': 'low', 'Volume': 'volume'}, inplace=True)


asset["sma_200"] = ta.sma(asset["close"], length=200)
asset["sma_9"]   = ta.sma(asset)
asset["sma_5"]   = ta.sma(asset["close"], length=9)
asset["atr"] = ta.atr(asset["high"],asset["low"],asset["close"], length=14)
asset =  utils_helpers.donchainChannel(asset, lookback=5, offset=1)

print(asset['close'])


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

    # =============================
    # Results
    # =============================
    print(pf.stats())
    pf.plot().show()
    
    
    return


@njit(cache=True, fastmath=True)
def simulate_numba_fast(
    close,
    entries,
    exits,
     initial_capital = 0,
    slippage = 0.001,
    fees = 0
):
    
    n = close.shape[0]
    
    cash = initial_capital
    position = 0.0
    
    equity = np.empty(n, dtype=np.float64)
    
    for i in range(n):
        
        price = close[i]
        
        if entries[i] and position == 0.0:
            
            exec_price = price * (1.0 + slippage)
            
            
            size = cash / exec_price if initial_capital > 0 else 1 
            
            cost = size * exec_price
            fee_cost = cost * fees
            
            cash -= cost + fee_cost
            position = size
        
        elif exits[i] and position > 0.0:
            
            exec_price = price * (1.0 - slippage)
            proceeds = position * exec_price
            fee_cost = proceeds * fees
            
            cash += proceeds - fee_cost
            position = 0.0
        
        equity[i] = cash + position * price
    
    return equity


def simulate_slow(
    close,
    entries,
    exits,
    initial_capital = 0,
    slippage = 0.001,
    fees = 0
):
    
   
    print(close.shape)
    n = close.shape[0]
    
    temp =  np.full((n, 3), np.nan)
    
    print(temp)
    
    cash = initial_capital
    position = 0.0
    
    equity = np.empty(n, dtype=np.float64)
    
    
    
    for i in range(n):
        
        price = close[i]
        idx = close.index[i]
        #print(idx,price)
        
        if entries[i] and position == 0.0:
            
            exec_price = price * (1.0 + slippage)
            
            
            size = cash / exec_price if initial_capital > 0 else 1 
            
            cost = size * exec_price
            fee_cost = cost * fees
            
            cash -= cost + fee_cost
            position = size
        
        elif exits[i] and position > 0.0:
            
            exec_price = price * (1.0 - slippage)
            proceeds = position * exec_price
            fee_cost = proceeds * fees
            
            cash += proceeds - fee_cost
            position = 0.0
        
        equity[i] = cash + position * price
    
    return equity


def test_sim(df):
    
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
   

    # =============================
    # Exit Conditions (reversal logic)
    # =============================
    long_exits = close < lower_shifted
    
    equity  = simulate_slow(close=close,entries=long_entries, exits=long_exits)
   
    print(equity)
    
    
    return


test_sim(asset)



