import vectorbt as vbt
import pandas as pd
import numpy as np

print(" --- testing vectorBT ---")
df_smx = pd.read_csv('SMX.csv')
df_mask = pd.read_csv('MASK.csv')
print(df_smx)
print(df_mask)

dfs = {
    'SMX': df_smx,
    'MASK': df_mask
}

df_merge = pd.concat(
        dfs,
        axis=1,
        names=['ticker', 'field']
    )

print(df_merge)

open_  = df_merge.xs('open', level='field', axis=1)
close  = df_merge.xs('close', level='field', axis=1)
high  = df_merge.xs('high', level='field', axis=1)
low  = df_merge.xs('low', level='field', axis=1)
sma9  = df_merge.xs('SMA_9', level='field', axis=1)
sma200  = df_merge.xs('SMA_200', level='field', axis=1)
prev_close = df_merge.xs('previous_day_close', level='field', axis=1)
can_trade  = df_merge.xs('can_trade', level='field', axis=1).astype(bool)

print(high)

entries = (
    (sma9 > sma200) &
    (close > prev_close * 1.10) &
    can_trade
)

entry_price = open_.shift(-1)
stop_price = low
risk = entry_price - stop_price
take_profit = entry_price + 2 * risk
entries &= risk > 0

pf = vbt.Portfolio.from_signals(
    close=close,
    entries=entries,
    exits=None,
    price=entry_price,
    sl_stop=stop_price,
    tp_stop=take_profit,
    direction='longonly',
    accumulate=False,
    freq='1min'
)

print(pf.trades.records_readable)
