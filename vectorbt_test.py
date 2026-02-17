import vectorbt as vbt
import pandas as pd
import numpy as np
from utils import helpers as utils_helpers, trade_metrics as tm
from pprint import pprint

print(" --- testing vectorBT ---")

# all_data = utils_helpers.fetch_all_data_from_gappers(connectionParams)
# backtest_data = all_data.iloc[:10000]
# df_data =  backtest_data[backtest_data['gap_perc'] >= 40]
    

df_data = pd.read_csv('small_caps_strategies/gappers_backtest_dataset.csv')

filter =  ['ticker', 'date_str', 
       'previous_close',  'volume', 
       'premarket_volume', 'market_hours_volume', 'high_mh', 'high_pm',
       'low_pm', 'highest_in_pm', 'gap_perc']
filtered = df_data


df_trades = pd.read_parquet('backtest_dataset/in_sample/trades/backside_short/backside_short_in_sample_trades.parquet')
filter2 = ['ticker', 'type', 'entry_price', 'stop_loss_price','previous_day_close' ,
      'entry_time','pnl',
       'strategy']

trades =  df_trades[filter2]


filtered['date'] = pd.to_datetime(filtered['date_str']).dt.date
trades['date']   = pd.to_datetime(trades['entry_time']).dt.date

print(filtered)
print(trades)

merged = trades.merge(
    filtered,
    on=['ticker', 'date'],
    how='left'
)

print(merged.columns)

print(merged[['ticker','entry_time','date','date_str', 'previous_close','previous_day_close','open','gap_perc','high_mh', 'high_pm','pnl', 'entry_price','strategy']])

winners = merged[merged['pnl'] > 0]

# ==================================================
# FILTER WINNERS
# ==================================================
winners = merged[merged['pnl'] > 0].copy()


# ==================================================
# CLASSIFY ENTRY VS HIGH_PM
# ==================================================
winners['below_high_pm'] = winners['entry_price'] < winners['high_pm']
winners['above_high_pm'] = winners['entry_price'] > winners['high_pm']
winners['equal_high_pm'] = winners['entry_price'] == winners['high_pm']


lossers = merged[merged['pnl'] < 0].copy()
lossers['below_high_pm'] = lossers['entry_price'] < lossers['high_pm']
lossers['above_high_pm'] = lossers['entry_price'] > lossers['high_pm']
lossers['equal_high_pm'] = lossers['entry_price'] == lossers['high_pm']


# ==================================================
# GLOBAL SUMMARY
# ==================================================
print("===== WINNING TRADES (PNL > 0) =====")
print(f"Entry BELOW high_pm : {winners['below_high_pm'].sum()}")
print(f"Entry ABOVE high_pm : {winners['above_high_pm'].sum()}")
print(f"Entry EQUAL high_pm : {winners['equal_high_pm'].sum()}")
print(f"TOTAL WINNERS       : {len(winners)}")


# ==================================================
# SUMMARY BY STRATEGY
# ==================================================
summary_by_strategy = (
    winners
    .groupby('strategy')
    .agg(
        total_winners=('pnl', 'count'),
        below_high_pm=('below_high_pm', 'sum'),
        above_high_pm=('above_high_pm', 'sum'),
        equal_high_pm=('equal_high_pm', 'sum')
    )
    .reset_index()
)

print("\n===== WINNING TRADES BY STRATEGY =====")
print(summary_by_strategy)


# ==================================================
# GLOBAL SUMMARY
# ==================================================
print("===== lossers TRADES (PNL < 0) =====")
print(f"Entry BELOW high_pm : {lossers['below_high_pm'].sum()}")
print(f"Entry ABOVE high_pm : {lossers['above_high_pm'].sum()}")
print(f"Entry EQUAL high_pm : {lossers['equal_high_pm'].sum()}")
print(f"TOTAL lossers       : {len(lossers)}")


# ==================================================
# SUMMARY BY STRATEGY
# ==================================================
summary_by_strategy_1 = (
    lossers
    .groupby('strategy')
    .agg(
        total_lossers=('pnl', 'count'),
        below_high_pm=('below_high_pm', 'sum'),
        above_high_pm=('above_high_pm', 'sum'),
        equal_high_pm=('equal_high_pm', 'sum')
    )
    .reset_index()
)
print("\n===== lossers TRADES BY STRATEGY =====")
print(summary_by_strategy_1)

df_trades['is_profit'] = df_trades['pnl'] >0
(_stats, df )= utils_helpers.stats(trades_df=df_trades)
pprint(_stats)
print(trades)
