import os
import sys
sys.path.insert(0, os.path.abspath("."))
import pandas as pd
from utils import utils, helpers
from pprint import pprint
import matplotlib.pyplot as plt
from pprint import pprint


def filter_trades(df):
    
    trades = df.copy()
    
    risk = (trades["entry_price"] - trades["stop_loss_price"]).abs().round(3)
    Rs = trades["pnl"] /risk
    trades['R'] = Rs
    trades['prct'] = (trades['entry_price'] - trades['previous_day_close'])/trades['previous_day_close']
    print(Rs.max(), Rs.min()   )
    print(trades[trades['R'] == trades['R'].max()][['ticker', 'previous_day_close', 'entry_price', 'exit_price', 'stop_loss_price', 'pnl', 'R','entry_time', 'exit_time','prct']])
    print(trades[trades['R'] == trades['R'].min()][['ticker', 'previous_day_close', 'entry_price', 'exit_price', 'stop_loss_price', 'pnl', 'R','entry_time', 'exit_time','prct']])
    
    trades.sort_values(by=['R'], inplace=True)
    df['prct'] = (df['entry_price'] - df['previous_day_close'])/df['previous_day_close']
    df = df[df['entry_price'] < 20]

    print(trades[['ticker', 'previous_day_close', 'entry_price', 'exit_price', 'stop_loss_price', 'pnl', 'R','entry_time', 'prct']])

    return df

@helpers.trades_stats
def get_backtest_stats_short_break_structure_at_50():
    
    df1 = pd.read_parquet('trades/strategy_short_break_structure_at_50_pipeline_1.parquet')
    df2 = pd.read_parquet('trades/strategy_short_break_structure_at_50_pipeline_2.parquet')
    df3 = pd.read_parquet('trades/strategy_short_break_structure_at_50_pipeline_3.parquet')
    df4 = pd.read_parquet('trades/strategy_short_break_structure_at_50_pipeline_4.parquet')
    df5 = pd.read_parquet('trades/strategy_short_break_structure_at_50_pipeline_5.parquet')
    df6 = pd.read_parquet('trades/strategy_short_break_structure_at_50_pipeline_6.parquet')
    df7 = pd.read_parquet('trades/strategy_short_break_structure_at_50_pipeline_7.parquet')
    df8 = pd.read_parquet('trades/strategy_short_break_structure_at_50_pipeline_8.parquet')

    _df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8], ignore_index=True)
    return _df
   
@helpers.trades_stats
def get_out_of_sample_test_stats_short_break_structure_at_50():
    
    df1 = pd.DataFrame([]) # pd.read_parquet('trades/out_of_sample/strategy_short_break_structure_at_50_pipeline_1.parquet')
    df2 = pd.read_parquet('trades/out_of_sample/strategy_short_break_structure_at_50_pipeline_2.parquet')
    df3 = pd.read_parquet('trades/out_of_sample/strategy_short_break_structure_at_50_pipeline_3.parquet')
    df4 = pd.read_parquet('trades/out_of_sample/strategy_short_break_structure_at_50_pipeline_4.parquet')
    df5 = pd.read_parquet('trades/out_of_sample/strategy_short_break_structure_at_50_pipeline_5.parquet')
    df6 = pd.read_parquet('trades/out_of_sample/strategy_short_break_structure_at_50_pipeline_6.parquet')
    df7 = pd.read_parquet('trades/out_of_sample/strategy_short_break_structure_at_50_pipeline_7.parquet')
    df8 = pd.read_parquet('trades/out_of_sample/strategy_short_break_structure_at_50_pipeline_8.parquet')

    _df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8], ignore_index=True)
    return _df
    
@helpers.trades_stats
def get_backtest_stats_short_break_structure_at_50_end_of_day():
    df1 = pd.read_parquet('trades/strategy_short_break_structure_at_50_end_of_day_pipeline_1.parquet')
    df2 = pd.read_parquet('trades/strategy_short_break_structure_at_50_end_of_day_pipeline_2.parquet')
    df3 = pd.read_parquet('trades/strategy_short_break_structure_at_50_end_of_day_pipeline_3.parquet')
    df4 = pd.read_parquet('trades/strategy_short_break_structure_at_50_end_of_day_pipeline_4.parquet')
    df5 = pd.read_parquet('trades/strategy_short_break_structure_at_50_end_of_day_pipeline_5.parquet')
    df6 = pd.read_parquet('trades/strategy_short_break_structure_at_50_end_of_day_pipeline_6.parquet')
    df7 = pd.read_parquet('trades/strategy_short_break_structure_at_50_end_of_day_pipeline_7.parquet')
    df8 = pd.read_parquet('trades/strategy_short_break_structure_at_50_end_of_day_pipeline_8.parquet')

    _df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8], ignore_index=True)

    return _df

@helpers.trades_stats
def get_out_of_sample_test_stats_short_break_structure_at_50_end_of_day():
    df1 = pd.read_parquet('trades/out_of_sample/strategy_short_break_structure_at_50_end_of_day_pipeline_1.parquet')
    df2 = pd.read_parquet('trades/out_of_sample/strategy_short_break_structure_at_50_end_of_day_pipeline_2.parquet')
    df3 = pd.read_parquet('trades/out_of_sample/strategy_short_break_structure_at_50_end_of_day_pipeline_3.parquet')
    df4 = pd.read_parquet('trades/out_of_sample/strategy_short_break_structure_at_50_end_of_day_pipeline_4.parquet')
    df5 = pd.read_parquet('trades/out_of_sample/strategy_short_break_structure_at_50_end_of_day_pipeline_5.parquet')
    df6 = pd.read_parquet('trades/out_of_sample/strategy_short_break_structure_at_50_end_of_day_pipeline_6.parquet')
    df7 = pd.read_parquet('trades/out_of_sample/strategy_short_break_structure_at_50_end_of_day_pipeline_7.parquet')
    df8 = pd.read_parquet('trades/out_of_sample/strategy_short_break_structure_at_50_end_of_day_pipeline_8.parquet')

    _df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8], ignore_index=True)
    return _df

@helpers.trades_stats
def get_backtest_stats_short_break_structure_at_80():
    
    df1 = pd.read_parquet('trades/strategy_short_break_structure_at_80_pipeline_1.parquet')
    df2 = pd.read_parquet('trades/strategy_short_break_structure_at_80_pipeline_2.parquet')
    df3 = pd.read_parquet('trades/strategy_short_break_structure_at_80_pipeline_3.parquet')
    df4 = pd.read_parquet('trades/strategy_short_break_structure_at_80_pipeline_4.parquet')
    df5 = pd.read_parquet('trades/strategy_short_break_structure_at_80_pipeline_5.parquet')
    df6 = pd.read_parquet('trades/strategy_short_break_structure_at_80_pipeline_6.parquet')
    df7 = pd.read_parquet('trades/strategy_short_break_structure_at_80_pipeline_7.parquet')
    df8 = pd.read_parquet('trades/strategy_short_break_structure_at_80_pipeline_8.parquet')
    _df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8], ignore_index=True)
    return _df
   
@helpers.trades_stats
def get_out_of_sample_test_stats_short_break_structure_at_80():
    
    df1 = pd.DataFrame([]) # pd.read_parquet('trades/out_of_sample/strategy_short_break_structure_at_80_pipeline_1.parquet')
    df2 = pd.read_parquet('trades/out_of_sample/strategy_short_break_structure_at_80_pipeline_2.parquet')
    df3 = pd.read_parquet('trades/out_of_sample/strategy_short_break_structure_at_80_pipeline_3.parquet')
    df4 = pd.read_parquet('trades/out_of_sample/strategy_short_break_structure_at_80_pipeline_4.parquet')
    df5 = pd.read_parquet('trades/out_of_sample/strategy_short_break_structure_at_80_pipeline_5.parquet')
    df6 = pd.read_parquet('trades/out_of_sample/strategy_short_break_structure_at_80_pipeline_6.parquet')
    df7 = pd.read_parquet('trades/out_of_sample/strategy_short_break_structure_at_80_pipeline_7.parquet')
    df8 = pd.read_parquet('trades/out_of_sample/strategy_short_break_structure_at_80_pipeline_8.parquet')

    _df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8], ignore_index=True)
    return _df


@helpers.trades_stats
def get_backtest_stats_short_break_structure_at_80_end_of_day():
    df1 = pd.read_parquet('trades/strategy_short_break_structure_at_80_end_of_day_pipeline_1.parquet')
    df2 = pd.read_parquet('trades/strategy_short_break_structure_at_80_end_of_day_pipeline_2.parquet')
    df3 = pd.read_parquet('trades/strategy_short_break_structure_at_80_end_of_day_pipeline_3.parquet')
    df4 = pd.read_parquet('trades/strategy_short_break_structure_at_80_end_of_day_pipeline_4.parquet')
    df5 = pd.read_parquet('trades/strategy_short_break_structure_at_80_end_of_day_pipeline_5.parquet')
    df6 = pd.read_parquet('trades/strategy_short_break_structure_at_80_end_of_day_pipeline_6.parquet')
    df7 = pd.read_parquet('trades/strategy_short_break_structure_at_80_end_of_day_pipeline_7.parquet')
    df8 = pd.read_parquet('trades/strategy_short_break_structure_at_80_end_of_day_pipeline_8.parquet')

    _df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8], ignore_index=True)

    return filter_trades(_df)

@helpers.trades_stats
def get_out_of_sample_test_stats_short_break_structure_at_80_end_of_day():
    df1 = pd.read_parquet('trades/out_of_sample/strategy_short_break_structure_at_80_end_of_day_pipeline_1.parquet')
    df2 = pd.read_parquet('trades/out_of_sample/strategy_short_break_structure_at_80_end_of_day_pipeline_2.parquet')
    df3 = pd.read_parquet('trades/out_of_sample/strategy_short_break_structure_at_80_end_of_day_pipeline_3.parquet')
    df4 = pd.read_parquet('trades/out_of_sample/strategy_short_break_structure_at_80_end_of_day_pipeline_4.parquet')
    df5 = pd.read_parquet('trades/out_of_sample/strategy_short_break_structure_at_80_end_of_day_pipeline_5.parquet')
    df6 = pd.read_parquet('trades/out_of_sample/strategy_short_break_structure_at_80_end_of_day_pipeline_6.parquet')
    df7 = pd.read_parquet('trades/out_of_sample/strategy_short_break_structure_at_80_end_of_day_pipeline_7.parquet')
    df8 = pd.read_parquet('trades/out_of_sample/strategy_short_break_structure_at_80_end_of_day_pipeline_8.parquet')

    _df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8], ignore_index=True)
    return _df


@helpers.trades_stats
def get_backtest_stats_pipeline_short_extention_candle_signal():
    df1 = pd.read_parquet('trades/strategy_short_extention_candle_signal_pipeline_1.parquet')
    df2 = pd.read_parquet('trades/strategy_short_extention_candle_signal_pipeline_2.parquet')
    df3 = pd.read_parquet('trades/strategy_short_extention_candle_signal_pipeline_3.parquet')
    df4 = pd.read_parquet('trades/strategy_short_extention_candle_signal_pipeline_4.parquet')
    df5 = pd.read_parquet('trades/strategy_short_extention_candle_signal_pipeline_5.parquet')
    df6 = pd.read_parquet('trades/strategy_short_extention_candle_signal_pipeline_6.parquet')
    df7 = pd.read_parquet('trades/strategy_short_extention_candle_signal_pipeline_7.parquet')
    df8 = pd.read_parquet('trades/strategy_short_extention_candle_signal_pipeline_8.parquet')

    _df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8], ignore_index=True)

    return filter_trades(_df)

@helpers.trades_stats
def get_backtest_stats_pipeline_short_extention_candle_signal_MH():
    df1 = pd.read_parquet('trades/strategy_short_extention_candle_signal_MH_pipeline_1.parquet')
    df2 = pd.read_parquet('trades/strategy_short_extention_candle_signal_MH_pipeline_2.parquet')
    df3 = pd.read_parquet('trades/strategy_short_extention_candle_signal_MH_pipeline_3.parquet')
    df4 = pd.read_parquet('trades/strategy_short_extention_candle_signal_MH_pipeline_4.parquet')
    df5 = pd.read_parquet('trades/strategy_short_extention_candle_signal_MH_pipeline_5.parquet')
    df6 = pd.read_parquet('trades/strategy_short_extention_candle_signal_MH_pipeline_6.parquet')
    df7 = pd.read_parquet('trades/strategy_short_extention_candle_signal_MH_pipeline_7.parquet')
    df8 = pd.read_parquet('trades/strategy_short_extention_candle_signal_MH_pipeline_8.parquet')

    _df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8], ignore_index=True)

    return filter_trades(_df)

#get_backtest_stats_short_break_structure_at_50()
#get_out_of_sample_test_stats_short_break_structure_at_50()
#get_backtest_stats_short_break_structure_at_50_end_of_day()
#get_out_of_sample_test_stats_short_break_structure_at_50_end_of_day()
#get_backtest_stats_short_break_structure_at_80_end_of_day()
#get_out_of_sample_test_stats_short_break_structure_at_80_end_of_day()
get_backtest_stats_pipeline_short_extention_candle_signal()
   