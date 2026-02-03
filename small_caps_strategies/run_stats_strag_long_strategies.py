import os
import sys
sys.path.insert(0, os.path.abspath("."))
import pandas as pd
from utils import utils, helpers
from pprint import pprint
import matplotlib.pyplot as plt


def filter_trades(df):
   
    df['entry_vs_prev_close_pct'] = ((df['entry_price'] - df['previous_day_close'])/df['previous_day_close'])*100
    df = df[df['entry_vs_prev_close_pct'] < 100 ]

    profits = df[df['is_profit'] ]
    losses = df[df['is_profit'] == False ]
    print(profits)
    print(losses)
    print(df)
    return df

@helpers.trades_stats
def get_backtest_stats_9sma_up():
    df1 = pd.read_parquet('trades/strategy_2x_sma9_up_pipeline_1.parquet')
    df2 = pd.read_parquet('trades/strategy_2x_sma9_up_pipeline_2.parquet')
    df3 = pd.read_parquet('trades/strategy_2x_sma9_up_pipeline_3.parquet')
    df4 = pd.read_parquet('trades/strategy_2x_sma9_up_pipeline_4.parquet')
    df5 = pd.read_parquet('trades/strategy_2x_sma9_up_pipeline_5.parquet')
    df6 = pd.read_parquet('trades/strategy_2x_sma9_up_pipeline_6.parquet')
    df7 = pd.read_parquet('trades/strategy_2x_sma9_up_pipeline_7.parquet')
    df8 = pd.read_parquet('trades/strategy_2x_sma9_up_pipeline_8.parquet')

    df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8], ignore_index=True)

    return df

@helpers.trades_stats
def get_out_of_sample_test_stats_9sma_up():
    df1 = pd.read_parquet('trades/out_of_sample/strategy_2x_sma9_up_pipeline_1.parquet')
    df2 = pd.read_parquet('trades/out_of_sample/strategy_2x_sma9_up_pipeline_2.parquet')
    df3 = pd.read_parquet('trades/out_of_sample/strategy_2x_sma9_up_pipeline_3.parquet')
    df4 = pd.read_parquet('trades/out_of_sample/strategy_2x_sma9_up_pipeline_4.parquet')
    df5 = pd.read_parquet('trades/out_of_sample/strategy_2x_sma9_up_pipeline_5.parquet')
    df6 = pd.read_parquet('trades/out_of_sample/strategy_2x_sma9_up_pipeline_6.parquet')
    df7 = pd.read_parquet('trades/out_of_sample/strategy_2x_sma9_up_pipeline_7.parquet')
    df8 = pd.read_parquet('trades/out_of_sample/strategy_2x_sma9_up_pipeline_8.parquet')

    df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8], ignore_index=True)

    return df

@helpers.trades_stats
def get_backtest_stats_1x():
    df1 = pd.read_parquet('trades/strategy_1x_tp_pipeline_1.parquet')
    df2 = pd.read_parquet('trades/strategy_1x_tp_pipeline_2.parquet')
    df3 = pd.read_parquet('trades/strategy_1x_tp_pipeline_3.parquet')
    df4 = pd.read_parquet('trades/strategy_1x_tp_pipeline_4.parquet')
    df5 = pd.read_parquet('trades/strategy_1x_tp_pipeline_5.parquet')
    df6 = pd.read_parquet('trades/strategy_1x_tp_pipeline_6.parquet')
    df7 = pd.read_parquet('trades/strategy_1x_tp_pipeline_7.parquet')
    df8 = pd.read_parquet('trades/strategy_1x_tp_pipeline_8.parquet')

    _df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8], ignore_index=True)
    return _df

@helpers.trades_stats
def get_out_of_sample_test_stats_1x():
    df1 = pd.read_parquet('trades/out_of_sample/strategy_1x_tp_pipeline_1.parquet')
    df2 = pd.read_parquet('trades/out_of_sample/strategy_1x_tp_pipeline_2.parquet')
    df3 = pd.read_parquet('trades/out_of_sample/strategy_1x_tp_pipeline_3.parquet')
    df4 = pd.read_parquet('trades/out_of_sample/strategy_1x_tp_pipeline_4.parquet')
    df5 = pd.read_parquet('trades/out_of_sample/strategy_1x_tp_pipeline_5.parquet')
    df6 = pd.read_parquet('trades/out_of_sample/strategy_1x_tp_pipeline_6.parquet')
    df7 = pd.read_parquet('trades/out_of_sample/strategy_1x_tp_pipeline_7.parquet')
    df8 = pd.read_parquet('trades/out_of_sample/strategy_1x_tp_pipeline_8.parquet')

    _df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8], ignore_index=True)
    return _df

@helpers.trades_stats
def get_backtest_stats_2x():
    df1 = pd.read_parquet('trades/strategy_2x_tp_pipeline_1.parquet')
    df2 = pd.read_parquet('trades/strategy_2x_tp_pipeline_2.parquet')
    df3 = pd.read_parquet('trades/strategy_2x_tp_pipeline_3.parquet')
    df4 = pd.read_parquet('trades/strategy_2x_tp_pipeline_4.parquet')
    df5 = pd.read_parquet('trades/strategy_2x_tp_pipeline_5.parquet')
    df6 = pd.read_parquet('trades/strategy_2x_tp_pipeline_6.parquet')
    df7 = pd.read_parquet('trades/strategy_2x_tp_pipeline_7.parquet')
    df8 = pd.read_parquet('trades/strategy_2x_tp_pipeline_8.parquet')

    _df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8], ignore_index=True)

    trades = filter_trades(_df)
    
    return trades

@helpers.trades_stats
def get_out_of_sample_test_stats_2x():
    df1 = pd.read_parquet('trades/out_of_sample/strategy_2x_tp_pipeline_1.parquet')
    df2 = pd.read_parquet('trades/out_of_sample/strategy_2x_tp_pipeline_2.parquet')
    df3 = pd.read_parquet('trades/out_of_sample/strategy_2x_tp_pipeline_3.parquet')
    df4 = pd.read_parquet('trades/out_of_sample/strategy_2x_tp_pipeline_4.parquet')
    #df5 = pd.read_parquet('trades/out_of_sample/strategy_2x_tp_pipeline_5.parquet')
    df5 = pd.DataFrame([])
    df6 = pd.read_parquet('trades/out_of_sample/strategy_2x_tp_pipeline_6.parquet')
    df7 = pd.read_parquet('trades/out_of_sample/strategy_2x_tp_pipeline_7.parquet')
    df8 = pd.read_parquet('trades/out_of_sample/strategy_2x_tp_pipeline_8.parquet')

    _df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8], ignore_index=True)
    trades = filter_trades(_df)
    
    return trades

@helpers.trades_stats
def get_backtest_stats_3x():
    df1 = pd.read_parquet('trades/strategy_3x_tp_pipeline_1.parquet')
    df2 = pd.read_parquet('trades/strategy_3x_tp_pipeline_2.parquet')
    df3 = pd.read_parquet('trades/strategy_3x_tp_pipeline_3.parquet')
    df4 = pd.read_parquet('trades/strategy_3x_tp_pipeline_4.parquet')
    df5 = pd.read_parquet('trades/strategy_3x_tp_pipeline_5.parquet')
    df6 = pd.read_parquet('trades/strategy_3x_tp_pipeline_6.parquet')
    df7 = pd.read_parquet('trades/strategy_3x_tp_pipeline_7.parquet')
    df8 = pd.read_parquet('trades/strategy_3x_tp_pipeline_8.parquet')

    _df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8], ignore_index=True)
    return _df

@helpers.trades_stats
def get_out_of_sample_test_stats_3x():
    df1 = pd.read_parquet('trades/out_of_sample/strategy_3x_tp_pipeline_1.parquet')
    df2 = pd.read_parquet('trades/out_of_sample/strategy_3x_tp_pipeline_2.parquet')
    df3 = pd.read_parquet('trades/out_of_sample/strategy_3x_tp_pipeline_3.parquet')
    df4 = pd.read_parquet('trades/out_of_sample/strategy_3x_tp_pipeline_4.parquet')
    df5 = pd.read_parquet('trades/out_of_sample/strategy_3x_tp_pipeline_5.parquet')
    df6 = pd.read_parquet('trades/out_of_sample/strategy_3x_tp_pipeline_6.parquet')
    df7 = pd.read_parquet('trades/out_of_sample/strategy_3x_tp_pipeline_7.parquet')
    df8 = pd.read_parquet('trades/out_of_sample/strategy_3x_tp_pipeline_8.parquet')

    _df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8], ignore_index=True)
    return _df

@helpers.trades_stats
def get_backtest_stats_4x():
    df1 = pd.read_parquet('trades/strategy_4x_tp_pipeline_1.parquet')
    df2 = pd.read_parquet('trades/strategy_4x_tp_pipeline_2.parquet')
    df3 = pd.read_parquet('trades/strategy_4x_tp_pipeline_3.parquet')
    df4 = pd.read_parquet('trades/strategy_4x_tp_pipeline_4.parquet')
    df5 = pd.read_parquet('trades/strategy_4x_tp_pipeline_5.parquet')
    df6 = pd.read_parquet('trades/strategy_4x_tp_pipeline_6.parquet')
    df7 = pd.read_parquet('trades/strategy_4x_tp_pipeline_7.parquet')
    df8 = pd.read_parquet('trades/strategy_4x_tp_pipeline_8.parquet')

    _df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8], ignore_index=True)
    return _df

@helpers.trades_stats
def get_out_of_sample_test_stats_4x():
    df1 = pd.read_parquet('trades/out_of_sample/strategy_4x_tp_pipeline_1.parquet')
    df2 = pd.read_parquet('trades/out_of_sample/strategy_4x_tp_pipeline_2.parquet')
    df3 = pd.read_parquet('trades/out_of_sample/strategy_4x_tp_pipeline_3.parquet')
    df4 = pd.read_parquet('trades/out_of_sample/strategy_4x_tp_pipeline_4.parquet')
    df5 = pd.read_parquet('trades/out_of_sample/strategy_4x_tp_pipeline_5.parquet')
    df6 = pd.read_parquet('trades/out_of_sample/strategy_4x_tp_pipeline_6.parquet')
    df7 = pd.read_parquet('trades/out_of_sample/strategy_4x_tp_pipeline_7.parquet')
    df8 = pd.read_parquet('trades/out_of_sample/strategy_4x_tp_pipeline_8.parquet')

    _df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8], ignore_index=True)
    return _df

@helpers.trades_stats
def get_backtest_stats_5x():
    df1 = pd.read_parquet('trades/strategy_5x_tp_pipeline_1.parquet')
    df2 = pd.read_parquet('trades/strategy_5x_tp_pipeline_2.parquet')
    df3 = pd.read_parquet('trades/strategy_5x_tp_pipeline_3.parquet')
    df4 = pd.read_parquet('trades/strategy_5x_tp_pipeline_4.parquet')
    df5 = pd.read_parquet('trades/strategy_5x_tp_pipeline_5.parquet')
    df6 = pd.read_parquet('trades/strategy_5x_tp_pipeline_6.parquet')
    df7 = pd.read_parquet('trades/strategy_5x_tp_pipeline_7.parquet')
    df8 = pd.read_parquet('trades/strategy_5x_tp_pipeline_8.parquet')

    _df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8], ignore_index=True)
    return _df

@helpers.trades_stats
def get_out_of_sample_test_stats_5x():
    df1 = pd.read_parquet('trades/out_of_sample/strategy_5x_tp_pipeline_1.parquet')
    df2 = pd.read_parquet('trades/out_of_sample/strategy_5x_tp_pipeline_2.parquet')
    df3 = pd.read_parquet('trades/out_of_sample/strategy_5x_tp_pipeline_3.parquet')
    df4 = pd.read_parquet('trades/out_of_sample/strategy_5x_tp_pipeline_4.parquet')
    df5 = pd.read_parquet('trades/out_of_sample/strategy_5x_tp_pipeline_5.parquet')
    df6 = pd.read_parquet('trades/out_of_sample/strategy_5x_tp_pipeline_6.parquet')
    df7 = pd.read_parquet('trades/out_of_sample/strategy_5x_tp_pipeline_7.parquet')
    df8 = pd.read_parquet('trades/out_of_sample/strategy_5x_tp_pipeline_8.parquet')

    _df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8], ignore_index=True)
    return _df

@helpers.trades_stats
def get_backtest_stats_dynamic_exit():
    df1 = pd.read_parquet('trades/strategy_dynamic_tp_stp_log__pipeline_1.parquet')
    df2 = pd.read_parquet('trades/strategy_dynamic_tp_stp_log__pipeline_2.parquet')
    df3 = pd.read_parquet('trades/strategy_dynamic_tp_stp_log__pipeline_3.parquet')
    df4 = pd.read_parquet('trades/strategy_dynamic_tp_stp_log__pipeline_4.parquet')
    df5 = pd.read_parquet('trades/strategy_dynamic_tp_stp_log__pipeline_5.parquet')
    df6 = pd.read_parquet('trades/strategy_dynamic_tp_stp_log__pipeline_6.parquet')
    df7 = pd.read_parquet('trades/strategy_dynamic_tp_stp_log__pipeline_7.parquet')
    df8 = pd.read_parquet('trades/strategy_dynamic_tp_stp_log__pipeline_8.parquet')

    df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8], ignore_index=True)
    return df

@helpers.trades_stats
def get_out_of_sample_test_stats_dynamic_exit():
    df1 = pd.read_parquet('trades/out_of_sample/strategy_dynamic_tp_stp_log__pipeline_1.parquet')
    df2 = pd.read_parquet('trades/out_of_sample/strategy_dynamic_tp_stp_log__pipeline_2.parquet')
    df3 = pd.read_parquet('trades/out_of_sample/strategy_dynamic_tp_stp_log__pipeline_3.parquet')
    df4 = pd.read_parquet('trades/out_of_sample/strategy_dynamic_tp_stp_log__pipeline_4.parquet')
    df5 = pd.read_parquet('trades/out_of_sample/strategy_dynamic_tp_stp_log__pipeline_5.parquet')
    df6 = pd.read_parquet('trades/out_of_sample/strategy_dynamic_tp_stp_log__pipeline_6.parquet')
    df7 = pd.read_parquet('trades/out_of_sample/strategy_dynamic_tp_stp_log__pipeline_7.parquet')
    df8 = pd.read_parquet('trades/out_of_sample/strategy_dynamic_tp_stp_log__pipeline_8.parquet')

    df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8], ignore_index=True)
    return df


get_backtest_stats_1x()