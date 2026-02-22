
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import vectorbt as vbt
import pandas as pd
pd.options.display.float_format = '{:.3f}'.format

import numpy as np
from utils import helpers as utils_helpers, trade_metrics as tm
from pprint import pprint
from small_caps_strategies import commons
from datetime import datetime
import yfinance as yf
import pandas_ta as ta

import strategies as stratg


nvda = stratg.get_asset_from_yf('NVDA', date_from='2000-02-10')
tsla = stratg.get_asset_from_yf('TSLA', date_from='2000-02-10')


f_dict = {'NVDA':nvda, 'TSLA': tsla}

bkout = stratg.breakout_donchain(f_dict)

mrev = stratg.mean_reversion_donchain(f_dict)



# print(bkout['TSLA'][[ 'ticker', 'type','entry_price',
#     'exit_price','stop_loss_price',  'pnl',
#     'Return', 'entry_time','exit_time','strategy']])


# print(mrev['TSLA'][[ 'ticker', 'type','entry_price',
#     'exit_price','stop_loss_price',  'pnl',
#     'Return', 'entry_time','exit_time','strategy']])

risk_pct = 0.01
#mrevpnl = mrev['NVDA'][['pnl']]

rep_bkout = tm.summary_report(bkout['NVDA'],initial_capital=1000, risk_pct=risk_pct)
#rep_mrev = tm.summary_report(mrev['NVDA'],initial_capital=1000, risk_pct=risk_pct)


print('======= report BKOUT ==========')
print(rep_bkout)
# print('======= report MEN rev ==========')
# print(rep_mrev)

#tm.returns_distribution(mrevpnl)

tm.monte_carlo_final_equity_dd_sim(bkout['NVDA'], f=risk_pct)
#tm.monte_carlo_final_equity_dd_sim(mrev['NVDA'], f=risk_pct)


#print("*********** walk fordward **********")
#tslaWalkFordwardData = stratg.walk_fordward_split("TSLA",tsla)

# bkout_wfd = stratg.mean_reversion_donchain(tslaWalkFordwardData)
# for key in bkout_wfd:
#     print(f"************** Report {key}*************")
#     print(bkout_wfd[key])
#     rep_bkout = tm.summary_report(bkout_wfd[key],initial_capital=1000, risk_pct=0.05)
#     print(rep_bkout)






#tm.analysis_and_plot(mrev['TSLA'],initial_capital=1000, risk_pct=0.05)



