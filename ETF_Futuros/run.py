
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import vectorbt as vbt
import pandas as pd
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
#f_dict = {'NVDA':nvda}

#stratg.breakout_donchain(tsla)

stratg.breakout_donchain_multi_asset(f_dict)

stratg.mean_reversion_donchain_multi_asset(f_dict)


