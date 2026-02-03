import os
import sys
sys.path.insert(0, os.path.abspath("."))
from datetime import datetime, timedelta, time, date
import re
from utils import utils

connectionParams ={}
connectionParams['POSTGREST_H'] = 'localhost'  # os.getenv("POSTGREST_H", "none")
connectionParams['POSTGREST_P'] = '3030' # os.getenv("POSTGREST_P", "none")
connectionParams['POSTGREST_TOKEN'] =  os.getenv("POSTGREST_TOKEN", "none")

utils.createTickerList(connectionParams)