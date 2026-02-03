import pandas as pd

from utils import utils, helpers
import json
import asyncio
import aiohttp
import time
import pandas as pd
import numpy as np
import os
import sys
import requests
from pprint import pprint


connectionParams ={}
connectionParams['POSTGREST_H'] = 'localhost'  # os.getenv("POSTGREST_H", "none")
connectionParams['POSTGREST_P'] = '3030' # os.getenv("POSTGREST_P", "none")
connectionParams['POSTGREST_TOKEN'] =  os.getenv("POSTGREST_TOKEN", "none")

def fetch_all_data(connectionParams):
    
    
    df = pd.DataFrame()
    if connectionParams != None:
            try:


                POSTGREST_H =  connectionParams['POSTGREST_H']  # os.getenv("POSTGREST_H", "none")
                POSTGREST_P =  connectionParams['POSTGREST_P']  # os.getenv("POSTGREST_P", "none")
                POSTGREST_TOKEN = connectionParams['POSTGREST_TOKEN']  # os.getenv("POSTGREST_TOKEN", "none")
                
                #connect_to_DB_1(user=DB_USER, password=DB_PASSWORD, host=POSTGREST_H, port=POSTGREST_P, database=DB_NAME)
                
                # Headers with Authorization
                headers = {
                    "Authorization": f"Bearer {POSTGREST_TOKEN}",
                    "Content-Type": "application/json",
                    'Accept':'application/json'
                }

                # The API endpoint
                url = f'http://{POSTGREST_H}:{POSTGREST_P}/gappers_pm_and_market_hours'

                response = requests.get(url)
              
                if response.status_code != 200:
                    print(f"error fetching data: get_latest_ticker_date: {response.status_code}")
                    return pd.DataFrame()
                
                data = response.json()
                df = pd.DataFrame(data)
                
               

            except Exception as e:
                print("Error:", e)

    return df






    



   

