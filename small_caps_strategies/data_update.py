import os
import sys
sys.path.insert(0, os.path.abspath("."))
from datetime import time, date, datetime
from utils import utils
from data_injection import injection_stock_data
import pandas as pd
import requests
import re
import json
from dotenv import load_dotenv
load_dotenv() 


connectionParams ={}
connectionParams['POSTGREST_H'] = 'localhost'  # os.getenv("POSTGREST_H", "none")
connectionParams['POSTGREST_P'] = '3001' # os.getenv("POSTGREST_P", "none")
connectionParams['POSTGREST_TOKEN'] =  os.getenv("POSTGREST_TOKEN", "none")

# ===============  
def refresh_materialized_views(connectionParams):
    """
    refresh the materilized view for gappers and after hour runner 
    """
    
    data = None
    if connectionParams != None :
            try:


                POSTGREST_H =  connectionParams['POSTGREST_H']  # os.getenv("POSTGREST_H", "none")
                POSTGREST_P =  connectionParams['POSTGREST_P']  # os.getenv("POSTGREST_P", "none")
                POSTGREST_TOKEN = connectionParams['POSTGREST_TOKEN']  # os.getenv("POSTGREST_TOKEN", "none")
                
                # Headers with Authorization
                headers = {
                    "Authorization": f"Bearer {POSTGREST_TOKEN}",
                    "Content-Type": "application/json",
                    'Accept':'application/json'
                }

                # The API endpoint
                url = f'http://{POSTGREST_H}:{POSTGREST_P}/rpc/refresh_materialized_view'

                response = requests.post(url)
              
                print(response.status_code)
                if response.status_code != 200:
                    print("error fetching data")
                    return 'Error'
                
                data = response.json()

               
            except Exception as e:
                print("Error:", e)   
    return data

# ===============
def log(text):
    
   
    today = datetime.today() - pd.Timedelta(hours=5)
    _date = today.strftime("%Y-%m-%d %H:%M:%S")
   
    
    data = {
    "Date": [_date],
    "message": [text],
   
    }
    df = pd.DataFrame(data)
    old_data = pd.DataFrame()
    
    old_data =  pd.read_csv("./small_caps_strategies/process_logs.csv")
    df_result = pd.concat([old_data, df], ignore_index=True)
    df_result.to_csv("./small_caps_strategies/process_logs.csv", index=False)
    
# IMPRTANT: this code allways start from scratch, if the process is stopped for any reason to resume it call the resume functions 
# for the different states of the process. 

print("****** refresh materializedviews  data...")
refresh_materialized_views(connectionParams)
log("refresh_materialized_views")

params = injection_stock_data.prepare_params_for_fetch(connectionParams)
log("prepare_params_for_fetch")

injection_stock_data.main_multiprocess_pipeline(params, 8, connectionParams)
log("main_multiprocess_pipeline")




