import os
import sys

sys.path.insert(0, os.path.abspath("."))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import utils
from utils import helpers as utils_helpers
from lightweight_charts import Chart
from datetime import datetime, timedelta, time, date
import time as tm
from dotenv import load_dotenv
from multiprocessing import Pool, cpu_count, current_process
from functools import wraps
load_dotenv() 

connectionParams ={}
connectionParams['POSTGREST_H'] = 'localhost'  # os.getenv("POSTGREST_H", "none")
connectionParams['POSTGREST_P'] = '3030' # os.getenv("POSTGREST_P", "none")
connectionParams['POSTGREST_TOKEN'] =  os.getenv("POSTGREST_TOKEN", "none")

WORKER_ID = None

def init_worker():
    global WORKER_ID
    WORKER_ID = current_process()._identity[0] - 1
 
    
def _pipeline_wrapper(func, *args, **kwargs):
    global WORKER_ID
    try:
        return func(*args, id=WORKER_ID, **kwargs)
    except Exception as e:
        return {
            "worker_id": WORKER_ID,
            "status": "failed",
            "error": str(e)
           
        }

# decorator that allows to apply wrap the call of any function/strategy and use them inside the process Pool.
def pipeline(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        return _pipeline_wrapper(func, *args, **kwargs)
        
    return wrapped

# the starting point
def main(init_worker = None, strategy_func=None, n_cpus=1, chunk_size = 10):
    """
    this will get the the gappers from data base and apply the given strategy
    """
    start = tm.perf_counter()
    ids = range(0, n_cpus)  # example ids
    #all_data = utils_helpers.fetch_all_data_from_gappers(connectionParams)
    #all_data = all_data.iloc[:30000]
    gappers_backtest_dataset = pd.read_csv('small_caps_strategies/gappers_backtest_dataset.csv')
    splits = utils_helpers.split_df_by_size(gappers_backtest_dataset, chunk_size = chunk_size)

    with Pool(processes=n_cpus, initializer=init_worker) as pool:
        results = list(pool.imap_unordered(strategy_func, splits))
        
    end = tm.perf_counter()

    print(f"Total elapsed time: {end - start:.2f} seconds")
    return

# ====== out of sample runs ======
def out_of_sample(init_worker = None, strategy_func=None, n_cpus=1, chunk_size = 10):
    """
    this will get the the gappers from data base and apply the given strategy
    """

    start = tm.perf_counter()
    ids = range(0, n_cpus)  # example ids
    gappers_backtest_dataset = pd.read_csv('small_caps_strategies/gappers_out_of_sample_dataset.csv')
    splits = utils_helpers.split_df_by_size(gappers_backtest_dataset, chunk_size = chunk_size)

    with Pool(processes=n_cpus, initializer=init_worker) as pool:
        results = list(pool.imap_unordered(strategy_func, splits))
        
    end = tm.perf_counter()

    print(f"Total elapsed time: {end - start:.2f} seconds")
    return

def walk_fordward_runner(init_worker = None, strategy_func=None, n_cpus=1, chunk_size = 10, sample_type='in_sample', walk_fordward_step = 1):
    """
    this will get the the gappers from data base and apply the given strategy
    """
    
    print(" Walk fordward step:", walk_fordward_step)

    start = tm.perf_counter()
    ids = range(0, n_cpus)  # example ids
    
    file_name = f'{sample_type}_{walk_fordward_step}'
    file_path = f'backtest_dataset/walk_fordward/{file_name}.parquet'
    #out_sample_path = f'backtest_dataset/walk_fordward/out_sample_{walk_fordward_step}.parquet'
    
    gappers_backtest_dataset = pd.read_parquet(file_path)
    splits = utils_helpers.split_df_by_size(gappers_backtest_dataset, chunk_size = chunk_size)
    tasks = [(split, walk_fordward_step, file_name) for split in splits ]

    with Pool(processes=n_cpus, initializer=init_worker) as pool:
        results = list(pool.imap_unordered(strategy_func, tasks))
        
    end = tm.perf_counter()

    print(f"Total elapsed time: {end - start:.2f} seconds")
    return