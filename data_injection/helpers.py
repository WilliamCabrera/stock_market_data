import aiohttp
import multiprocessing
from typing import List, Tuple, Dict, Any
import os
import sys
sys.path.insert(0, os.path.abspath("."))
from datetime import datetime, timedelta, time, date
import re
from utils import utils
import json
import time
import pandas as pd
import numpy as np


# first time any data needs to be injected in table stock_data (stock_data == empty)
def prepare_parameters_of_fetch_for_first_injection(connectionParams):
    
    today = datetime.today()
    start = datetime(today.year - 5, today.month, today.day).strftime('%Y-%m-%d')
    #_dates = utils.generate_date_interval_to_fetch(start)
    _dates = utils.month_ranges(start)

    stock_list = utils.get_ticker_list_from_db(connectionParams)
    if len(stock_list) == 0:
        print("===== please check the ticker info table is created and populated")
        return []
    
    tickers = stock_list['ticker'].to_numpy()
    #tickers = ['MOVE']
    filtered = [s for s in tickers if re.fullmatch(r"[A-Za-z]+", s)]
    
    list_len = len(filtered)
    parameters = []
    
    for index in range(0,list_len):
        
        ticker = filtered[index]        
        for item in _dates:
            (date1, date2) = item
            parameters.append((ticker, date1, date2))
    
    
    return parameters

# this function will check if new tickers has been listed in the market and prepare params to inject them in the DB.
def prepare_to_inject_new_listed_tickers(connectionParams):
    
    today = datetime.today()
    start = datetime(today.year - 4, today.month, today.day).strftime('%Y-%m-%d')
    #_dates = utils.generate_date_interval_to_fetch(start)
    _dates = utils.month_ranges(start)
    
    # getting the recently listed stocks
    df_dif_clean = utils.get_new_listed_stocks(connectionParams)
        
    # inject them in the stock list table in the DB
    utils.update_stock_list_table(connectionParams, df_dif_clean)

    if len(df_dif_clean) == 0:
        return []
    
    filtered = df_dif_clean['ticker'].to_numpy()
    
    list_len = len(filtered)
    parameters = []
    
    for index in range(0,list_len):
        
        ticker = filtered[index]        
        for item in _dates:
            (date1, date2) = item
            parameters.append((ticker, date1, date2))
    
    
    return parameters
    

# get the latest date update
def prepare_parameters_of_fetch_from_db(connectionParams):
    
    try:
        df = utils.get_latest_ticker_date(connectionParams)
        list_of_records = df.to_dict('records') 
        parameters = []
        
        #print(" ======= prepare_parameters_of_fetch_from_db ======")
        #print(df[df['latest_ticker'] == 'TSLA'])        
        # 2. Iterar sobre la lista de diccionarios
        for record in list_of_records:
           
            # Usar nombres descriptivos como 'record' o 'row'
            ticker = record['latest_ticker']
            date_str = record['latest_date_str']
            
            date_object = datetime.strptime(date_str, '%Y-%m-%d')
        
            # 2. Sumar un día usando timedelta
            next_day_object = date_object + timedelta(days=1)
            
            # 3. Formatear el objeto datetime resultante de vuelta a la cadena 'YYYY-MM-DD'
            next_day_str = next_day_object.strftime('%Y-%m-%d')
            
            # 3. Usar un nombre de variable diferente para el bucle interno (ej. 'date_interval')
            #_dates = utils.generate_date_interval_to_fetch_30_minutes(next_day_str)
            _dates = utils.month_ranges(next_day_str)
            
            for date_interval in _dates:
                (date1, date2) = date_interval
                parameters.append((ticker, date1, date2))
    
        
        return parameters 
        
    except Exception as e:
                print("Error prepare_parameters_of_fetch_from_db:", e)
                print("injection process will be stopped")
                return None
            

# prepare params for stock table injection (it will go first with the latest)    
def prepare_params_for_fetch(connectionParams):
    
    if connectionParams is None:
        return []
    
    
    
    params = prepare_parameters_of_fetch_from_db(connectionParams)
    params = []
    # this means that there is problems , need to stop the process
    if params is None:    
        return []
    
    
    # # the DB base has already data so need to check if new ticker has been listed
    # if len(params) > 0:
    #     new_stock_params = prepare_to_inject_new_listed_tickers(connectionParams)
    #     new_list = params + new_stock_params
              
    #     return new_list
        
    
    
    # BD empty need to inject everything
    elif len(params) == 0:
        
        params =  prepare_parameters_of_fetch_for_first_injection(connectionParams)
    
    return params #  [("RVPH","2022-01-08","2022-01-11")]


# Función A: Fetch a una API (Asíncrona)
async def fetch_data_30_min(session: aiohttp.ClientSession, params) :
    # URL de ejemplo
    
    (ticker, start_date, end_date) = params

    # Polygon.io API details
    API_KEY = os.getenv("API_KEY", "none")  # Replace with your Polygon.io API key
    
    BASE_URL_30_MINUTES = "https://api.polygon.io/v2/aggs/ticker/{ticker}/range/30/minute/{start_date}/{end_date}?adjusted=true&sort=asc&apiKey={apiKey}&limit=50000"
    url_30_min = BASE_URL_30_MINUTES.format(ticker=ticker, start_date=start_date, end_date=end_date,apiKey=API_KEY)
    
   
    try:
        async with session.get(url_30_min) as response:
            # Lanza una excepción para códigos de estado 4xx/5xx
            response.raise_for_status() 
            # Devuelve el JSON de la respuesta
            res = await response.json()
            data = res.get("results", [])
            return data
    except KeyError as e:
        # Maneja la falta de un parámetro en connectionParams
        print(f"Error: Falta un parámetro de conexión clave: {e}")
        return False

    except aiohttp.ClientResponseError as e:
        # Manejo específico para errores de respuesta HTTP (4xx/5xx)
        # Puedes intentar leer el cuerpo de la respuesta para obtener más detalles
        try:
            error_details = await response.text()
        except:
            error_details = "No se pudo leer el cuerpo de la respuesta."
            
        print(f"Error HTTP {e.status} al hacer POST a {url}. Detalles: {error_details}")
        return False 
        
    except aiohttp.ClientError as e:
        # Captura otros errores de aiohttp (ej. errores de conexión o timeout)
        print(f"Error de aiohttp (Conexión/Timeout) al hacer POST a {url}: {e}")
        return False 
        
    except Exception as e:
        # Captura cualquier otro error (ej. error al hacer dumps de json)
        print(f"Error inesperado durante la ingesta: {e}")
        return False


# Función A: Fetch a una API (Asíncrona)
async def fetch_data_1_min(session: aiohttp.ClientSession, params) :
    # URL de ejemplo
    
    (ticker, start_date, end_date) = params

    # Polygon.io API details
    API_KEY = os.getenv("API_KEY", "none")  # Replace with your Polygon.io API key
    
    BASE_URL_1_MINUTES = "https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{start_date}/{end_date}?adjusted=true&sort=asc&apiKey={apiKey}&limit=50000"
    url_1_min = BASE_URL_1_MINUTES.format(ticker=ticker, start_date=start_date, end_date=end_date,apiKey=API_KEY)
    
    print(url_1_min)
    try:
        async with session.get(url_1_min) as response:
            # Lanza una excepción para códigos de estado 4xx/5xx
            response.raise_for_status() 
            # Devuelve el JSON de la respuesta
            res = await response.json()
            data = res.get("results", [])
            return data
    except KeyError as e:
        # Maneja la falta de un parámetro en connectionParams
        print(f"Error: Falta un parámetro de conexión clave: {e}")
        return False

    except aiohttp.ClientResponseError as e:
        # Manejo específico para errores de respuesta HTTP (4xx/5xx)
        # Puedes intentar leer el cuerpo de la respuesta para obtener más detalles
        try:
            error_details = await response.text()
        except:
            error_details = "No se pudo leer el cuerpo de la respuesta."
            
        print(f"Error HTTP {e.status} al hacer POST a {url}. Detalles: {error_details}")
        return False 
        
    except aiohttp.ClientError as e:
        # Captura otros errores de aiohttp (ej. errores de conexión o timeout)
        print(f"Error de aiohttp (Conexión/Timeout) al hacer POST a {url}: {e}")
        return False 
        
    except Exception as e:
        # Captura cualquier otro error (ej. error al hacer dumps de json)
        print(f"Error inesperado durante la ingesta: {e}")
        return False


# Función A: Fetch a una API (Asíncrona)
async def fetch_data_minutes_bar(session: aiohttp.ClientSession, params, multiplier=30) :
    # URL de ejemplo
    
    (ticker, start_date, end_date) = params

    # Polygon.io API details
    API_KEY = os.getenv("API_KEY", "none")  # Replace with your Polygon.io API key
    
    BASE_URL_1_MINUTES = "https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/minute/{start_date}/{end_date}?adjusted=true&sort=asc&apiKey={apiKey}&limit=50000"
    url_x_min = BASE_URL_1_MINUTES.format(ticker=ticker, start_date=start_date, end_date=end_date,apiKey=API_KEY)
    
    print(url_x_min)
    try:
        async with session.get(url_x_min) as response:
            # Lanza una excepción para códigos de estado 4xx/5xx
            response.raise_for_status() 
            # Devuelve el JSON de la respuesta
            res = await response.json()
            data = res.get("results", [])
            return data
    except KeyError as e:
        # Maneja la falta de un parámetro en connectionParams
        print(f"Error: Falta un parámetro de conexión clave: {e}")
        return False

    except aiohttp.ClientResponseError as e:
        # Manejo específico para errores de respuesta HTTP (4xx/5xx)
        # Puedes intentar leer el cuerpo de la respuesta para obtener más detalles
        try:
            error_details = await response.text()
        except:
            error_details = "No se pudo leer el cuerpo de la respuesta."
            
        print(f"Error HTTP {e.status} al hacer POST a {url}. Detalles: {error_details}")
        return False 
        
    except aiohttp.ClientError as e:
        # Captura otros errores de aiohttp (ej. errores de conexión o timeout)
        print(f"Error de aiohttp (Conexión/Timeout) al hacer POST a {url}: {e}")
        return False 
        
    except Exception as e:
        # Captura cualquier otro error (ej. error al hacer dumps de json)
        print(f"Error inesperado durante la ingesta: {e}")
        return False


# ===============
def log(text, file_path="logs.csv"):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    df = pd.DataFrame([[now, text]], columns=["Date", "message"])

    df.to_csv(
        file_path,
        mode="a",                          # append
        header=not os.path.exists(file_path),
        index=False
    )
   


   