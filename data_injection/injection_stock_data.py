import pandas as pd
import asyncio
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
import helpers as helpers

connectionParams ={}
connectionParams['POSTGREST_H'] = 'localhost'  # os.getenv("POSTGREST_H", "none")
connectionParams['POSTGREST_P'] = '3030' # os.getenv("POSTGREST_P", "none")
connectionParams['POSTGREST_TOKEN'] =  os.getenv("POSTGREST_TOKEN", "none")


MAX_CONCURRENT_REQUESTS = 100

# ==============================================================================
#                      MOCK-UPS DE FUNCIONES EXTERNAS
# ==============================================================================
# Nota: Estas funciones deben ser reemplazadas por tus implementaciones reales.

# Función A: Fetch a una API (Asíncrona)
async def fetch_data_30_min(session: aiohttp.ClientSession, params) :
    # URL de ejemplo
    
    (ticker, start_date, end_date) = params

    # Polygon.io API details
    API_KEY = os.getenv("API_KEY", "none")  # Replace with your Polygon.io API key
    
    BASE_URL_30_MINUTES = "https://api.polygon.io/v2/aggs/ticker/{ticker}/range/30/minute/{start_date}/{end_date}?adjusted=false&sort=asc&apiKey={apiKey}&limit=50000"
    url_30_min = BASE_URL_30_MINUTES.format(ticker=ticker, start_date=start_date, end_date=end_date,apiKey=API_KEY)
    
    print(url_30_min)
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
    
    BASE_URL_30_MINUTES = "https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{start_date}/{end_date}?adjusted=false&sort=asc&apiKey={apiKey}&limit=50000"
    url_30_min = BASE_URL_30_MINUTES.format(ticker=ticker, start_date=start_date, end_date=end_date,apiKey=API_KEY)
    
    print(url_30_min)
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


async def ingest_data(session: aiohttp.ClientSession, data, connectionParams = None):
    data_len = len(data)
    if connectionParams is None:
            return False
     
    #print(data[['ticker','time','date_str',"split_date_str", 'open', 'previous_close','split_adjust_factor' ]])  
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
        url = f'http://{POSTGREST_H}:{POSTGREST_P}/rpc/upsert_stock_data_from_json'
        
        result_json = {"p_data": data.to_dict(orient='records')}
        
        #print("result_json")
        #print(result_json)
        
        async with session.post(url, headers=headers, data=json.dumps(result_json)) as response:
            
            # Lanza una excepción para códigos de estado 4xx/5xx
            response.raise_for_status() 
            # Devuelve el JSON de la respuesta
            r =  await response.json()
           
            print(f"****** :  {r}:{data_len} *******")
            
            if response.status > 199 & response.status < 300 :
                return True
            
            return False
        
    except aiohttp.ClientResponseError as e:
        print(f"HTTP Error: {e.status} - {e.message}")
    
    try:
        # Crucial step: Read the response body for PostgreSQL details
        error_details = await e.response.text() 
        print(f"PostgREST Error: {error_details}")
    except Exception:
        print("Could not read error details from response body.")
        return -1
        
    except Exception as e:
        # Captura cualquier otro error (ej. error al hacer dumps de json)
        print(f"Error inesperado durante la ingesta: {e}")
        return False


def group_parameters_by_ticker(parameters: List[Tuple[str, str, str]]) -> Dict[str, List[Tuple[str, str, str]]]:
    """Agrupa la lista plana de (ticker, start_date, end_date) por ticker."""
    
    # Asumimos que los parámetros son (ticker, date1, date2)
    temp_df = pd.DataFrame(parameters, columns=["ticker", "date1", "date2"])
    
    # Agrupa y convierte de nuevo a un diccionario
    grouped_params = {
        ticker: list(df_group.itertuples(index=False, name=None))
        for ticker, df_group in temp_df.groupby("ticker")
    }
    return grouped_params

def partition_tickers_into_batches(grouped_params_dict: Dict[str, List], num_batches: int) -> List[Dict]:
    """Divide los tickers agrupados en N lotes, asegurando que cada ticker 
       permanezca en un único lote."""
    
    tickers = list(grouped_params_dict.keys())
    
    # Inicializa los N lotes (cada lote es un diccionario)
    batches = [{} for _ in range(num_batches)]
    
    # Asigna cada ticker (y sus parámetros) a un lote en rotación (round-robin)
    for i, ticker in enumerate(tickers):
        batch_index = i % num_batches
        batches[batch_index][ticker] = grouped_params_dict[ticker]
        
    return batches

# ==============================================================================
#                      2. FUNCIONES ASÍNCRONAS 
# ==============================================================================

async def fetch_and_process(session: aiohttp.ClientSession, params, api_semaphore: asyncio.Semaphore):
    """
    Función de tarea individual que realiza el Fetch (A) y el Process (B).
    Devuelve un DataFrame si tiene éxito, o None si hay fallo.
    """
    (ticker, start_date, end_date) = params
    
    if ticker == 'VERO':
        print(" ===== fetch_and_process =======")
        print( (ticker, start_date, end_date))
    
    # Usamos el semáforo para limitar la concurrencia a la API
    async with api_semaphore:
        try:
            # Función A
            raw_data = await fetch_data_1_min(session, params)
            if not raw_data:
                return None

            # Función B
            #processed_data = utils.process_data_30_minutes(raw_data)
            processed_data = utils.process_data_minutes(raw_data)
            
            # Añadir información clave para la consolidación
            processed_data['ticker'] = ticker
            return processed_data
            
        except Exception as e:
            print(f"🔥 Error en FETCH/PROCESS para {ticker} ({start_date}): {e}")
            return None

# ==============================================================================
#                      3. WORKER DE MULTIPROCESAMIENTO (Nivel Síncrono)
# ==============================================================================

def process_batch_worker(batch_id: int, ticker_batch: Dict, connectionParams: Dict):
    """
    Función síncrona que corre en un proceso separado (un núcleo de CPU).
    Contiene el ciclo de asyncio.
    """
    print(f"\n[Worker {batch_id}] Iniciando. Contiene {len(ticker_batch)} Tickers.")
    
    # El worker necesita un ciclo de eventos de asyncio
    asyncio.run(
        async_batch_runner(batch_id, ticker_batch, connectionParams)
    )
    
async def async_batch_runner(batch_id: int, ticker_batch: Dict, connectionParams: Dict):
    
    # Semáforos para el control de recursos
    api_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS) # Límite de peticiones a la API
    ingest_semaphore = asyncio.Semaphore(3) # Límite de conexiones a la DB
    
    # Usamos una sesión para todo el worker
    async with aiohttp.ClientSession() as session:
        
        # Iterar secuencialmente sobre CADA Ticker dentro del batch
        for ticker, ticker_param_list in ticker_batch.items():
            
            # 1. Ejecutar A y B concurrentemente para UN SOLO Ticker
            fetch_tasks = [
                fetch_and_process(session, params, api_semaphore)
                for params in ticker_param_list
            ]
            
            processed_results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
            
            # 2. Consolidar el DataFrame (Paso B final)
            valid_dataframes = [df for df in processed_results if isinstance(df, pd.DataFrame) and not df.empty]
            
            if not valid_dataframes:
                print(f"  [Worker {batch_id}] ⚠️ Advertencia: No hay datos válidos para {ticker}.")
                continue

            df = pd.concat(valid_dataframes, ignore_index=True)
            
            
         
            # La manipulación de Pandas es crítica
            # df['date_str'] = pd.to_datetime(pd.to_datetime(df["day"])).dt.strftime('%Y-%m-%d') 
            df = df.sort_values(by="time").reset_index(drop=True)
        
        
        
            # ==================== getting previous day close ===========
            """
            there might be case where the data is incomplete for certain dates example: ticker: WW 2025/03/07,...no data until, 2025/07/07
            so If you use df['previous_close'] = df['close'].shift(1) will create errors since the previous element in the dataframe might not be the previous day
            The approach used only get previous element close as previous_close when there is a maximin day difference of 3. To include Weekends or holidays. other  
            """
            
            
            # if len(df) == 1:
            #     df['previous_close'] = df['open']
            # else:
            
            #     # Day difference
            #     day_diff = df['day'].diff().dt.days
                
            #     # Previous close with fallback to current close
            #     df['previous_close'] = df['close'].shift(1).where(day_diff < 4, df['open'])

            # First row fallback
            #df.loc[day_diff.isna(), 'previous_close'] = df['open']
            #df['previous_close'] = df['close'].shift(1) // Do not use this code to get previous day close
            
            df = utils.sync_data_with_prev_day_close(df)
            #=========================getting previous day close ends ===============================
            
            
            df['gap'] = df['open'] - df['previous_close']
            df['gap_perc'] = np.where( df['previous_close'] > 0, (df['open'] - df['previous_close']) *100 / df['previous_close'], 0 )
            # # Fill NaN gaps with 0 for the first day
            df['market_cap'] = -1
            df['stock_float'] = -1
            df['daily_200_sma'] = -1
            df['daily_range'] =   np.where( df['previous_close'] > 0, (df['high_mh'] - df['previous_close']) , 0 )
            df['day_range_perc'] = np.where( df['previous_close'] > 0, (df['high_mh'] - df['previous_close']) *100 / df['previous_close'], 0 )
            

            DSMA_200 = pd.DataFrame(utils.get_200_DSMA(ticker))
            if DSMA_200.empty == False:
                DSMA_200.rename(columns={'Daily_200_SMA': 'daily_200_sma', "date_s":'date_str'}, inplace=True)
                DSMA_200 = DSMA_200.drop(columns=['timestamp'])
                df = (
                    df
                    .merge(
                        DSMA_200,
                        on='date_str',
                        how='inner',          # 👈 only equal dates
                        suffixes=('', '_new')
                    )
                )

                df['daily_200_sma'] = df['daily_200_sma_new'].combine_first(df['daily_200_sma'])
                df = df.drop(columns='daily_200_sma_new')

           
            df = df.drop(columns=['day'])
            df  = df.fillna(-1)
            df.fillna(
                    {
                        'gap_perc': -1,           # Replace NaN in 'volume' column with -1
                        'daily_range': -1,           # Replace NaN in 'volume' column with -1
                        'previous_close': -1,           # Replace NaN in 'volume' column with -1
                        'previous_close': -1,           # Replace NaN in 'volume' column with -1
                        'market_cap': 0,         # Replace NaN in 'market_cap' column with 0 (just for example)
                        'daily_200_sma':-1
                    },
                    inplace=True
                )
            _df = df[[  "ticker",
                        "date_str",
                        "gap",
                        "gap_perc", 
                        "daily_range",
                        "previous_close",
                        "high",
                        "low",
                        "volume", 
                        "open",
                        "close",
                        'premarket_volume',  
                        "market_hours_volume",
                        "high_mh",
                        "high_pm",
                        "low_pm",
                        "highest_in_pm",
                        "time",
                        "day_range_perc",
                        "ah_open",
                        "ah_close",
                        "ah_high",
                        "ah_low",   
                        'ah_range',         
                        'ah_range_perc' ,  
                        "ah_volume",
                        "market_cap",
                        "stock_float",
                        "daily_200_sma",
                        "split_date_str", 
                        'split_adjust_factor',
                        'high_pm_time'
                        ]]
            
            print(_df[['ticker','date_str','high_pm_time']])
            
        
            _df = _df[~((_df['open'] == 0) & (_df['close'] == 0))]
            
            # Create a mask that keeps only the first occurrence of each column name
            
          
            # print('============ async_batch_runner =============')
            # print(_df[["ticker",
            #             "date_str",
            #             "gap",
            #             "gap_perc", 
            #             "daily_range",
            #             "previous_close",]])
            #df = pd.DataFrame([])
            
            # 3. Función C (Ingestión por Lote) - Se ejecuta solo una vez por ticker
            
            ## IMPORTANT!!!!!!!!!!!!!!!!!!!!!!!! the 2 next line inject the data into the database
            
            async with ingest_semaphore:
                await ingest_data(session, _df, connectionParams)
                
                
            # La memoria de ticker_combined_df se libera antes de pasar al siguiente ticker.

# ==============================================================================
#                      4. FUNCIÓN PRINCIPAL DE LANZAMIENTO
# ==============================================================================

def main_multiprocess_pipeline(parameters: List[Tuple], num_processes: int, connectionParams: Dict):
    
    print(f"Iniciando pipeline con {len(parameters)} tareas a procesar usando {num_processes} procesos...")
    
    utils.update_stock_list_DB(connectionParams)
    
    # ⏱️ 1. CAPTURAR EL TIEMPO DE INICIO
    start_time = time.perf_counter()
    
    # 1. Agrupar por ticker
    grouped_params = group_parameters_by_ticker(parameters)
    
    # 2. Particionar en N batches
    batches = partition_tickers_into_batches(grouped_params, num_processes)
    
    processes = []
    
    # 3. Lanzar N procesos (uno por batch)
    for i, batch in enumerate(batches):
        if not batch:
            continue
            
        p = multiprocessing.Process(
            target=process_batch_worker, 
            args=(i + 1, batch, connectionParams)
        )
        processes.append(p)
        p.start()
        
    # 4. Esperar a que todos los procesos terminen
    for p in processes:
        p.join()
        
    # ⏱️ 5. CAPTURAR EL TIEMPO FINAL
    end_time = time.perf_counter()
    
    # 6. CALCULAR Y MOSTRAR LA DURACIÓN
    total_duration = end_time - start_time
    
    print("\n==============================================")
    print("✅ Proceso de Multiprocesamiento (A+B+C) COMPLETADO.")
    print(f"⏰ Tiempo total de ejecución: {total_duration:.2f} segundos.")
    print("==============================================")


# params = []
# today = datetime.today()
# start = datetime(today.year - 4, today.month, today.day).strftime('%Y-%m-%d')
# _dates = utils.month_ranges("2024-12-01")
# for item in _dates:
    
#     (date1, date2) = item
#     params.append(("MNTS", date1, date2))

params = helpers.prepare_params_for_fetch(connectionParams)
#print(params)
# for ticker,d1,d2 in params:
#     if ticker == 'VERO':
#         print(f'{ticker}: {d1}, {d2}')
    
main_multiprocess_pipeline(params, 8, connectionParams)


# last_date = {"lastdate": datetime.today().strftime('%Y-%m-%d')}
# ld = pd.DataFrame([last_date])
# ld.to_csv('data_injection/lastdate.csv', index=False)
# xx = pd.read_csv('data_injection/lastdate.csv')






