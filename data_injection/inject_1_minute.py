

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

print("--- hello 1 minute ---")

connectionParams ={}
connectionParams['POSTGREST_H'] = 'localhost'  # os.getenv("POSTGREST_H", "none")
connectionParams['POSTGREST_P'] = '3030' # os.getenv("POSTGREST_P", "none")
connectionParams['POSTGREST_TOKEN'] =  os.getenv("POSTGREST_TOKEN", "none")

    
MAX_CONCURRENT_REQUESTS = 100

# ==============================================================================
#                      2. FUNCIONES ASÍNCRONAS 
# ==============================================================================

async def fetch_1_minute(session: aiohttp.ClientSession, params, api_semaphore: asyncio.Semaphore):
    """
    Función de tarea individual que realiza el Fetch (A) y el Process (B).
    Devuelve un DataFrame si tiene éxito, o None si hay fallo.
    """
    (ticker, start_date, end_date) = params
    
    # Usamos el semáforo para limitar la concurrencia a la API
    async with api_semaphore:
        try:
            # Función A
            raw_data = await helpers.fetch_data_minutes_bar(session, params, multiplier=1)
            if not raw_data:
                return None

         
            processed_data  = pd.DataFrame(raw_data)
            processed_data.rename(columns={'o': 'open', 'c': 'close', 'h': 'high', 'l': 'low', 'v': 'volume', 't': 'time', 'vw':'vwap'}, inplace=True)
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
                fetch_1_minute(session, params, api_semaphore)
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
            #df['date_str'] = pd.to_datetime(pd.to_datetime(df["day"])).dt.strftime('%Y-%m-%d') 
            df = df.sort_values(by="time").reset_index(drop=True)

            
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




params =  helpers.prepare_parameters_of_fetch_for_first_injection(connectionParams)

print(params)