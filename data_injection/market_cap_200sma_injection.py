import asyncio
import aiohttp
from typing import List, Dict, Any
import os
import sys
sys.path.insert(0, os.path.abspath("."))
from utils import utils
import pandas as pd
import json
# LÍMITES DE CONCURRENCIA
# Usado para las peticiones de fetch_dependency_1 y fetch_dependency_2.
MAX_CONCURRENT_API_FETCHES = 20

# Usado para las peticiones a PostgREST (Fetch NULL y UPSERT).
MAX_CONCURRENT_DB_OPS = 20

TICKERS_TO_PROCESS = ["TSLA"]

connectionParams ={}
connectionParams['POSTGREST_H'] = 'localhost'  # os.getenv("POSTGREST_H", "none")
connectionParams['POSTGREST_P'] = '3030' # os.getenv("POSTGREST_P", "none")
connectionParams['POSTGREST_TOKEN'] =  os.getenv("POSTGREST_TOKEN", "none")


# de llamadas a la API (por ejemplo, para obtener Market Cap y 200 SMA).
# Aquí simulamos que obtenemos el Market Cap de una API y el 200 SMA de otra.

async def fetch_dependency_market_cap_data(session: aiohttp.ClientSession, ticker: str, date_str: str):
    
    """Simula la obtención de Market Cap y Stock Float desde una API."""
    # Reemplazar con tu URL y lógica de API real
    return utils.get_float_and_marketcap(ticker, date_str)
    

async def fetch_dependency_sma(session: aiohttp.ClientSession, ticker: str, date_str: str):
    
    """Simula la obtención de Daily 200 SMA desde otra API."""
    # Reemplazar con tu URL y lógica de API real
    return utils.get_200_DSMA_by_date(ticker, date_str)

# ==============================================================================
#                      FUNCIÓN PRINCIPAL DE ENRIQUECIMIENTO
# ==============================================================================

async def fetch_and_enrich_data(
    session: aiohttp.ClientSession, 
    record: Dict, 
    api_semaphore: asyncio.Semaphore
):
    """
    Realiza las dos peticiones dependientes en paralelo (Fase 2) y combina 
    los resultados con el registro original (record).
    """
    ticker = record.get('ticker')
    time_value = record.get('time')
    
    if not ticker or not time_value:
        return None

    # Usamos el semáforo para limitar la concurrencia a la API externa
    async with api_semaphore:
        date_str = record.get('date_str') # Necesario para la API, si es por fecha

        try:
            
            
            # 1. Lanzar ambas peticiones simultáneamente y esperar a que ambas terminen
            results = await asyncio.gather(
                # Dependencia 1: Market Cap y Float
                fetch_dependency_market_cap_data(session, ticker, date_str),
                # Dependencia 2: 200 SMA
                fetch_dependency_sma(session, ticker, date_str),
                return_exceptions=True
            )

            # 2. Verificar si hubo errores en cualquiera de las peticiones
            if any(isinstance(r, Exception) for r in results):
                print(r)
                print(f"🔥 Error en enriquecimiento para {ticker} ({date_str}, {time_value}): {results}")
                return None
            
            # 3. Combinar los resultados
           
            
            # Resultado 1: Métricas de mercado
            # result =  {'date_s':date, 'daily_200_sma': -1, "ticker":ticker}
            #  
            
           
            stock_float = results[0]['stock_float']
            market_cap = results[0]['market_cap']
            
            daily_200_sma = results[1]['daily_200_sma']
            
            
            
            #enriched_data.update(results[0]) 
            
            # Resultado 2: SMA
            
            
            enriched_data =  {"stock_float": stock_float, "market_cap":market_cap, "ticker":ticker, "time":time_value, "date_str": date_str , "daily_200_sma": daily_200_sma}
            
            # 4. Devolver el diccionario completo listo para el UPSERT en la Fase 3
            return enriched_data

        except Exception as e:
            # Captura errores que podrían ocurrir fuera de la llamada HTTP (ej. parsing JSON)
            print(f"🔥 Error desconocido en enriquecimiento para {ticker} ({date_str}, {time_value}): {e}")
            return None

async def fetch_null_sma_single(session: aiohttp.ClientSession, ticker: str, connectionParams: dict, db_semaphore: asyncio.Semaphore) -> List[Dict]:
    """
    Fetches records from stock_data where daily_200_sma is NULL for a single ticker.
    """
    
    POSTGREST_H = connectionParams.get('POSTGREST_H')
    POSTGREST_P = connectionParams.get('POSTGREST_P')
    POSTGREST_TOKEN = connectionParams.get('POSTGREST_TOKEN')
    
    url = (
        f'http://{POSTGREST_H}:{POSTGREST_P}/gappers_pm_and_market_hours?'
        f'ticker=eq.{ticker}&'               # Filtro específico para un ticker
        'daily_200_sma=eq.-1&'             # Filtro: IS NULL
        'select=ticker,time,date_str'#&'       # Optimización: solo columnas clave
        #'limit=5000'                         # Límite por ticker (debería ser suficiente)
    )
    
    print(f"  Fetching NULL SMA records for ticker {url}...")
    
    headers = {
        #"Authorization": f"Bearer {POSTGREST_TOKEN}",
        "Accept": "application/json"
    }
    
    # Usar un semáforo DB para limitar la concurrencia a PostgREST
    async with db_semaphore:
        try:
            async with session.get(url, headers=headers) as response:
                response.raise_for_status() 
                data = await response.json()
                # print(f"  Fetched {len(data)} null SMA records for {ticker}.")
                return data
        except Exception as e:
            print(f"🔥 Error DB fetch para ticker {ticker}: {e}")
            return []


async def ingest_enriched_data(
    session: aiohttp.ClientSession, 
    data_list: List[Dict], 
    connectionParams: Dict, 
    db_semaphore: asyncio.Semaphore
):
    """
    Envía la lista de datos enriquecidos a la base de datos (UPSERT)
    a través del endpoint RPC de PostgREST.
    
    Args:
        session: aiohttp client session.
        data_list: Lista de diccionarios de datos enriquecidos.
        connectionParams: Diccionario con host, puerto y token.
        db_semaphore: Semáforo para limitar la concurrencia a la base de datos.
        
    Returns: El número de filas insertadas/actualizadas (devuelto por la función SQL).
    """
    
  
    
    if not data_list:
        return 0

    POSTGREST_H = connectionParams.get('POSTGREST_H')
    POSTGREST_P = connectionParams.get('POSTGREST_P')
    POSTGREST_TOKEN = connectionParams.get('POSTGREST_TOKEN')

    # Endpoint RPC para llamar a la función de PostgreSQL: upsert_stock_data_from_json
    # PostgREST usa el prefijo /rpc/ para las llamadas a funciones.
    rpc_url = (
        f'http://{POSTGREST_H}:{POSTGREST_P}/rpc/update_stock_data_mc_dsma_float'
    )

    headers = {
        "Authorization": f"Bearer {POSTGREST_TOKEN}",
        # PostgREST espera un JSON simple para RPC
        "Content-Type": "application/json", 
        "Accept": "application/json"
    }

    # 1. Usar el semáforo DB para limitar la concurrencia de ingesta
    async with db_semaphore:
        try:
            # 2. El cuerpo de la petición es directamente la lista de diccionarios
            
           
            df = pd.DataFrame(data_list)
            payload_json = json.dumps( {"json_data": data_list})
            
            async with session.post(rpc_url, headers=headers, data=payload_json) as response:
                
                # Levantar excepción si el código es 4xx o 5xx
                response.raise_for_status()
                
                # 3. La función SQL devuelve un INTEGER (el número de filas procesadas)
                # PostgREST envuelve los resultados de RPC en un array JSON.
                result = await response.json()
                
                print(f"  Ingested batch of {len(data_list)} records. DB response: {response.status}")
                
                if isinstance(result, list) and result:
                    # El resultado de RPC es [4] si se procesaron 4 filas
                    rows_processed = result[0]
                    return rows_processed
                
                # Si la función no devuelve un resultado válido, asumir 0
                return 0

        except aiohttp.ClientResponseError as e:
            # Manejo de errores específicos de HTTP (e.g., Token inválido, Error 500 de la DB)
            try:
                error_details = await e.response.text()
                print(f"🚨 ERROR {e.status} en Ingesta a DB (PostgREST): {e.message}")
                print(f"Detalles del Servidor: {error_details}")
            except:
                print(f"🚨 ERROR {e.status} en Ingesta a DB: {e.message}")
            return 0
            
        except Exception as e:
            print(f"🚨 ERROR Inesperado durante la Ingesta: {e}")
            return 0
        
        
async def process_single_ticker_pipeline(
    session: aiohttp.ClientSession, 
    ticker: str, 
    connectionParams: Dict,
    api_semaphore: asyncio.Semaphore,
    db_semaphore: asyncio.Semaphore
):
    """
    Ejecuta el pipeline completo (Fetch NULL, Enriquecimiento y UPSERT) 
    para un único ticker. Devuelve el número de filas procesadas.
    """
    
    
    
    # 1. FASE 1: Fetch de registros NULL (Usando el semáforo DB)
    # Se utiliza la función fetch_null_sma_single que ya creamos.
    null_records = await fetch_null_sma_single(session, ticker, connectionParams, db_semaphore)
    
    
    
    if not null_records:
        # print(f"  [Pipeline {ticker}] ⚠️ No hay registros NULL para procesar.")
        return 0
    
    # 2. FASE 2: Enriquecimiento Concurrente
    # Creamos las tareas de enriquecimiento para CADA registro NULL
    enrichment_tasks = [
        fetch_and_enrich_data(session, record, api_semaphore)
        for record in null_records
    ]
    
    # Ejecutamos las tareas de enriquecimiento en paralelo, limitadas por api_semaphore
    all_enriched_data = await asyncio.gather(*enrichment_tasks)
    
    # Filtrar resultados válidos
    valid_enriched_data = [d for d in all_enriched_data if d is not None]
    
    

    if not valid_enriched_data:
        print(f"  [Pipeline {ticker}] ⚠️ Enriquecimiento falló para todos los registros.")
        return 0

    # 3. FASE 3: UPSERT (Usando el semáforo DB)
    # Aquí consolidamos los datos y los enviamos a la función de UPSERT
    
    # Nota: Aquí deberías transformar valid_enriched_data en un DataFrame 
    # y aplicar cualquier limpieza/cálculo final antes de llamar a la DB.
    # Usaremos una lista simple para el ejemplo.
    
    # Lote de ingesta (para un solo ticker, es un solo lote)
    
   
    
    total_ingested = await ingest_enriched_data(
        session, 
        valid_enriched_data, # Lista de diccionarios ya enriquecidos
        connectionParams, 
        db_semaphore # Usar el semáforo de DB para la ingesta
    )
    
    
    
    print(f"  [Pipeline {ticker}] ✅ Procesado completo. Filas actualizadas: {0}")
    return total_ingested


async def run_efficient_pipeline(all_tickers: List[str], connectionParams: Dict):
    
    # Semáforos
    api_semaphore = asyncio.Semaphore(MAX_CONCURRENT_API_FETCHES)
    db_semaphore = asyncio.Semaphore(MAX_CONCURRENT_DB_OPS)
    
    start_time = asyncio.get_event_loop().time()
    total_tickers = len(all_tickers)
    
    async with aiohttp.ClientSession() as session:
        
        print(f"Iniciando pipeline para {total_tickers} tickers...")

        # Lanzar una tarea de pipeline completa por CADA ticker
        all_pipeline_tasks = [
            process_single_ticker_pipeline(
                session, 
                ticker, 
                connectionParams, 
                api_semaphore, 
                db_semaphore
            )
            for ticker in all_tickers
        ]

        # Ejecutar todos los pipelines de tickers en paralelo
        # Nota: La concurrencia global está limitada por los semáforos internos.
        results = await asyncio.gather(*all_pipeline_tasks)
        
        # Consolidar el conteo total de filas procesadas
        total_rows_processed = sum(results)
        
    end_time = asyncio.get_event_loop().time()
    
    print("\n==============================================")
    print(f"✅ Proceso COMPLETADO. {total_tickers} tickers procesados.")
    print(f"Total de filas actualizadas/insertadas: {total_rows_processed}")
    print(f"⏰ Tiempo total: {end_time - start_time:.2f} segundos.")
    print("==============================================")
    

def main():
    
    df = utils.gappers_pm_and_market_hours(connectionParams)
  
    
    if df is None or df.empty:
        print('===== please check if the gappers_pm_and_market_hours Matetialized view of created and populated')
        return
   
    """Synchronous entry point to start the asynchronous pipeline."""
    TICKERS_TO_PROCESS =  df['ticker'].to_list()
    
    # 1. Check for necessary dependencies before starting
    if not all(connectionParams.values()):
        print("🚨 ERROR: Please set actual values in CONNECTION_PARAMS before running.")
        return

    print("--- Starting Efficient Async Pipeline ---")
    
    try:
        # 2. Run the main asynchronous function
        asyncio.run(
            run_efficient_pipeline(
                TICKERS_TO_PROCESS, 
                connectionParams
            )
        )
    except KeyboardInterrupt:
        print("\nPipeline stopped by user (Ctrl+C).")
    except Exception as e:
        print(f"\nFATAL ERROR during pipeline execution: {e}")
        

main()