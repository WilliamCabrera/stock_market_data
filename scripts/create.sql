-- Create the database
-- 


CREATE ROLE read_only_all;
ALTER ROLE read_only_all WITH NOSUPERUSER INHERIT NOCREATEROLE NOCREATEDB NOLOGIN NOREPLICATION NOBYPASSRLS;
CREATE ROLE read_only_public;
ALTER ROLE read_only_public WITH NOSUPERUSER INHERIT NOCREATEROLE NOCREATEDB NOLOGIN NOREPLICATION NOBYPASSRLS;
CREATE ROLE read_write_all;
ALTER ROLE read_write_all WITH NOSUPERUSER INHERIT NOCREATEROLE NOCREATEDB NOLOGIN NOREPLICATION NOBYPASSRLS;

create role web_anon nologin;
create role web_admin nologin;

--DROP DATABASE IF EXISTS market_data;
--CREATE DATABASE market_data;

-- Connect to the database
--\c market_data;


CREATE TABLE stock_data (
    ticker VARCHAR(10) NOT NULL,
    date_str VARCHAR(20) NOT NULL,
    gap DECIMAL(10, 2) DEFAULT -1,
    gap_perc DECIMAL(10, 2) DEFAULT -1,
    daily_range DECIMAL(10, 2) DEFAULT -1,
    previous_close DECIMAL(10, 2) DEFAULT -1,
    high DECIMAL(20, 2) DEFAULT -1,
    low DECIMAL(20, 2) DEFAULT -1,
    volume DECIMAL(20, 2) DEFAULT -1,
    open DECIMAL(20, 2) DEFAULT -1,
    close DECIMAL(20, 2) DEFAULT -1,
    premarket_volume DECIMAL(20, 2) DEFAULT -1,
    market_hours_volume DECIMAL(20, 2) DEFAULT -1,
    high_mh DECIMAL(20, 2) DEFAULT -1,
    high_pm DECIMAL(20, 2) DEFAULT -1,
    low_pm DECIMAL(20, 2) DEFAULT -1,
    highest_in_PM BOOLEAN,
    time BIGINT NOT NULL,
    market_cap DECIMAL(20, 2) DEFAULT -1,
    daily_200_sma DECIMAL(20, 2) DEFAULT -1,
    stock_float DECIMAL(20, 2) DEFAULT -1,
    day_range_perc DECIMAL(20, 2) DEFAULT -1,
    ah_open DECIMAL(20, 2) DEFAULT -1,
    ah_close DECIMAL(20, 2) DEFAULT -1,
    ah_high DECIMAL(20, 2) DEFAULT -1,
    ah_low DECIMAL(20, 2) DEFAULT -1,
    ah_range DECIMAL(20, 2) DEFAULT -1,
    ah_range_perc DECIMAL(20, 2) DEFAULT -1,
    ah_volume DECIMAL(20, 2) DEFAULT -1,
    split_date_str VARCHAR(20),            -- Optional: To store intraday time if needed,
    split_adjust_factor double precision,
    PRIMARY KEY (ticker, time)
);

CREATE TABLE candles_5m (
    ticker TEXT NOT NULL,
    time BIGINT NOT NULL,   -- epoch ms
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low  DOUBLE PRECISION,
    close DOUBLE PRECISION,
    volume BIGINT,
    PRIMARY KEY (ticker, time)
);

CREATE TABLE ticker_info (
     ticker VARCHAR(10) NOT NULL,
     company_name TEXT NOT NULL,
     stock_market TEXT NOT NULL
);


--- 5 minutes data ---
--- this will be populated only for gappers , not for all tickers. The gapers will be selected from the Materialized view
CREATE TABLE ticker_data_5min(
    ticker        TEXT,
    open          double precision,
    close         double precision,
    high          double precision,
    low           double precision,
    time BIGINT NOT NULL,
    date_str      TEXT,
    volume        double precision,
    vwap          double precision,
    daily_200_SMA double precision,
    PRIMARY KEY (ticker, time)
);

--- after hour gappers data ---
CREATE TABLE stock_data_after_hours (
    ticker VARCHAR(10) NOT NULL,
    open double precision,
    close double precision,
    high double precision,
    low double precision,
    ah_gap_perc double precision,
    ah_gap double precision,
    volume_ah double precision,
    volume_mh double precision,
    time BIGINT NOT NULL,
    date_str VARCHAR(20) NOT NULL,            -- Optional: To store intraday time if needed,
    PRIMARY KEY (ticker, time)  -- Composite primary key
);

CREATE INDEX idx_ticker_time
ON stock_data (ticker, time DESC);

CREATE INDEX idx_ticker_time_5mins
ON ticker_data_5min (ticker, time DESC);

CREATE INDEX idx_ticker_time_afterhours
ON stock_data_after_hours (ticker, time DESC);

grant usage on schema public to web_anon;
grant select on public.stock_data to web_anon;
grant select on public.ticker_info to web_anon;
grant select on public.ticker_data_5min to web_anon;
grant select on public.stock_data_after_hours to web_anon;
grant select on public.candles_5m to web_anon;


grant usage on schema public to web_admin;
grant all on public.stock_data to web_admin;
grant all on public.ticker_info to web_admin;
grant all on public.ticker_data_5min to web_admin;
grant all on public.stock_data_after_hours to web_admin;
grant all on public.candles_5m to web_admin;

CREATE USER willy WITH PASSWORD 'password' SUPERUSER;

--- DROP MATERIALIZED VIEW gappers_pm_and_market_hours;
CREATE  MATERIALIZED VIEW gappers_pm_and_market_hours AS
SELECT *
FROM stock_data
WHERE previous_close >= 0.2
  AND (
       ((open - previous_close) * 100.0 / previous_close) > 10
    OR ((high_pm - previous_close) * 100.0 / previous_close) > 10
    OR day_range_perc > 10
      )
  AND market_cap < 1000000000
   AND previous_close < 10 and open >0 and close > 0
ORDER BY time ASC
WITH NO DATA;

GRANT SELECT ON gappers_pm_and_market_hours TO web_anon;
grant usage on schema public to web_anon;
grant select on public.gappers_pm_and_market_hours to web_anon;
grant select on public.gappers_pm_and_market_hours to web_anon;

--- DROP MATERIALIZED VIEW after_hours_runers
CREATE MATERIALIZED VIEW after_hours_runers AS
SELECT *
FROM stock_data
WHERE open >= 0.2 AND ah_range_perc > 10 AND ah_volume > 100000
ORDER BY time ASC
WITH NO DATA;

GRANT SELECT ON after_hours_runers TO web_anon;



--- refresh gappers_pm_and_market_hours and after_hours_runers
CREATE OR REPLACE FUNCTION refresh_materialized_view()
RETURNS json AS $$
DECLARE
  result json;
BEGIN
  BEGIN
    RAISE NOTICE 'Refreshing materialized views...';
    REFRESH MATERIALIZED VIEW  gappers_pm_and_market_hours;
    REFRESH MATERIALIZED VIEW  after_hours_runers;

    -- If we reach here, everything went well
    result := json_build_object(
      'status', 'success',
      'message', 'Materialized views refreshed successfully',
      'timestamp', now()
    );

  EXCEPTION WHEN OTHERS THEN
    -- Handle any unexpected errors
    result := json_build_object(
      'status', 'error',
      'message', SQLERRM,
      'timestamp', now()
    );
  END;

  RETURN result;
END;
$$ LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER;

ALTER FUNCTION refresh_materialized_view() OWNER TO postgres;
ALTER FUNCTION refresh_materialized_view() SECURITY DEFINER;
ALTER FUNCTION refresh_materialized_view() OWNER TO postgres;
GRANT EXECUTE ON FUNCTION refresh_materialized_view() TO web_anon;

-- get the list of tickers with the latest update date
CREATE OR REPLACE FUNCTION get_latest_ticker_time()
RETURNS TABLE (
    latest_ticker TEXT,
    latest_time BIGINT, -- Asumiendo que 'time' es un timestamp en milisegundos (BIGINT)
    latest_date_str TEXT
)
LANGUAGE sql
AS $$
    SELECT
        t.ticker,
        t.time,
        t.date_str
    FROM
        (
            SELECT
                *,
                -- Asigna el número 1 al registro más reciente para cada ticker
                ROW_NUMBER() OVER(PARTITION BY ticker ORDER BY time DESC) as rn
            FROM
                stock_data
        ) AS t
    WHERE
        -- Filtra para obtener solo el registro más reciente
        t.rn = 1;
$$;

ALTER FUNCTION get_latest_ticker_time() OWNER TO postgres;
ALTER FUNCTION get_latest_ticker_time() SECURITY DEFINER;
ALTER FUNCTION get_latest_ticker_time() OWNER TO postgres;
GRANT EXECUTE ON FUNCTION get_latest_ticker_time() TO web_anon;

-- insert into stock_data
CREATE OR REPLACE FUNCTION insert_stock_data_from_json(p_data JSONB)
RETURNS INTEGER AS $$
DECLARE
    rows_inserted INTEGER;
BEGIN
    -- Usamos una CTE (WITH) para capturar las filas insertadas
    WITH inserted_rows AS (
        INSERT INTO stock_data (
            ticker, date_str, gap, gap_perc, daily_range, previous_close, high, low, 
            volume, open, close, premarket_volume, market_hours_volume, 
            high_mh, high_pm, low_pm, highest_in_PM, time, market_cap, 
            daily_200_sma, stock_float, day_range_perc, ah_open, ah_close, 
            ah_high, ah_low, ah_range, ah_range_perc, ah_volume, split_date_str , split_adjust_factor 
        )
        SELECT 
            ticker, date_str, gap, gap_perc, daily_range, previous_close, high, low, 
            volume, open, close, premarket_volume, market_hours_volume, 
            high_mh, high_pm, low_pm, highest_in_PM, time, market_cap, 
            daily_200_sma, stock_float, day_range_perc, ah_open, ah_close, 
            ah_high, ah_low, ah_range, ah_range_perc, ah_volume
        FROM jsonb_to_recordset(p_data) AS x(
            ticker VARCHAR(10),
            date_str VARCHAR(20),
            gap DECIMAL(10, 2),
            gap_perc DECIMAL(10, 2),
            daily_range DECIMAL(10, 2),
            previous_close DECIMAL(10, 2),
            high DECIMAL(20, 2),
            low DECIMAL(20, 2),
            volume DECIMAL(20, 2),
            open DECIMAL(20, 2),
            close DECIMAL(20, 2),
            premarket_volume DECIMAL(20, 2),
            market_hours_volume DECIMAL(20, 2),
            high_mh DECIMAL(20, 2),
            high_pm DECIMAL(20, 2),
            low_pm DECIMAL(20, 2),
            highest_in_PM BOOLEAN,
             time BIGINT,
            market_cap DECIMAL(20, 2),
            daily_200_sma DECIMAL(20, 2),
            stock_float DECIMAL(20, 2),
            day_range_perc DECIMAL(20, 2),
            ah_open DECIMAL(20, 2),
            ah_close DECIMAL(20, 2),
            ah_high DECIMAL(20, 2),
            ah_low DECIMAL(20, 2),
            ah_range DECIMAL(20, 2),
            ah_range_perc DECIMAL(20, 2),
            ah_volume DECIMAL(20, 2),
            split_date_str VARCHAR(20),
            split_adjust_factor DECIMAL(20, 2)
        )
        -- Si ya existe (ticker + time), no hace nada e ignora la fila
        ON CONFLICT (ticker, time) DO NOTHING
        
        -- Solo retorna datos si la inserción fue exitosa
        RETURNING 1
    )
    -- Contamos cuántas filas devolvió el RETURNING
    SELECT count(*) INTO rows_inserted FROM inserted_rows;

    -- Devolvemos el total
    RETURN rows_inserted;
END;
$$ LANGUAGE plpgsql;

ALTER FUNCTION insert_stock_data_from_json(JSONB) OWNER TO postgres;
ALTER FUNCTION insert_stock_data_from_json(JSONB) SECURITY DEFINER;
ALTER FUNCTION insert_stock_data_from_json(JSONB) OWNER TO postgres;
GRANT EXECUTE ON FUNCTION insert_stock_data_from_json(JSONB) TO web_anon;


-- insert or update into stock_data
CREATE OR REPLACE FUNCTION upsert_stock_data_from_json(p_data JSONB)
RETURNS INTEGER AS $$
DECLARE
    rows_processed INTEGER;
BEGIN
    WITH upserted_rows AS (
        INSERT INTO stock_data (
            ticker, date_str, gap, gap_perc, daily_range, previous_close, high, low, 
            volume, open, close, premarket_volume, market_hours_volume, 
            high_mh, high_pm, low_pm, highest_in_PM, time, market_cap, 
            daily_200_sma, stock_float, day_range_perc, ah_open, ah_close, 
            ah_high, ah_low, ah_range, ah_range_perc, ah_volume, split_date_str, split_adjust_factor 
        )
        SELECT 
            ticker, date_str, gap, gap_perc, daily_range, previous_close, high, low, 
            volume, open, close, premarket_volume, market_hours_volume, 
            high_mh, high_pm, low_pm, highest_in_PM, time, market_cap, 
            daily_200_sma, stock_float, day_range_perc, ah_open, ah_close, 
            ah_high, ah_low, ah_range, ah_range_perc, ah_volume , split_date_str, split_adjust_factor 
        FROM jsonb_to_recordset(p_data) AS x(
            -- Mapeo de tipos, debe coincidir con el JSON de entrada
            ticker VARCHAR(10), date_str VARCHAR(20), gap DECIMAL(10, 2), gap_perc DECIMAL(10, 2), 
            daily_range DECIMAL(10, 2), previous_close DECIMAL(10, 2), high DECIMAL(20, 2), low DECIMAL(20, 2), 
            volume DECIMAL(20, 2), open DECIMAL(20, 2), close DECIMAL(20, 2), premarket_volume DECIMAL(20, 2), 
            market_hours_volume DECIMAL(20, 2), high_mh DECIMAL(20, 2), high_pm DECIMAL(20, 2), low_pm DECIMAL(20, 2), 
            highest_in_PM BOOLEAN,  time BIGINT, market_cap DECIMAL(20, 2), daily_200_sma DECIMAL(20, 2), 
            stock_float DECIMAL(20, 2), day_range_perc DECIMAL(20, 2), ah_open DECIMAL(20, 2), ah_close DECIMAL(20, 2), 
            ah_high DECIMAL(20, 2), ah_low DECIMAL(20, 2), ah_range DECIMAL(20, 2), ah_range_perc DECIMAL(20, 2), 
            ah_volume DECIMAL(20, 2), split_date_str VARCHAR(20), split_adjust_factor DECIMAL(20, 2)
        )
        -- CLÁUSULA DE UPSERT: Si hay conflicto en (ticker, time), actualiza todos los campos.
        ON CONFLICT (ticker, time) DO UPDATE SET
            date_str = EXCLUDED.date_str,
            gap = EXCLUDED.gap,
            gap_perc = EXCLUDED.gap_perc,
            daily_range = EXCLUDED.daily_range,
            previous_close = EXCLUDED.previous_close,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            volume = EXCLUDED.volume,
            open = EXCLUDED.open,
            close = EXCLUDED.close,
            premarket_volume = EXCLUDED.premarket_volume,
            market_hours_volume = EXCLUDED.market_hours_volume,
            high_mh = EXCLUDED.high_mh,
            high_pm = EXCLUDED.high_pm,
            low_pm = EXCLUDED.low_pm,
            highest_in_PM = EXCLUDED.highest_in_PM,
            market_cap = EXCLUDED.market_cap,
            daily_200_sma = EXCLUDED.daily_200_sma,
            stock_float = EXCLUDED.stock_float,
            day_range_perc = EXCLUDED.day_range_perc,
            ah_open = EXCLUDED.ah_open,
            ah_close = EXCLUDED.ah_close,
            ah_high = EXCLUDED.ah_high,
            ah_low = EXCLUDED.ah_low,
            ah_range = EXCLUDED.ah_range,
            ah_range_perc = EXCLUDED.ah_range_perc,
            ah_volume = EXCLUDED.ah_volume,
            split_date_str = EXCLUDED.split_date_str,
            split_adjust_factor = EXCLUDED.split_adjust_factor
        
        -- RETURNING devuelve 1 para filas insertadas Y actualizadas.
        RETURNING 1
    )
    -- Contamos cuántas filas fueron PROCESADAS (insertadas o actualizadas)
    SELECT count(*) INTO rows_processed FROM upserted_rows;

    -- Devolvemos el total de filas procesadas
    RETURN rows_processed;
END;
$$ LANGUAGE plpgsql;

ALTER FUNCTION upsert_stock_data_from_json(JSONB) OWNER TO postgres;
ALTER FUNCTION upsert_stock_data_from_json(JSONB) SECURITY DEFINER;
ALTER FUNCTION upsert_stock_data_from_json(JSONB) OWNER TO postgres;
GRANT EXECUTE ON FUNCTION upsert_stock_data_from_json(JSONB) TO web_anon;


--- update the value for stock_float, market_cap and daily_200_sma
CREATE OR REPLACE FUNCTION update_stock_data_mc_dsma_float(json_data JSONB)
RETURNS void AS $$
DECLARE
    item JSONB;
BEGIN
    -- Loop through JSON array elements
    FOR item IN SELECT * FROM jsonb_array_elements(json_data)
    LOOP
        UPDATE stock_data
        SET
            stock_float    = COALESCE((item->>'stock_float')::BIGINT, stock_float),
            market_cap     = COALESCE((item->>'market_cap')::BIGINT, market_cap),
            daily_200_sma  = COALESCE((item->>'daily_200_sma')::DECIMAL, daily_200_sma)
        WHERE
            ticker = item->>'ticker'
            AND time = (item->>'time')::BIGINT;

    END LOOP;
END;
$$ LANGUAGE plpgsql;

ALTER FUNCTION update_stock_data_mc_dsma_float(JSONB) OWNER TO postgres;
ALTER FUNCTION update_stock_data_mc_dsma_float(JSONB) SECURITY DEFINER;
ALTER FUNCTION update_stock_data_mc_dsma_float(JSONB) OWNER TO postgres;
GRANT EXECUTE ON FUNCTION update_stock_data_mc_dsma_float(JSONB) TO web_anon;


CREATE OR REPLACE FUNCTION get_gappers_list()
RETURNS TABLE (ticker text)
LANGUAGE sql
AS $$
    SELECT DISTINCT ticker
    FROM gappers_pm_and_market_hours
    ORDER BY ticker;
$$;

ALTER FUNCTION get_gappers_list() OWNER TO postgres;
ALTER FUNCTION get_gappers_list() SECURITY DEFINER;
ALTER FUNCTION get_gappers_list() OWNER TO postgres;
GRANT EXECUTE ON FUNCTION get_gappers_list() TO web_anon;


