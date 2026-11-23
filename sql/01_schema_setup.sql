-- sql/01_schema_setup.sql
-- Purpose: Ingest raw BTS Form 234 (.asc) files with strict schema enforcement,
--          including post-flight delay cause columns strictly reserved for offline Error Analysis.

CREATE OR REPLACE TABLE raw_flights AS
SELECT
    CAST(column00 AS VARCHAR) AS marketing_airline,
    CAST(column01 AS INTEGER) AS flight_number,
    CAST(column04 AS VARCHAR) AS operating_airline,
    CAST(column06 AS VARCHAR) AS origin,
    CAST(column07 AS VARCHAR) AS dest,
    STRPTIME(CAST(column08 AS VARCHAR), '%Y%m%d')::DATE AS flight_date,
    CAST(column09 AS INTEGER) AS day_of_week,
    LPAD(CAST(column10 AS VARCHAR), 4, '0') AS crs_dep_time_str,
    LPAD(CAST(column13 AS VARCHAR), 4, '0') AS crs_arr_time_str,
    CAST(column16 AS INTEGER) AS cancelled,
    CAST(column17 AS INTEGER) AS diverted,
    CAST(column18 AS DOUBLE)  AS crs_elapsed_time,
    CAST(column20 AS DOUBLE)  AS dep_delay,
    CAST(column21 AS DOUBLE)  AS arr_delay,
    NULLIF(TRIM(CAST(column25 AS VARCHAR)), '') AS tail_number,
    -- Delay cause breakdown (Columns 31-35 in 1-based BTS spec -> indices 30-34)
    -- Strictly excluded from model features; used ONLY for post-hoc Error Analysis
    COALESCE(CAST(column30 AS DOUBLE), 0.0) AS carrier_delay,
    COALESCE(CAST(column31 AS DOUBLE), 0.0) AS weather_delay,
    COALESCE(CAST(column32 AS DOUBLE), 0.0) AS nas_delay,
    COALESCE(CAST(column33 AS DOUBLE), 0.0) AS security_delay,
    COALESCE(CAST(column34 AS DOUBLE), 0.0) AS late_aircraft_delay,
    CAST(column70 AS VARCHAR) AS duplicate_flag
FROM read_csv(
    'data/raw/*.asc',
    delim = '|',
    header = false,
    auto_detect = true,
    null_padding = true
);