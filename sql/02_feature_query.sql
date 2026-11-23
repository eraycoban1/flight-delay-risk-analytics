-- sql/02_feature_query.sql
-- Purpose: Engineer leak-free pre-flight features strictly adhering to a T-180 min
--          prediction horizon, including 3-hour rolling airport congestion/weather
--          proxy windows, net turnaround buffers, and error-analysis metadata.

CREATE OR REPLACE TABLE model_features AS
WITH valid_flights AS (
    SELECT
        operating_airline,
        flight_number,
        origin,
        dest,
        flight_date,
        EXTRACT(MONTH FROM flight_date) AS flight_month,
        day_of_week,
        CAST(SUBSTRING(crs_dep_time_str, 1, 2) AS INTEGER) AS dep_hour,
        (CAST(SUBSTRING(crs_dep_time_str, 1, 2) AS INTEGER) * 60 + 
         CAST(SUBSTRING(crs_dep_time_str, 3, 2) AS INTEGER)) AS crs_dep_mins,
        (CAST(SUBSTRING(crs_arr_time_str, 1, 2) AS INTEGER) * 60 + 
         CAST(SUBSTRING(crs_arr_time_str, 3, 2) AS INTEGER)) AS crs_arr_mins,
        crs_elapsed_time,
        dep_delay,
        arr_delay,
        tail_number,
        carrier_delay,
        weather_delay,
        nas_delay,
        security_delay,
        late_aircraft_delay,
        CASE WHEN arr_delay >= 15.0 THEN 1 ELSE 0 END AS is_delayed,
        CASE WHEN dep_delay >= 15.0 THEN 1 ELSE 0 END AS is_dep_delayed
    FROM raw_flights
    WHERE cancelled = 0
      AND diverted = 0
      AND COALESCE(duplicate_flag, 'N') = 'N'
      AND arr_delay IS NOT NULL
      AND dep_delay IS NOT NULL
      AND origin IS NOT NULL
      AND dest IS NOT NULL
      AND crs_dep_time_str IS NOT NULL
),

top_10_airports AS (
    SELECT origin
    FROM valid_flights
    GROUP BY origin
    ORDER BY COUNT(*) DESC
    LIMIT 10
),

filtered_top_flights AS (
    SELECT vf.*
    FROM valid_flights vf
    INNER JOIN top_10_airports t10
        ON vf.origin = t10.origin
),

lagged_aircraft AS (
    SELECT
        *,
        LAG(crs_arr_mins) OVER (
            PARTITION BY tail_number, flight_date
            ORDER BY crs_dep_mins
        ) AS prev_crs_arr_mins,
        LAG(arr_delay) OVER (
            PARTITION BY tail_number, flight_date
            ORDER BY crs_dep_mins
        ) AS raw_prev_arr_delay
    FROM filtered_top_flights
),

engineered_features AS (
    SELECT
        -- Chronological & Error Analysis Metadata (Excluded from model X matrix)
        flight_date,
        crs_dep_mins,
        flight_number,
        carrier_delay,
        weather_delay,
        nas_delay,
        security_delay,
        late_aircraft_delay,

        -- Model Predictors
        flight_month,
        operating_airline,
        origin,
        dest,
        day_of_week,
        CASE WHEN day_of_week IN (6, 7) THEN 1 ELSE 0 END AS is_weekend,
        CASE
            WHEN dep_hour BETWEEN 5 AND 8   THEN 'Early_Morning'
            WHEN dep_hour BETWEEN 9 AND 12  THEN 'Morning'
            WHEN dep_hour BETWEEN 13 AND 16 THEN 'Afternoon'
            WHEN dep_hour BETWEEN 17 AND 20 THEN 'Evening'
            ELSE 'Night_RedEye'
        END AS dep_time_block,
        dep_hour AS scheduled_dep_hour,
        crs_elapsed_time AS scheduled_flight_mins,

        -- Static Schedule Congestion
        COUNT(*) OVER (PARTITION BY origin, flight_date) AS origin_daily_flight_volume,
        COUNT(*) OVER (PARTITION BY origin, flight_date, dep_hour) AS origin_hourly_flight_volume,
        COUNT(*) OVER (PARTITION BY origin, dest, flight_date) AS route_daily_frequency,

        -- Rolling 3-Hour Airport Congestion / Weather Proxy Window (Strictly T-360 to T-180 mins)
        -- Captures live ATC ground stops or convective weather building up at the origin hub
        COALESCE(
            AVG(dep_delay) OVER (
                PARTITION BY origin, flight_date
                ORDER BY crs_dep_mins
                RANGE BETWEEN 360 PRECEDING AND 180 PRECEDING
            ),
            0.0
        ) AS origin_rolling_3h_dep_delay,

        COALESCE(
            AVG(is_dep_delayed) OVER (
                PARTITION BY origin, flight_date
                ORDER BY crs_dep_mins
                RANGE BETWEEN 360 PRECEDING AND 180 PRECEDING
            ),
            0.0
        ) AS origin_rolling_3h_delay_rate,

        -- Expanding Historical Carrier-Origin Delay Rate
        COALESCE(
            AVG(is_delayed) OVER (
                PARTITION BY operating_airline, origin
                ORDER BY flight_date, crs_dep_mins
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ),
            0.0
        ) AS carrier_origin_hist_delay_rate,

        -- T-180 Min Aircraft Propagation Cutoff
        CASE
            WHEN tail_number IS NULL OR prev_crs_arr_mins IS NULL OR raw_prev_arr_delay IS NULL THEN 0.0
            WHEN (prev_crs_arr_mins + raw_prev_arr_delay) <= (crs_dep_mins - 180) THEN raw_prev_arr_delay
            ELSE 0.0
        END AS known_prev_aircraft_delay,

        -- Scheduled Turnaround Buffer (capped between 15 and 480 mins)
        CASE
            WHEN tail_number IS NULL OR prev_crs_arr_mins IS NULL THEN 240.0
            ELSE LEAST(GREATEST(crs_dep_mins - prev_crs_arr_mins, 15.0), 480.0)
        END AS scheduled_turnaround_mins,

        is_delayed
    FROM lagged_aircraft
)

SELECT
    -- Metadata columns (for split & error analysis only)
    flight_date,
    crs_dep_mins,
    flight_number,
    carrier_delay,
    weather_delay,
    nas_delay,
    security_delay,
    late_aircraft_delay,

    -- Pre-flight Predictive Features
    flight_month,
    operating_airline,
    origin,
    dest,
    day_of_week,
    is_weekend,
    dep_time_block,
    scheduled_dep_hour,
    scheduled_flight_mins,
    origin_daily_flight_volume,
    origin_hourly_flight_volume,
    route_daily_frequency,
    ROUND(origin_rolling_3h_dep_delay, 2) AS origin_rolling_3h_dep_delay,
    ROUND(origin_rolling_3h_delay_rate, 4) AS origin_rolling_3h_delay_rate,
    ROUND(carrier_origin_hist_delay_rate, 4) AS carrier_origin_hist_delay_rate,
    known_prev_aircraft_delay,
    scheduled_turnaround_mins,
    (scheduled_turnaround_mins - GREATEST(known_prev_aircraft_delay, 0.0)) AS net_turnaround_buffer,
    ROUND(GREATEST(known_prev_aircraft_delay, 0.0) / scheduled_turnaround_mins, 4) AS delay_to_buffer_ratio,
    is_delayed
FROM engineered_features
USING SAMPLE 400000 ROWS (reservoir, 42);