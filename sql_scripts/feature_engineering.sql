CREATE OR REPLACE VIEW processed_engine_data AS
WITH EngineLife AS (
    SELECT 
        engine_id, 
        MAX(cycle) as max_cycle 
    FROM raw_sensor_data 
    GROUP BY engine_id
)
SELECT 
    r.*, 
    (e.max_cycle - r.cycle) as RUL
FROM raw_sensor_data r
JOIN EngineLife e ON r.engine_id = e.engine_id;

SELECT engine_id, cycle, RUL 
FROM processed_engine_data 
WHERE engine_id = 1 
ORDER BY cycle DESC 
LIMIT 10;

CREATE OR REPLACE VIEW feature_engineered_data AS
SELECT 
    *,
    -- Rolling average for Sensor 2
    AVG(s2) OVER(PARTITION BY engine_id ORDER BY cycle ROWS BETWEEN 10 PRECEDING AND CURRENT ROW) as s2_rolling_avg,
    -- Rolling average for Sensor 4
    AVG(s4) OVER(PARTITION BY engine_id ORDER BY cycle ROWS BETWEEN 10 PRECEDING AND CURRENT ROW) as s4_rolling_avg,
    -- Rolling average for Sensor 11
    AVG(s11) OVER(PARTITION BY engine_id ORDER BY cycle ROWS BETWEEN 10 PRECEDING AND CURRENT ROW) as s11_rolling_avg
FROM processed_engine_data;