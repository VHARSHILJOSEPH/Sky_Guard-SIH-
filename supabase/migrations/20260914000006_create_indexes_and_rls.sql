-- =====================================================================
-- Migration 006: Create Performance Indexes and RLS Policies
-- SkyGuard AI — Database Optimization and Security
-- =====================================================================

-- ---------------------------------------------------------------------
-- 1. Performance Indexes
-- ---------------------------------------------------------------------

-- Stations
CREATE INDEX IF NOT EXISTS idx_stations_status ON public.stations(status);
CREATE INDEX IF NOT EXISTS idx_stations_source_type ON public.stations(source_type);
CREATE INDEX IF NOT EXISTS idx_stations_is_simulated ON public.stations(is_simulated);

-- Sensor Readings
CREATE INDEX IF NOT EXISTS idx_sensor_readings_station_time ON public.sensor_readings(station_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_sensor_readings_timestamp ON public.sensor_readings(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_sensor_readings_source ON public.sensor_readings(source);
CREATE INDEX IF NOT EXISTS idx_sensor_readings_session_id ON public.sensor_readings(session_id);
CREATE INDEX IF NOT EXISTS idx_sensor_readings_source_time ON public.sensor_readings(source, timestamp DESC);

-- Anomalies
CREATE INDEX IF NOT EXISTS idx_anomalies_station_time ON public.anomalies(station_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_anomalies_timestamp ON public.anomalies(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_anomalies_status ON public.anomalies(status);
CREATE INDEX IF NOT EXISTS idx_anomalies_severity ON public.anomalies(severity);
CREATE INDEX IF NOT EXISTS idx_anomalies_model_version ON public.anomalies(model_version);
CREATE INDEX IF NOT EXISTS idx_anomalies_session_id ON public.anomalies(session_id);

-- Station Health
CREATE INDEX IF NOT EXISTS idx_station_health_station_time ON public.station_health(station_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_station_health_status ON public.station_health(status);

-- Model Versions
CREATE INDEX IF NOT EXISTS idx_model_versions_model_name ON public.model_versions(model_name);

-- ---------------------------------------------------------------------
-- 2. Row Level Security (RLS)
-- ---------------------------------------------------------------------

ALTER TABLE public.stations ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.sensor_readings ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.anomalies ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.station_health ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.model_versions ENABLE ROW LEVEL SECURITY;

-- Allow public read access to operational tables
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies WHERE tablename = 'stations' AND policyname = 'Allow public read access to stations'
    ) THEN
        CREATE POLICY "Allow public read access to stations" ON public.stations FOR SELECT USING (true);
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_policies WHERE tablename = 'sensor_readings' AND policyname = 'Allow public read access to sensor_readings'
    ) THEN
        CREATE POLICY "Allow public read access to sensor_readings" ON public.sensor_readings FOR SELECT USING (true);
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_policies WHERE tablename = 'anomalies' AND policyname = 'Allow public read access to anomalies'
    ) THEN
        CREATE POLICY "Allow public read access to anomalies" ON public.anomalies FOR SELECT USING (true);
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_policies WHERE tablename = 'station_health' AND policyname = 'Allow public read access to station_health'
    ) THEN
        CREATE POLICY "Allow public read access to station_health" ON public.station_health FOR SELECT USING (true);
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_policies WHERE tablename = 'model_versions' AND policyname = 'Allow public read access to model_versions'
    ) THEN
        CREATE POLICY "Allow public read access to model_versions" ON public.model_versions FOR SELECT USING (true);
    END IF;
END
$$;
