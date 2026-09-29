-- =====================================================================
-- Migration 002: Create and Standardize Sensor Readings Table
-- SkyGuard AI — Persistent Telemetry Storage
-- =====================================================================

CREATE TABLE IF NOT EXISTS public.sensor_readings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    station_id TEXT NOT NULL REFERENCES public.stations(station_id) ON DELETE CASCADE,
    timestamp TIMESTAMPTZ NOT NULL,
    temperature_c DOUBLE PRECISION,
    humidity_pct DOUBLE PRECISION,
    pressure_hpa DOUBLE PRECISION,
    wind_speed_ms DOUBLE PRECISION,
    wind_direction_deg DOUBLE PRECISION,
    rainfall_mm DOUBLE PRECISION,
    source TEXT NOT NULL DEFAULT 'IMD_AWS',
    session_id TEXT,
    quality_status TEXT NOT NULL DEFAULT 'VALID',
    raw_payload JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now())
);

-- Ensure all required columns exist if table was previously created
ALTER TABLE public.sensor_readings ADD COLUMN IF NOT EXISTS wind_speed_ms DOUBLE PRECISION;
ALTER TABLE public.sensor_readings ADD COLUMN IF NOT EXISTS wind_direction_deg DOUBLE PRECISION;
ALTER TABLE public.sensor_readings ADD COLUMN IF NOT EXISTS rainfall_mm DOUBLE PRECISION;
ALTER TABLE public.sensor_readings ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT 'IMD_AWS';
ALTER TABLE public.sensor_readings ADD COLUMN IF NOT EXISTS session_id TEXT;
ALTER TABLE public.sensor_readings ADD COLUMN IF NOT EXISTS quality_status TEXT NOT NULL DEFAULT 'VALID';
ALTER TABLE public.sensor_readings ADD COLUMN IF NOT EXISTS raw_payload JSONB;
ALTER TABLE public.sensor_readings ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now());
