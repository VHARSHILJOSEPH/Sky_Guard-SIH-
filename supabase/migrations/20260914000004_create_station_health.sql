-- =====================================================================
-- Migration 004: Create and Standardize Station Health Table
-- SkyGuard AI — Persistent Station Availability and Quality Auditing
-- =====================================================================

CREATE TABLE IF NOT EXISTS public.station_health (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    station_id TEXT NOT NULL REFERENCES public.stations(station_id) ON DELETE CASCADE,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now()),
    availability DOUBLE PRECISION NOT NULL DEFAULT 100.0,
    expected_observations INTEGER NOT NULL DEFAULT 0,
    received_observations INTEGER NOT NULL DEFAULT 0,
    missing_observations INTEGER NOT NULL DEFAULT 0,
    anomaly_frequency DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    false_alarm_metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    health_score DOUBLE PRECISION NOT NULL DEFAULT 100.0,
    status TEXT NOT NULL DEFAULT 'HEALTHY',
    issues JSONB NOT NULL DEFAULT '[]'::jsonb,
    last_seen TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now()),
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now()),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now())
);

-- Ensure all required columns exist if table was previously created
ALTER TABLE public.station_health ADD COLUMN IF NOT EXISTS availability DOUBLE PRECISION NOT NULL DEFAULT 100.0;
ALTER TABLE public.station_health ADD COLUMN IF NOT EXISTS expected_observations INTEGER NOT NULL DEFAULT 0;
ALTER TABLE public.station_health ADD COLUMN IF NOT EXISTS received_observations INTEGER NOT NULL DEFAULT 0;
ALTER TABLE public.station_health ADD COLUMN IF NOT EXISTS missing_observations INTEGER NOT NULL DEFAULT 0;
ALTER TABLE public.station_health ADD COLUMN IF NOT EXISTS anomaly_frequency DOUBLE PRECISION NOT NULL DEFAULT 0.0;
ALTER TABLE public.station_health ADD COLUMN IF NOT EXISTS false_alarm_metrics JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE public.station_health ADD COLUMN IF NOT EXISTS health_score DOUBLE PRECISION NOT NULL DEFAULT 100.0;
ALTER TABLE public.station_health ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'HEALTHY';
ALTER TABLE public.station_health ADD COLUMN IF NOT EXISTS issues JSONB NOT NULL DEFAULT '[]'::jsonb;
ALTER TABLE public.station_health ADD COLUMN IF NOT EXISTS last_seen TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now());
ALTER TABLE public.station_health ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now());
