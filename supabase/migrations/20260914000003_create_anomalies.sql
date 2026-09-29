-- =====================================================================
-- Migration 003: Create and Standardize Anomalies Table
-- SkyGuard AI — Persistent Anomaly and Incident Tracking
-- =====================================================================

CREATE TABLE IF NOT EXISTS public.anomalies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    station_id TEXT NOT NULL REFERENCES public.stations(station_id) ON DELETE CASCADE,
    reading_id UUID REFERENCES public.sensor_readings(id) ON DELETE SET NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    anomaly_type TEXT NOT NULL,
    final_status TEXT NOT NULL DEFAULT 'ANOMALY',
    severity TEXT NOT NULL DEFAULT 'LOW',
    anomaly_score DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    confidence DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    evidence JSONB NOT NULL DEFAULT '{}'::jsonb,
    explanation TEXT,
    model_name TEXT NOT NULL DEFAULT 'SkyGuard-LOF-BME280',
    model_version TEXT NOT NULL DEFAULT 'v1.0',
    status TEXT NOT NULL DEFAULT 'DETECTED',
    first_detected TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now()),
    last_detected TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now()),
    occurrence_count INTEGER NOT NULL DEFAULT 1,
    session_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now()),
    resolved_at TIMESTAMPTZ
);

-- Ensure all required columns exist if table was previously created
ALTER TABLE public.anomalies ADD COLUMN IF NOT EXISTS reading_id UUID REFERENCES public.sensor_readings(id) ON DELETE SET NULL;
ALTER TABLE public.anomalies ADD COLUMN IF NOT EXISTS final_status TEXT NOT NULL DEFAULT 'ANOMALY';
ALTER TABLE public.anomalies ADD COLUMN IF NOT EXISTS confidence DOUBLE PRECISION NOT NULL DEFAULT 0.0;
ALTER TABLE public.anomalies ADD COLUMN IF NOT EXISTS evidence JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE public.anomalies ADD COLUMN IF NOT EXISTS model_name TEXT NOT NULL DEFAULT 'SkyGuard-LOF-BME280';
ALTER TABLE public.anomalies ADD COLUMN IF NOT EXISTS model_version TEXT NOT NULL DEFAULT 'v1.0';
ALTER TABLE public.anomalies ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'DETECTED';
ALTER TABLE public.anomalies ADD COLUMN IF NOT EXISTS first_detected TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now());
ALTER TABLE public.anomalies ADD COLUMN IF NOT EXISTS last_detected TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now());
ALTER TABLE public.anomalies ADD COLUMN IF NOT EXISTS occurrence_count INTEGER NOT NULL DEFAULT 1;
ALTER TABLE public.anomalies ADD COLUMN IF NOT EXISTS session_id TEXT;
ALTER TABLE public.anomalies ADD COLUMN IF NOT EXISTS resolved_at TIMESTAMPTZ;
