-- =====================================================================
-- Migration 005: Create and Seed Model Versions Table
-- SkyGuard AI — Persistent ML Model Governance and Provenance
-- =====================================================================

CREATE TABLE IF NOT EXISTS public.model_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL UNIQUE,
    algorithm TEXT NOT NULL,
    dataset_version TEXT NOT NULL,
    feature_schema_version TEXT NOT NULL,
    threshold DOUBLE PRECISION NOT NULL,
    training_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    evaluation_metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    artifact_hash TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now())
);

-- Ensure all required columns exist if table was previously created
ALTER TABLE public.model_versions ADD COLUMN IF NOT EXISTS algorithm TEXT;
ALTER TABLE public.model_versions ADD COLUMN IF NOT EXISTS dataset_version TEXT;
ALTER TABLE public.model_versions ADD COLUMN IF NOT EXISTS feature_schema_version TEXT;
ALTER TABLE public.model_versions ADD COLUMN IF NOT EXISTS threshold DOUBLE PRECISION;
ALTER TABLE public.model_versions ADD COLUMN IF NOT EXISTS training_metadata JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE public.model_versions ADD COLUMN IF NOT EXISTS evaluation_metrics JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE public.model_versions ADD COLUMN IF NOT EXISTS artifact_hash TEXT;

-- Seed Reference SkyGuard LOF BME280 v1.0 Production Model
INSERT INTO public.model_versions (
    model_name,
    model_version,
    algorithm,
    dataset_version,
    feature_schema_version,
    threshold,
    training_metadata,
    evaluation_metrics,
    artifact_hash
)
VALUES (
    'SkyGuard-LOF-BME280',
    'LOF_BME280_v1.0',
    'LocalOutlierFactor',
    'benchmark_v1.0_clean',
    'features_v1.0_temp_hum_pres_wind',
    1.50,
    '{
        "contamination": 0.05,
        "n_neighbors": 20,
        "metric": "minkowski",
        "training_data_source": "IMD_AWS_REFERENCE_HISTORICAL",
        "calibration_split": "validation_only_chronological"
    }'::jsonb,
    '{
        "precision": 0.952,
        "recall": 0.941,
        "f1": 0.946,
        "specificity": 0.988,
        "fpr": 0.012,
        "fnr": 0.059,
        "balanced_accuracy": 0.965,
        "mcc": 0.912
    }'::jsonb,
    'sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
)
ON CONFLICT (model_version) DO UPDATE SET
    algorithm = EXCLUDED.algorithm,
    dataset_version = EXCLUDED.dataset_version,
    feature_schema_version = EXCLUDED.feature_schema_version,
    threshold = EXCLUDED.threshold,
    training_metadata = EXCLUDED.training_metadata,
    evaluation_metrics = EXCLUDED.evaluation_metrics,
    artifact_hash = EXCLUDED.artifact_hash;
