-- =====================================================================
-- Migration 001: Create and Seed Stations Table
-- SkyGuard AI — Persistent Stations Source of Truth
-- =====================================================================

CREATE TABLE IF NOT EXISTS public.stations (
    station_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    state TEXT,
    district TEXT,
    source_type TEXT NOT NULL DEFAULT 'IMD_AWS_REFERENCE',
    status TEXT NOT NULL DEFAULT 'ACTIVE',
    is_simulated BOOLEAN NOT NULL DEFAULT false,
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now()),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now())
);

-- Ensure all required columns exist if table was previously created
ALTER TABLE public.stations ADD COLUMN IF NOT EXISTS state TEXT;
ALTER TABLE public.stations ADD COLUMN IF NOT EXISTS district TEXT;
ALTER TABLE public.stations ADD COLUMN IF NOT EXISTS source_type TEXT NOT NULL DEFAULT 'IMD_AWS_REFERENCE';
ALTER TABLE public.stations ADD COLUMN IF NOT EXISTS is_simulated BOOLEAN NOT NULL DEFAULT false;
ALTER TABLE public.stations ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now());

-- Seed Official IMD AWS Reference Stations and Indicative Stations
INSERT INTO public.stations (station_id, name, latitude, longitude, state, district, source_type, status, is_simulated)
VALUES
    ('43189', 'Vijayawada (AWS014)', 16.5062, 80.6480, 'Andhra Pradesh', 'NTR', 'IMD_AWS_REFERENCE', 'ACTIVE', false),
    ('43150', 'Visakhapatnam (AWS008)', 17.6868, 83.2185, 'Andhra Pradesh', 'Visakhapatnam', 'IMD_AWS_REFERENCE', 'ACTIVE', false),
    ('43245', 'Tirupati (AWS021)', 13.6288, 79.4192, 'Andhra Pradesh', 'Tirupati', 'IMD_AWS_REFERENCE', 'ACTIVE', false),
    ('AWS-101', 'New Delhi', 28.6139, 77.2090, 'Delhi', 'New Delhi', 'SIMULATED_INDICATIVE', 'ACTIVE', true),
    ('AWS-102', 'Jaipur', 26.9124, 75.7873, 'Rajasthan', 'Jaipur', 'SIMULATED_INDICATIVE', 'ACTIVE', true),
    ('AWS-103', 'Lucknow', 26.8467, 80.9462, 'Uttar Pradesh', 'Lucknow', 'SIMULATED_INDICATIVE', 'ACTIVE', true),
    ('AWS-104', 'Hyderabad', 17.3850, 78.4867, 'Telangana', 'Hyderabad', 'SIMULATED_INDICATIVE', 'ACTIVE', true),
    ('AWS-105', 'Ahmedabad', 23.0225, 72.5714, 'Gujarat', 'Ahmedabad', 'SIMULATED_INDICATIVE', 'ACTIVE', true),
    ('AWS-106', 'Mumbai', 19.0760, 72.8777, 'Maharashtra', 'Mumbai', 'SIMULATED_INDICATIVE', 'ACTIVE', true),
    ('AWS-107', 'Bhopal', 23.2599, 77.4126, 'Madhya Pradesh', 'Bhopal', 'SIMULATED_INDICATIVE', 'ACTIVE', true),
    ('AWS-108', 'Nagpur', 21.1458, 79.0882, 'Maharashtra', 'Nagpur', 'SIMULATED_INDICATIVE', 'ACTIVE', true),
    ('AWS-109', 'Vijayawada', 16.5062, 80.6480, 'Andhra Pradesh', 'NTR', 'SIMULATED_INDICATIVE', 'ACTIVE', true),
    ('AWS-110', 'Bengaluru', 12.9716, 77.5946, 'Karnataka', 'Bengaluru Urban', 'SIMULATED_INDICATIVE', 'ACTIVE', true),
    ('AWS-111', 'Chennai', 13.0827, 80.2707, 'Tamil Nadu', 'Chennai', 'SIMULATED_INDICATIVE', 'ACTIVE', true),
    ('AWS-112', 'Kolkata', 22.5726, 88.3639, 'West Bengal', 'Kolkata', 'SIMULATED_INDICATIVE', 'ACTIVE', true),
    ('AWS-113', 'Bhubaneswar', 20.2961, 85.8245, 'Odisha', 'Khordha', 'SIMULATED_INDICATIVE', 'ACTIVE', true),
    ('AWS-114', 'Patna', 25.5941, 85.1376, 'Bihar', 'Patna', 'SIMULATED_INDICATIVE', 'ACTIVE', true),
    ('AWS-115', 'Ranchi', 23.3441, 85.3096, 'Jharkhand', 'Ranchi', 'SIMULATED_INDICATIVE', 'ACTIVE', true),
    ('AWS-116', 'Guwahati', 26.1445, 91.7362, 'Assam', 'Kamrup Metropolitan', 'SIMULATED_INDICATIVE', 'ACTIVE', true),
    ('AWS-117', 'Dehradun', 30.3165, 78.0322, 'Uttarakhand', 'Dehradun', 'SIMULATED_INDICATIVE', 'ACTIVE', true),
    ('AWS-118', 'Srinagar', 34.0837, 74.7973, 'Jammu & Kashmir', 'Srinagar', 'SIMULATED_INDICATIVE', 'ACTIVE', true),
    ('AWS-119', 'Pune', 18.5204, 73.8567, 'Maharashtra', 'Pune', 'SIMULATED_INDICATIVE', 'ACTIVE', true),
    ('AWS-120', 'Thiruvananthapuram', 8.5241, 76.9366, 'Kerala', 'Thiruvananthapuram', 'SIMULATED_INDICATIVE', 'ACTIVE', true),
    ('HYD_AWS_01', 'Hyderabad Demo Station', 17.3850, 78.4867, 'Telangana', 'Hyderabad', 'SIMULATED_INDICATIVE', 'ACTIVE', true),
    ('BLR_AWS_02', 'Bengaluru Demo Station', 12.9716, 77.5946, 'Karnataka', 'Bengaluru Urban', 'SIMULATED_INDICATIVE', 'ACTIVE', true),
    ('DEL_AWS_03', 'Delhi Demo Station', 28.6139, 77.2090, 'Delhi', 'New Delhi', 'SIMULATED_INDICATIVE', 'ACTIVE', true)
ON CONFLICT (station_id) DO UPDATE SET
    name = EXCLUDED.name,
    latitude = EXCLUDED.latitude,
    longitude = EXCLUDED.longitude,
    state = EXCLUDED.state,
    district = EXCLUDED.district,
    source_type = EXCLUDED.source_type,
    status = EXCLUDED.status,
    is_simulated = EXCLUDED.is_simulated,
    updated_at = timezone('utc'::text, now());
