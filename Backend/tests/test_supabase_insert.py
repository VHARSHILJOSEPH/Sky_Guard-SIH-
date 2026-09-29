"""
SkyGuard AI — Supabase Insert & Retrieval Test Script
Tests inserting ONE test sensor reading:
  station_id: HYD_AWS_01
  temperature: 31.8
  humidity: 61.4
  pressure: 1008.7
  wind: 3.9
  rainfall: 0
  source: DEMO

Then retrieves the row from Supabase and verifies it exists.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys

# Ensure Backend directory is on sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.database.supabase_client import supabase
from app.services.database_service import (
    save_sensor_reading,
    get_latest_readings,
    check_database_health,
)


def run_test():
    print("=" * 60)
    print("SkyGuard AI — Testing Supabase Persistence")
    print("=" * 60)

    # 1. Health Check
    health = check_database_health()
    print(f"\n1. Database Health Check: {health}")
    if health.get("status") != "healthy":
        print("   [!] Supabase cannot be reached. Verify network and SUPABASE_URL in .env")
        return False

    # 2. Test Reading Data
    test_reading = {
        "station_id": "HYD_AWS_01",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "temperature_c": 31.8,
        "humidity_pct": 61.4,
        "pressure_hpa": 1008.7,
        "wind_speed_ms": 3.9,
        "rainfall_mm": 0.0,
        "source": "DEMO",
    }

    print("\n2. Attempting to insert test reading into 'sensor_readings':")
    for k, v in test_reading.items():
        print(f"   - {k}: {v}")

    # 3. Perform Insert
    inserted = save_sensor_reading(
        station_id=test_reading["station_id"],
        timestamp=test_reading["timestamp"],
        temperature_c=test_reading["temperature_c"],
        humidity_pct=test_reading["humidity_pct"],
        pressure_hpa=test_reading["pressure_hpa"],
        wind_speed_ms=test_reading["wind_speed_ms"],
        rainfall_mm=test_reading["rainfall_mm"],
        source=test_reading["source"],
    )

    if not inserted:
        print("\n[FAILED] Failed to insert reading into Supabase.")
        return False

    print(f"\n[SUCCESS] Successfully inserted row with ID: {inserted.get('id')}")

    # 4. Perform Retrieval
    print("\n3. Retrieving latest reading from 'sensor_readings' to verify:")
    latest = get_latest_readings(station_id="HYD_AWS_01", limit=1)

    if not latest:
        print("   [!] Could not retrieve reading from database.")
        return False

    retrieved_row = latest[0]
    print(f"   - ID: {retrieved_row.get('id')}")
    print(f"   - Station: {retrieved_row.get('station_id')}")
    print(f"   - Timestamp: {retrieved_row.get('timestamp')}")
    print(f"   - Temperature: {retrieved_row.get('temperature_c')} °C")
    print(f"   - Humidity: {retrieved_row.get('humidity_pct')} %")
    print(f"   - Pressure: {retrieved_row.get('pressure_hpa')} hPa")
    print(f"   - Source: {retrieved_row.get('source')}")

    # 5. Clean up test row (optional: remove the inserted test row)
    try:
        supabase.table("sensor_readings").delete().eq("id", inserted.get("id")).eq("source", "DEMO").execute()
        print("\n[CLEANUP] Deleted test DEMO row from 'sensor_readings'")
    except Exception as cleanup_err:
        print(f"\n[CLEANUP NOTICE] Could not delete test row: {cleanup_err}")

    print("\n" + "=" * 60)
    print("ALL SUPABASE TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)
