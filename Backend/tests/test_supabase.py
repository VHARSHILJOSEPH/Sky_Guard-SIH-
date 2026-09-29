import os
import sys
from pathlib import Path

# Ensure Backend root is on sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.database.supabase_client import supabase

def test_supabase_stations_query():
    try:
        response = (
            supabase
            .table("stations")
            .select("*")
            .execute()
        )
        print("Query response data:")
        print(response.data)
        assert response is not None
    except Exception as e:
        print(f"Error querying stations table: {e}")

if __name__ == "__main__":
    test_supabase_stations_query()
