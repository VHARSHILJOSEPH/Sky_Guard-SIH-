"""
Supabase Client Module for SkyGuard AI Backend.

Uses server-side SUPABASE_SERVICE_ROLE_KEY for all backend FastAPI database operations
to safely bypass Row Level Security (RLS) policies without disabling RLS on tables.
NEVER expose SUPABASE_SERVICE_ROLE_KEY to Vite, React, or frontend client bundles.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv, find_dotenv
from supabase import create_client, Client

logger = logging.getLogger("skyguard.supabase")

# 1. First priority: Load Backend/.env (isolated backend-only environment)
BACKEND_DIR = Path(__file__).resolve().parents[2]
backend_env = BACKEND_DIR / ".env"
if backend_env.exists():
    load_dotenv(backend_env, override=False)

# 2. Second priority: Load root .env
root_env = Path(__file__).resolve().parents[3] / ".env"
if root_env.exists():
    load_dotenv(root_env, override=False)
else:
    load_dotenv(find_dotenv(usecwd=True), override=False)

# 3. Read environment variables
SUPABASE_URL: str = (
    os.getenv("SUPABASE_URL")
    or os.getenv("VITE_SUPABASE_URL")
    or os.getenv("NEXT_PUBLIC_SUPABASE_URL", "")
)

# Server-side service-role secret key (FastAPI / Backend ONLY)
SUPABASE_SERVICE_ROLE_KEY: Optional[str] = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

# Client-side publishable key (fallback if service-role key not yet set)
SUPABASE_PUBLISHABLE_KEY: Optional[str] = (
    os.getenv("SUPABASE_PUBLISHABLE_KEY")
    or os.getenv("SUPABASE_KEY")
    or os.getenv("VITE_SUPABASE_ANON_KEY")
    or os.getenv("NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY")
)

# Select server-side credential: prioritize service-role key for backend writes
if SUPABASE_SERVICE_ROLE_KEY and SUPABASE_SERVICE_ROLE_KEY.strip():
    BACKEND_KEY = SUPABASE_SERVICE_ROLE_KEY.strip()
    KEY_TYPE = "service_role"
    logger.info("[SkyGuard Supabase] Initialized client with server-side service-role key (RLS bypass enabled)")
elif SUPABASE_PUBLISHABLE_KEY and SUPABASE_PUBLISHABLE_KEY.strip():
    BACKEND_KEY = SUPABASE_PUBLISHABLE_KEY.strip()
    KEY_TYPE = "publishable"
    logger.warning(
        "[SkyGuard Supabase] SUPABASE_SERVICE_ROLE_KEY not found in backend .env. "
        "Falling back to publishable key (writes may be blocked by RLS error 42501)."
    )
else:
    raise RuntimeError(
        "Supabase credentials missing. Please set SUPABASE_SERVICE_ROLE_KEY and SUPABASE_URL in Backend/.env"
    )

_client: Optional[Client] = None


def get_supabase() -> Client:
    """Returns the singleton server-side Supabase client."""
    global _client
    if _client is None:
        _client = create_client(SUPABASE_URL, BACKEND_KEY)
    return _client


def is_service_role_active() -> bool:
    """Returns True if the backend is using the server-side service-role credential."""
    return KEY_TYPE == "service_role"


# Export singleton instance for direct import: from app.database.supabase_client import supabase
supabase: Client = get_supabase()
