from __future__ import annotations

from fastapi import Depends, Header, HTTPException, Request

from .config import Settings, get_settings


def get_runtime_settings(request: Request) -> Settings:
    provider = getattr(request.app.state, "settings_provider", get_settings)
    return provider()


def require_api_key(x_api_key: str | None = Header(default=None), settings: Settings = Depends(get_runtime_settings)) -> None:
    if not settings.api_key:
        raise HTTPException(status_code=503, detail="RALLY_API_KEY is not configured")
    if x_api_key is None or x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def get_user_id(x_user_id: str | None = Header(default=None, alias="X-User-ID"), settings: Settings = Depends(get_runtime_settings)) -> str:
    if settings.is_production:
        raise HTTPException(status_code=503, detail="Production identity provider is not configured; user-scoped API is disabled")
    if not x_user_id:
        raise HTTPException(status_code=400, detail="X-User-ID is required in local development")
    return x_user_id
