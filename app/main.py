"""FastAPI entry point."""

from fastapi import FastAPI

from app.api import health
from app.config import get_settings

app = FastAPI(title=get_settings().app_name)
app.include_router(health.router)
