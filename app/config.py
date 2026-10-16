"""Application settings loaded from environment variables (.env in development)."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "llm-stock-test-refinement"
    database_url: str = "postgresql+psycopg://stock:stock@localhost:5432/stock"

    # LLM providers (used by the test layer)
    llm_api_key: str = ""
    llm_api_model: str = ""
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
