"""Application settings loaded from environment variables (.env in development)."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "llm-stock-test-refinement"
    database_url: str = "postgresql+psycopg://stock:stock@localhost:5432/stock"

    # LLM providers (used by the test layer)
    # Closed model: any OpenAI-compatible chat completions API.
    llm_api_key: str = ""
    llm_api_model: str = "gpt-4o-mini"
    llm_api_base_url: str = "https://api.openai.com/v1"
    # Open models served locally by Ollama.
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    # Shared generation settings.
    llm_temperature: float = 0.0
    llm_max_tokens: int = 2048
    llm_timeout_seconds: float = 120.0
    llm_max_retries: int = 3


@lru_cache
def get_settings() -> Settings:
    return Settings()
