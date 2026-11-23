from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    OLLAMA_API_KEY: str = "12345"
    OLLAMA_MODEL: str
    OLLAMA_MODEL_KEEP_ALIVE: int | str | None

    TAVILY_API_KEY: str | None = None

    SEARX_HOST: str
    SEARXNG_MAX_RESULTS: int = 5

    PHOENIX_PROJECT_NAME: str = "udemy-langchain-course"
    PHOENIX_TRACES_ENDPOINT: str = "http://localhost:6006/v1/traces"
    PHOENIX_AUTO_INSTRUMENT: bool = True

SETTINGS = Settings()
