from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    OLLAMA_API_KEY: str = "12345"
    OLLAMA_MODEL: str
    OLLAMA_MODEL_KEEP_ALIVE: int | str | None

    PHOENIX_PROJECT_ID: str = "udemy-langchain-course"
    PHOENIX_TRACES_ENDPOINT: str = "http://localhost:6006/v1/traces"
    PHOENIX_AUTO_INSTRUMENT: bool = True

SETTINGS = Settings()
