from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "auditRAG"
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    postgres_user: str = "auditrag"
    postgres_password: str = "auditrag"
    postgres_db: str = "auditrag"
    postgres_host: str = "postgres"
    postgres_port: int = 5432

    qdrant_host: str = "qdrant"
    qdrant_port: int = 6333
    qdrant_url_override: str = "https://68ee8d70-cbc4-49d5-90ef-0e6f69c63340.eu-west-1-0.aws.cloud.qdrant.io"  
    qdrant_api_key: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.6QignS8_MbzUV-HEWkwWV9zyYdMwW2HVddlZg1uCmG0"  # required for Qdrant Cloud

    openai_api_key: str = ""
    embedding_provider: str = "local"  # "local" or "openai"

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

    @property
    def postgres_url(self) -> str:
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def qdrant_url(self) -> str:
        if self.qdrant_url_override:
            return self.qdrant_url_override.rstrip("/")
        return f"http://{self.qdrant_host}:{self.qdrant_port}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
