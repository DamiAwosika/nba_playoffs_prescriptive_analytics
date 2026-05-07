from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NBA_", env_file=".env", extra="ignore")

    database_url: str = "sqlite:///./nba.db"
    models_dir: Path = Path("models")
    feature_version: str = "v1"
    balldontlie_api_key: str = ""
    secret_key: str = "dev-insecure-key"

settings = Settings()
