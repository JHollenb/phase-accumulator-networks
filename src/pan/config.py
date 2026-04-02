from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="PAN_",
        case_sensitive=False,
    )

    app_name: str = "pan"
    debug: bool = False
    greeting_target: str = Field(default="world")


@lru_cache
def get_settings() -> Settings:
    return Settings()
