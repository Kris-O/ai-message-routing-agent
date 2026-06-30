from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    ollama_base_url: str = "http://ollama:11434"
    ollama_api_key: str = "ollama"
    ollama_model: str = "hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:UD-Q4_K_XL"
    # Generous timeout: a 4B on CPU spends ~60-90s on the first request ingesting the ~1800-token system
    # prompt; later requests reuse the cached prefix and finish in a few seconds.
    request_timeout: float = 120.0
    smtp_host: str = "mailhog"
    smtp_port: int = 1025
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_starttls: bool = False
    mail_from: str = "router@example.com"
    rate_limit: str = "300/minute"
    rate_limit_enabled: bool = True

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
