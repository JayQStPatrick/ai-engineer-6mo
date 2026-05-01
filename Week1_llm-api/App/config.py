from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str
    default_model: str = "gpt-4o-mini"
    max_retries: int = 3
    request_timeout: int = 30

    class Config:
        env_file = ".env"


settings = Settings()
