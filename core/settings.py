import os
from pydantic_settings import BaseSettings


ENV_FILE = ".env"

class Settings(BaseSettings):

    api_port: str = "8000/api/v1"
    api_host: str = "127.0.0.1" # = os.environ.get("API_PORT", "127.0.0.1")
    name_user: str = "User" #  = os.environ.get("NAME_USER")
    password_user: str = "SDRAbvtg5ehg" # = os.environ.get("PASSWORD_USER")

    class Config:
        env_file = ENV_FILE



def get_settings():
    return Settings()


settings = get_settings()
