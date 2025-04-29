from loguru import logger
from pydantic import DirectoryPath, FilePath
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings for the application."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # model service
    data_file_name: FilePath
    model_path: DirectoryPath
    model_name: str

    # logging
    log_path: DirectoryPath
    svc_log_name: str
    svc_rotation: str
    svc_compression: str
    svc_retention: str


settings: Settings = Settings()
logger.add(
    settings.log_path / settings.svc_log_name,
    rotation=settings.svc_rotation,
    compression=settings.svc_compression,
    retention=settings.svc_retention,
)
