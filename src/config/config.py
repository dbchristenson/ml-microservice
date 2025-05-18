"""
This module contains the configuration settings for the application.

It uses Pydantic to manage settings and Loguru for logging. Variables
are collected from an environment file (.env) and are used to configure
this application.

The settings include:
- Model service settings
- Logging settings
- Database connection settings
"""

from loguru import logger
from pydantic import DirectoryPath
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import create_engine


class Settings(BaseSettings):
    """
    Settings for the application.

    Attributes:
        model_config (SettingsConfigDict): Configuration for the model.
        model_path (DirectoryPath): Path to the model directory.
        model_name (str): Name of the model file.
        log_path (DirectoryPath): Path to the log directory.
        svc_log_name (str): Name of the service log file.
        svc_rotation (str): Log rotation policy.
        svc_compression (str): Log compression policy.
        svc_retention (str): Log retention policy.
        db_conn_str (str): Database connection string.
        rent_apartments_table_name (str): Name of the rent apartments table.
    """

    model_config = SettingsConfigDict(
        env_file="src/config/.env",
        env_file_encoding="utf-8",
    )

    # model service
    model_path: DirectoryPath
    model_name: str

    # logging
    log_path: DirectoryPath
    svc_log_name: str
    svc_rotation: str
    svc_compression: str
    svc_retention: str

    # db
    db_conn_str: str
    rent_apartments_table_name: str


def configure_logging(log_level: str) -> None:
    """
    Configure logging settings for the application.

    This function sets up the Loguru logger with the specified log path,
    rotation, compression, and retention settings.

    Args:
        log_level (str): The logging level to set for the logger.

    Returns:
        None
    """
    logger.add(
        settings.log_path / settings.svc_log_name,
        rotation=settings.svc_rotation,
        compression=settings.svc_compression,
        retention=settings.svc_retention,
        level=log_level,
    )


settings: Settings = Settings()
configure_logging(log_level="INFO")


engine = create_engine(
    settings.db_conn_str,
    connect_args={"check_same_thread": False},
    echo=True,
)
