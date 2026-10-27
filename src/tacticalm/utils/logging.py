"""Logging utilities for the project."""
import logging
import sys
from pathlib import Path
from typing import Optional
import os

def setup_logging(level: str = "INFO", log_file: Optional[Path] = None) -> logging.Logger:
    """
    Configure and return a root logger.

    Args:
        level: Logging level (e.g., 'INFO', 'DEBUG').
        log_file: Optional file path to write logs to.

    Returns:
        Configured logger instance.
    """
    env_level = os.getenv("TACTICALM_LOG_LEVEL", level).upper()
    numeric_level = getattr(logging, env_level, logging.INFO)
    
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    logger = logging.getLogger()
    logger.setLevel(numeric_level)
    
    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
        
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name.

    Args:
        name: Name for the logger, typically __name__.

    Returns:
        A logger instance.
    """
    return logging.getLogger(name)
