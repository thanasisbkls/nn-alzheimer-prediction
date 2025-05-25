"""
Logging utilities

This module is used to configure the logging system for the entire application.
"""

import logging
import os
from typing import Optional


def setup_logger(name: str, log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with console and optional file output
    
    Args:
        name: Logger name
        log_file: Optional log file path
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    # Create logger instance
    logger = logging.getLogger(name)
    # Set minimum logging level
    logger.setLevel(level)
    
    # Clear existing handlers to avoid duplicate messages
    logger.handlers.clear()
    
    # Create Console handler to print to console
    console_handler = logging.StreamHandler()
    # Set the same logging level as the main logger
    console_handler.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create File handler (optional)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        # Create File handler to write to file
        file_handler = logging.FileHandler(log_file)
        # Set the same logging level as the main logger
        file_handler.setLevel(level)
        # Apply same formater
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger 