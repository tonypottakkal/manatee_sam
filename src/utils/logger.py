"""
Logging utilities for the manatee segmentation system.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(name: str = "manatee_segmentation", 
                verbose: bool = False,
                log_file: Optional[Path] = None) -> logging.Logger:
    """
    Set up a logger with appropriate formatting and handlers.
    
    Args:
        name: Logger name
        verbose: If True, set log level to DEBUG, otherwise INFO
        log_file: Optional path to log file
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Set log level
    level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger