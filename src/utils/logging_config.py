"""
Logging configuration module for the SQL Chatbot application.
Provides centralized logging configuration with structured logs
and various handlers for different log levels.
"""

import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path


def setup_logging(
    log_level=logging.INFO, 
    log_to_file=True, 
    logs_dir="logs", 
    app_name="sql_chatbot"
):
    """Set up application logging with rotating file handler and console output
    
    Args:
        log_level: The logging level to use (default: INFO)
        log_to_file: Whether to log to files in addition to console (default: True)
        logs_dir: Directory to store log files (default: "logs")
        app_name: Application name for log file naming (default: "sql_chatbot")
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Clear any existing handlers
    if logger.handlers:
        logger.handlers.clear()
    
    # Create console handler with formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # Create formatter
    log_format = "[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] %(message)s"
    formatter = logging.Formatter(log_format)
    console_handler.setFormatter(formatter)
    
    # Add console handler to logger
    logger.addHandler(console_handler)
    
    # Add file handler if enabled
    if log_to_file:
        # Create logs directory if it doesn't exist
        log_dir = Path(logs_dir)
        log_dir.mkdir(exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{app_name}_{timestamp}.log"
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        
        # Add file handler to logger
        logger.addHandler(file_handler)
    
    # Set specific levels for noisy modules
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("streamlit").setLevel(logging.WARNING)
    
    # Log startup message
    logger.info(f"Logging initialized at level {logging.getLevelName(log_level)}")
    if log_to_file:
        logger.info(f"Log file: {log_file}")
    
    return logger


class CustomAdapter(logging.LoggerAdapter):
    """Custom LoggerAdapter to add context to log messages"""
    
    def process(self, msg, kwargs):
        # Add timestamp to all messages
        timestamp = time.time()
        
        # Add extra context to log message
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        
        kwargs['extra'].update({
            'timestamp': timestamp,
            'user_id': getattr(self.extra, 'user_id', None),
            'request_id': getattr(self.extra, 'request_id', None),
        })
        
        return msg, kwargs


def get_logger(name, **kwargs):
    """Get a logger with the given name and extra context
    
    Args:
        name: Logger name (typically __name__)
        **kwargs: Extra context to add to log messages
    
    Returns:
        LoggerAdapter: Logger with added context
    """
    logger = logging.getLogger(name)
    return CustomAdapter(logger, kwargs)


# Exception logging decorator
def log_exceptions(logger):
    """Decorator to log exceptions raised in a function
    
    Args:
        logger: The logger to use
    
    Returns:
        Decorator function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Log exception details
                logger.exception(f"Exception in {func.__name__}: {str(e)}")
                # Re-raise the exception
                raise
        return wrapper
    return decorator