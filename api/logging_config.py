"""
Centralized logging configuration.

Sets up structured logging with timestamps, levels, and module names.
Called once at app startup. After this, every module just imports the
standard logging library and uses logger = logging.getLogger(__name__).
"""
import logging
import sys
from typing import Optional


def setup_logging(level: str = "INFO") -> None:
    """
    Configure root logger with structured format.
    
    Output format:
      2026-04-29 14:23:01.234 INFO  api.main - Server started
    
    In production, you'd typically use JSON-formatted logs that get
    shipped to a log aggregator. For development, human-readable
    is much easier to debug.
    """
    # Numeric level from string ("INFO" -> 20)
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Format with milliseconds
    formatter = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d %(levelname)-5s %(name)-25s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Send all logs to stdout (so they appear in your terminal)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.setLevel(numeric_level)
    
    # Configure root logger
    root = logging.getLogger()
    root.setLevel(numeric_level)
    
    # Remove existing handlers (avoid duplicate logs on uvicorn reload)
    for h in root.handlers[:]:
        root.removeHandler(h)
    
    root.addHandler(handler)
    
    # Quiet down noisy third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    
    logging.info("Logging configured at level: %s", level)