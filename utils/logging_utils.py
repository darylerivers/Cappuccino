"""Structured Logging Utilities

Simple structured logging to replace print() statements throughout the codebase.
Makes logs parseable, filterable, and easier to analyze.

Usage:
    from utils.logging_utils import get_logger
    logger = get_logger("paper_trader")
    logger.info("trade_executed", action=0.5, portfolio_value=1050.0)
"""

import json
import sys
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional


class LogLevel(Enum):
    """Log severity levels."""
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


class StructuredLogger:
    """Simple structured logger for trading system."""

    def __init__(
        self,
        name: str,
        min_level: LogLevel = LogLevel.INFO,
        output_file: Optional[Path] = None,
        console: bool = True
    ):
        """Initialize logger.

        Args:
            name: Logger name (e.g., "paper_trader", "environment")
            min_level: Minimum level to log
            output_file: Optional file to write logs to
            console: Whether to print to console
        """
        self.name = name
        self.min_level = min_level
        self.output_file = output_file
        self.console = console

        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)

    def _log(self, level: LogLevel, event: str, **kwargs):
        """Internal logging method."""
        if level.value < self.min_level.value:
            return

        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level.name,
            "logger": self.name,
            "event": event,
        }
        log_entry.update(kwargs)

        # Format for console (human-readable)
        if self.console:
            console_msg = self._format_console(log_entry)
            print(console_msg, file=sys.stdout if level.value < LogLevel.ERROR.value else sys.stderr)

        # Write to file (JSON for parsing)
        if self.output_file:
            with self.output_file.open("a") as f:
                f.write(json.dumps(log_entry) + "\n")

    def _format_console(self, log_entry: Dict[str, Any]) -> str:
        """Format log entry for console output."""
        level = log_entry["level"]
        event = log_entry["event"]
        timestamp = log_entry["timestamp"].split("T")[1][:8]  # Just HH:MM:SS

        # Level icons
        icons = {
            "DEBUG": "🔍",
            "INFO": "ℹ️",
            "WARNING": "⚠️",
            "ERROR": "❌",
            "CRITICAL": "🚨"
        }
        icon = icons.get(level, "•")

        # Build message
        msg = f"[{timestamp}] {icon} {self.name}.{event}"

        # Add key-value pairs
        extra_keys = [k for k in log_entry.keys() if k not in ["timestamp", "level", "logger", "event"]]
        if extra_keys:
            extras = ", ".join(f"{k}={log_entry[k]}" for k in extra_keys)
            msg += f" | {extras}"

        return msg

    def debug(self, event: str, **kwargs):
        """Log debug message."""
        self._log(LogLevel.DEBUG, event, **kwargs)

    def info(self, event: str, **kwargs):
        """Log info message."""
        self._log(LogLevel.INFO, event, **kwargs)

    def warning(self, event: str, **kwargs):
        """Log warning message."""
        self._log(LogLevel.WARNING, event, **kwargs)

    def error(self, event: str, **kwargs):
        """Log error message."""
        self._log(LogLevel.ERROR, event, **kwargs)

    def critical(self, event: str, **kwargs):
        """Log critical message."""
        self._log(LogLevel.CRITICAL, event, **kwargs)


# Global logger registry
_loggers: Dict[str, StructuredLogger] = {}


def get_logger(
    name: str,
    min_level: LogLevel = LogLevel.INFO,
    output_file: Optional[Path] = None,
    console: bool = True
) -> StructuredLogger:
    """Get or create a logger.

    Args:
        name: Logger name
        min_level: Minimum level to log
        output_file: Optional file path
        console: Whether to print to console

    Returns:
        StructuredLogger instance
    """
    if name not in _loggers:
        _loggers[name] = StructuredLogger(
            name=name,
            min_level=min_level,
            output_file=output_file,
            console=console
        )
    return _loggers[name]


def set_global_level(level: LogLevel):
    """Set minimum level for all loggers."""
    for logger in _loggers.values():
        logger.min_level = level


# Convenience exports
__all__ = [
    "get_logger",
    "LogLevel",
    "set_global_level",
]


# Example usage
if __name__ == "__main__":
    # Demo
    logger = get_logger("demo", min_level=LogLevel.DEBUG)

    logger.debug("system_start", version="1.0.0")
    logger.info("trade_executed", action=0.5, price=50000, profit_pct=2.5)
    logger.warning("high_volatility", volatility=0.15, threshold=0.10)
    logger.error("api_failure", endpoint="/bars", status_code=429)
    logger.critical("stop_loss_triggered", asset="BTC", loss_pct=10.5)
