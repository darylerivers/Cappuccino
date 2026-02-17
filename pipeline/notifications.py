"""
Pipeline Notifications
Sends alerts for pipeline events.
"""

import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


class PipelineNotifier:
    """Sends notifications for pipeline events."""

    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.notification_log = Path("logs/pipeline_notifications.log")
        self.notification_log.parent.mkdir(parents=True, exist_ok=True)

    def notify(self, event_type: str, trial_number: int, message: str,
               details: Optional[Dict] = None):
        """
        Send notification for an event.

        Args:
            event_type: Type of event (gate_passed, gate_failed, deployed, error)
            trial_number: Trial number
            message: Notification message
            details: Optional additional details
        """
        # Log to file
        if self.config.get("log_file", True):
            self._log_to_file(event_type, trial_number, message, details)

        # Desktop notification
        if self.config.get("desktop", True):
            self._send_desktop(event_type, trial_number, message)

        # Email notification (if configured)
        if self.config.get("email", False):
            self._send_email(event_type, trial_number, message, details)

    def _log_to_file(self, event_type: str, trial_number: int, message: str,
                     details: Optional[Dict]):
        """Log notification to file."""
        timestamp = datetime.now().isoformat()

        log_entry = f"[{timestamp}] [{event_type.upper()}] Trial {trial_number}: {message}"

        if details:
            log_entry += f" | Details: {details}"

        try:
            with open(self.notification_log, 'a') as f:
                f.write(log_entry + '\n')
        except Exception as e:
            self.logger.error(f"Failed to log notification: {e}")

    def _send_desktop(self, event_type: str, trial_number: int, message: str):
        """Send desktop notification using notify-send."""
        try:
            # Determine urgency and icon based on event type
            if event_type == "gate_passed":
                urgency = "normal"
                icon = "dialog-information"
                title = f"Pipeline: Gate Passed (Trial {trial_number})"
            elif event_type == "gate_failed":
                urgency = "normal"
                icon = "dialog-warning"
                title = f"Pipeline: Gate Failed (Trial {trial_number})"
            elif event_type == "deployed":
                urgency = "critical"
                icon = "emblem-default"
                title = f"Pipeline: Deployed (Trial {trial_number})"
            elif event_type == "error":
                urgency = "critical"
                icon = "dialog-error"
                title = f"Pipeline: Error (Trial {trial_number})"
            else:
                urgency = "normal"
                icon = "dialog-information"
                title = f"Pipeline: {event_type} (Trial {trial_number})"

            cmd = [
                "notify-send",
                "-u", urgency,
                "-i", icon,
                "-a", "Pipeline Orchestrator",
                title,
                message
            ]

            subprocess.run(cmd, check=False, capture_output=True, timeout=5)

        except Exception as e:
            self.logger.error(f"Failed to send desktop notification: {e}")

    def _send_email(self, event_type: str, trial_number: int, message: str,
                    details: Optional[Dict]):
        """Send email notification (placeholder for future implementation)."""
        # TODO: Implement email notifications
        self.logger.info(f"Email notification not implemented: {message}")

    def gate_passed(self, trial_number: int, gate_name: str, metrics: Dict):
        """Notify that a gate was passed."""
        message = f"{gate_name} gate passed"
        self.notify("gate_passed", trial_number, message, metrics)

    def gate_failed(self, trial_number: int, gate_name: str, reason: str):
        """Notify that a gate failed."""
        message = f"{gate_name} gate failed: {reason}"
        self.notify("gate_failed", trial_number, message)

    def deployed_to_paper(self, trial_number: int):
        """Notify that model was deployed to paper trading."""
        message = "Deployed to paper trading"
        self.notify("deployed", trial_number, message)

    def deployed_to_live(self, trial_number: int):
        """Notify that model was deployed to live trading."""
        message = "🚀 DEPLOYED TO LIVE TRADING 🚀"
        self.notify("deployed", trial_number, message)

    def error_occurred(self, trial_number: int, error: str):
        """Notify that an error occurred."""
        message = f"Error: {error}"
        self.notify("error", trial_number, message)
