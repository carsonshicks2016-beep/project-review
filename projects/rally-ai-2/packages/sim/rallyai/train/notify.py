"""
Minimal push notification module using ntfy.sh.

This module provides optional push notifications for long-running training tasks.
It depends only on the standard library to ensure notifications never crash a run
even if external dependencies are missing or the network is down.
"""

import sys
import urllib.request
import urllib.error


def send_notification(
    topic: str,
    title: str,
    message: str,
    *,
    priority: str = "default",
    tags: list[str] | None = None,
) -> bool:
    """Send a push notification via ntfy.sh. Returns True on success, False on failure.
    Never raises — a notification failure must not crash a training run.
    """
    try:
        url = f"https://ntfy.sh/{topic}"
        data = message.encode("utf-8")
        headers = {
            "Title": title,
            "Priority": priority,
        }
        if tags:
            headers["Tags"] = ",".join(tags)

        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=5.0) as response:
            return response.status == 200
    except Exception as e:
        print(f"ntfy: Failed to send notification: {e}", file=sys.stderr)
        return False


class Notifier:
    """Thin wrapper that caches the topic and adds a run_id prefix."""

    def __init__(self, topic: str | None, run_id: str = ""):
        self.topic = topic  # None means disabled
        self.run_id = run_id

    def send(
        self,
        title: str,
        message: str,
        *,
        priority: str = "default",
        tags: list[str] | None = None,
    ) -> bool:
        if not self.topic:
            return False
        full_title = f"[{self.run_id}] {title}" if self.run_id else title
        return send_notification(
            self.topic, full_title, message, priority=priority, tags=tags
        )

    # Convenience methods:
    def training_started(self, stage: str, workers: int, timesteps: int) -> bool:
        return self.send(
            title="Training Started",
            message=f"Stage: {stage}\nWorkers: {workers}\nTimesteps: {timesteps}",
            tags=["rocket"],
        )

    def collapse_detected(self, completion: float, best: float) -> bool:
        return self.send(
            title="Collapse Detected",
            message=f"Completion fell to {completion:.1f}% (best: {best:.1f}%)",
            priority="high",
            tags=["warning", "chart_with_downwards_trend"],
        )

    def rollback_executed(self, checkpoint: str) -> bool:
        return self.send(
            title="Rollback Executed",
            message=f"Rolled back to checkpoint: {checkpoint}",
            priority="high",
            tags=["rewind"],
        )

    def tier_promoted(self, old_tier: int, new_tier: int, rate: float) -> bool:
        return self.send(
            title="Tier Promoted",
            message=f"Tier {old_tier} -> {new_tier}\nCompletion Rate: {rate:.1f}%",
            tags=["tada", "arrow_up"],
        )

    def training_complete(self, timesteps: int, elapsed_s: float) -> bool:
        return self.send(
            title="Training Complete",
            message=f"Timesteps: {timesteps}\nElapsed: {elapsed_s:.1f}s",
            tags=["checkered_flag"],
        )

    def training_crashed(self, error: str) -> bool:
        return self.send(
            title="Training Crashed",
            message=f"Error:\n{error}",
            priority="urgent",
            tags=["rotating_light", "skull"],
        )
