"""
Time-decay calculations for financial events.
"""

from __future__ import annotations

from datetime import datetime, timezone
from math import pow


EVENT_HALF_LIVES_HOURS = {
    "Market": 24.0,
    "Earnings": 48.0,
    "Credit": 48.0,
    "Regulatory": 48.0,
    "Supply Chain": 72.0,
    "Product/Business": 72.0,
    "Corporate Action": 72.0,
    "Monetary Policy": 96.0,
    "Geopolitical": 96.0,
    "Other": 48.0,
}


class TimeDecayEngine:
    """
    Apply event-specific half-life decay.

    D(t) = 2^(-age / half_life)
    """

    def __init__(
        self,
        half_lives: dict[str, float] | None = None,
    ):
        self.half_lives = (
            half_lives
            or EVENT_HALF_LIVES_HOURS.copy()
        )

    def decay(
        self,
        event_type: str,
        event_timestamp: datetime,
        reference_time: datetime | None = None,
    ) -> float:

        if reference_time is None:
            reference_time = datetime.now(
                timezone.utc
            )

        if event_timestamp.tzinfo is None:
            event_timestamp = event_timestamp.replace(
                tzinfo=timezone.utc
            )

        if reference_time.tzinfo is None:
            reference_time = reference_time.replace(
                tzinfo=timezone.utc
            )

        age_seconds = (
            reference_time - event_timestamp
        ).total_seconds()

        # Future timestamps are treated as zero age.
        age_hours = max(
            0.0,
            age_seconds / 3600.0,
        )

        half_life = self.half_lives.get(
            event_type,
            self.half_lives["Other"],
        )

        return pow(
            2.0,
            -age_hours / half_life,
        )