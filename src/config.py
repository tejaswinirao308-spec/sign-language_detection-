from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, time


SIGN_LABELS = [
    "Hello", "Thank You", "Please", "Yes", "No", "Sorry", "Help", "Stop",
    "Good", "Bad", "I Love You", "Bye", "Okay", "Peace", "Welcome",
]

ALPHABET_LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


@dataclass(frozen=True)
class AppSettings:
    confidence_threshold: float = float(os.getenv("SIGN_CONFIDENCE_THRESHOLD", "0.70"))
    active_start_hour: int = int(os.getenv("SIGN_ACTIVE_START_HOUR", "18"))
    active_end_hour: int = int(os.getenv("SIGN_ACTIVE_END_HOUR", "22"))
    # Eight adjacent confident frames are deliberately required for the live
    # overlay.  This is still responsive on a 20–30 FPS camera while filtering
    # the one-frame errors that are common under webcam lighting changes.
    smoothing_frames: int = int(os.getenv("SIGN_SMOOTHING_FRAMES", "8"))
    stable_seconds: float = float(os.getenv("SIGN_STABLE_SECONDS", "2.0"))
    # Variance of the Laplacian on the uncompressed, padded hand ROI.  The
    # held-out A–Z hand crops are all above 130, so 80 rejects motion blur
    # conservatively without rejecting normally focused signs.
    webcam_blur_threshold: float = float(os.getenv("SIGN_WEBCAM_BLUR_THRESHOLD", "80"))
    # Temporary manual validation switch. It is off unless explicitly set in
    # the process environment, so production keeps the 6 PM–10 PM restriction.
    test_mode: bool = os.getenv("SIGN_TEST_MODE", "0") == "1"

    @property
    def active_period_label(self) -> str:
        return f"{self.active_start_hour:02d}:00 – {self.active_end_hour:02d}:00 (local time)"


def is_active_now(settings: AppSettings, now: datetime | None = None) -> bool:
    """Return whether local computer time is in the configured operating window."""
    if settings.test_mode:
        return True
    current = (now or datetime.now().astimezone()).time()
    start, end = time(settings.active_start_hour), time(settings.active_end_hour)
    if start <= end:
        return start <= current < end
    return current >= start or current < end
