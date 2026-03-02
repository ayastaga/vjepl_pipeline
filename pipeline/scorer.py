import numpy as np
from pipeline.quality import FLAG_BITS


HARD_FLAGS = {"MISSING_V", "DROP_P"}
SOFT_FLAGS = {"JITTER_A", "BLUR", "EXPO_S", "STALL", "COMP_A", "SYNC_ERR", "JITTER_V"}


class ExampleScorer:
    def __init__(self, config: dict):
        self.weights = config["quality"]["weights"]

    def score(self, bitmask: int, details: dict) -> float:
        """
        Compute EQS from bitmask and optional detail values.
        Returns float in [0, 1].
        """
        flag_names = set(details.get("flag_names", []))

        # Hard penalty: any hard flag -> 0.0
        hard_product = 1.0
        for flag in HARD_FLAGS:
            if flag in flag_names:
                hard_product = 0.0
                break

        if hard_product == 0.0:
            return 0.0

        # Soft penalty: weighted sum of active soft flags
        soft_sum = 0.0
        for flag, weight in self.weights.items():
            if flag in flag_names:
                soft_sum += weight

        eqs = hard_product * np.exp(-soft_sum)
        return float(np.clip(eqs, 0.0, 1.0))

    def score_batch(self, windows_with_flags: list[tuple[dict, int, dict]]) -> list[float]:
        """Score a batch of (window, bitmask, details) tuples."""
        return [self.score(bitmask, details) for _, bitmask, details in windows_with_flags]