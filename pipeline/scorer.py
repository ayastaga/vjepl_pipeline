import numpy as np

# Hard collapse: any hard flag → CIS = 0 immediately
HARD_FLAGS = {"MISSING_V", "DROP_P"}

# Soft flags — SYNC_ERR gets a 3× multiplier applied in code below
SOFT_FLAGS = {"JITTER_A", "BLUR", "EXPO_S", "STALL", "COMP_A",
              "SYNC_ERR", "JITTER_V", "DUP_FRAME", "ACT_SAT"}

SYNC_ERR_WEIGHT_MULTIPLIER = 3.0   # "unlearnable poison" multiplier


class ExampleScorer:
    def __init__(self, config: dict):
        self.weights  = config["quality"]["weights"]
        q             = config["quality"]
        self.min_eqs  = q["min_quality_score"]
        # Uncertainty band: [min_eqs, min_eqs + uncertainty_band] → flagged for human review
        self.uncertainty_band = q.get("uncertainty_band", 0.15)

    def score(self, bitmask: int, details: dict) -> dict:
        """
        Compute CIS and uncertainty tag from bitmask and flag details.

        Returns dict with:
          cis:          float in [0, 1]
          uncertain:    bool — True if score is in the manual-review priority band
          accepted:     bool — CIS >= min_quality_score
        """
        flag_names = set(details.get("flag_names", []))

        # Hard collapse
        for flag in HARD_FLAGS:
            if flag in flag_names:
                return {"cis": 0.0, "uncertain": False, "accepted": False}

        # Weighted soft penalty — SYNC_ERR is 3× heavier
        soft_sum = 0.0
        for flag, weight in self.weights.items():
            if flag in flag_names:
                effective_weight = weight * (SYNC_ERR_WEIGHT_MULTIPLIER if flag == "SYNC_ERR" else 1.0)
                soft_sum += effective_weight

        cis      = float(np.clip(np.exp(-soft_sum), 0.0, 1.0))
        accepted = cis >= self.min_eqs

        # Uncertainty tagging: borderline examples near the acceptance threshold
        uncertain = (not accepted) and (cis >= self.min_eqs - self.uncertainty_band)

        return {"cis": cis, "uncertain": uncertain, "accepted": accepted}

    def score_batch(self, windows_with_flags: list[tuple[dict, int, dict]]) -> list[dict]:
        return [self.score(bitmask, details) for _, bitmask, details in windows_with_flags]