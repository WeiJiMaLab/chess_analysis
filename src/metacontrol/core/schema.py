from typing import Tuple

class ChessFeatureSchema:
    def __init__(self):
        self.feature_names: Tuple[str, ...] = ("value", "wdl_win", "wdl_draw", "wdl_loss")
        self.num_features: int = len(self.feature_names)

    def vectorize(self, features: dict) -> list[float]:
        return [features.get(name, 0.0) for name in self.feature_names]
