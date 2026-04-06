from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Optional, Tuple

import torch


@dataclass(frozen=True)
class NodeFeatureSchema:
    feature_names: Tuple[str, ...]
    defaults: Mapping[str, float] = field(default_factory=dict)
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("Feature names must be unique.")

        normalized_defaults: Dict[str, float] = {}
        for feature in self.feature_names:
            if not isinstance(feature, str):
                raise TypeError("Feature names must be strings.")
            value = self.defaults.get(feature, 0.0)
            if not isinstance(value, (int, float)):
                raise TypeError(f"Default for feature '{feature}' must be numeric.")
            normalized_defaults[feature] = float(value)

        object.__setattr__(self, "feature_names", tuple(self.feature_names))
        object.__setattr__(self, "defaults", normalized_defaults)

    @classmethod
    def from_ordered_features(
        cls,
        feature_names: Iterable[str],
        defaults: Optional[Mapping[str, float]] = None,
        dtype: torch.dtype = torch.float32,
    ) -> "NodeFeatureSchema":
        return cls(tuple(feature_names), defaults or {}, dtype)

    def index(self, feature_name: str) -> int:
        try:
            return self.feature_names.index(feature_name)
        except ValueError as exc:
            raise KeyError(f"Unknown feature '{feature_name}'.") from exc

    def vectorize_tuple(self, scalar_features: Mapping[str, float]) -> Tuple[float, ...]:
        values = []
        for feature_name in self.feature_names:
            value = scalar_features.get(feature_name, self.defaults[feature_name])
            if not isinstance(value, (int, float)):
                raise TypeError(f"Feature '{feature_name}' must be numeric.")
            values.append(float(value))
        return tuple(values)

    def vectorize_tensor(
        self,
        scalar_features: Mapping[str, float],
        *,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        resolved_device = None if device is None else torch.device(device)
        return torch.tensor(self.vectorize_tuple(scalar_features), dtype=self.dtype, device=resolved_device)

    def vectorize(self, scalar_features: Mapping[str, float]) -> Tuple[float, ...]:
        return self.vectorize_tuple(scalar_features)
