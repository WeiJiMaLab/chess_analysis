"""Changing ``TREE_ENCODER_FEATURE_NAMES`` is a format-breaking change."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Optional, Tuple

import torch


@dataclass(frozen=True)
class NodeFeatureSchema:
    """Ordered, immutable mapping from feature name to tensor column.

    The ordering of ``feature_names`` defines which encoder column a feature
    occupies — every consumer reads from this schema so the columns can't
    silently desynchronize across modules. Frozen to make that guarantee.
    """

    feature_names: Tuple[str, ...]  # canonical column order; column i ↔ feature_names[i]
    defaults: Mapping[str, float] = field(default_factory=dict)  # fallback per feature; absent entries default to 0.0
    dtype: torch.dtype = torch.float32  # tensor dtype used by vectorize_tensor

    def __post_init__(self) -> None:
        # Validate uniqueness/types and fill any missing default to 0.0.
        # object.__setattr__ is required because the dataclass is frozen.
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
        """Construct from any iterable, treating ``defaults=None`` as empty.

        Args:
            feature_names: ordered feature names (any iterable; materialized to a tuple).
            defaults: optional fallback values; missing entries default to 0.0.
            dtype: tensor dtype for ``vectorize_tensor`` output.
        """
        return cls(tuple(feature_names), defaults or {}, dtype)

    def index(self, feature_name: str) -> int:
        """Return the column index of ``feature_name``. Raises ``KeyError`` if unknown."""
        try:
            return self.feature_names.index(feature_name)
        except ValueError as exc:
            raise KeyError(f"Unknown feature '{feature_name}'.") from exc

    def vectorize_tuple(self, scalar_features: Mapping[str, float]) -> Tuple[float, ...]:
        """Project a sparse feature dict onto the canonical ordered tuple.

        Missing features fall back to ``self.defaults``. Output length is
        always ``len(self.feature_names)``. Non-numeric values fail loudly
        here rather than as cryptic tensor-construction errors later.

        Args:
            scalar_features: name → value mapping; may omit any feature.
        """
        values = []
        for feature_name in self.feature_names:
            value = scalar_features.get(feature_name, self.defaults[feature_name])
            if not isinstance(value, (int, float)):
                raise TypeError(f"Feature '{feature_name}' must be numeric.")
            values.append(float(value))
        return tuple(values)


# Canonical encoder feature vector: scalar value estimate, WDL triple, and
# WDL variance. Changing this list breaks every saved checkpoint and packed
# pretrain dataset on disk.
TREE_ENCODER_FEATURE_NAMES: Tuple[str, ...] = ("value", "wdl_win", "wdl_draw", "wdl_loss", "wdl_var")


def tree_encoder_feature_schema() -> NodeFeatureSchema:
    """Column order for packed ``node_features`` (the 5 baseline encoder features)."""
    return NodeFeatureSchema.from_ordered_features(TREE_ENCODER_FEATURE_NAMES)
