"""Per-node feature schema for the tree encoder.

Single source of truth for which scalar features a tree node carries and in
what order. Disk format, in-memory ``SearchTree`` features, and encoder input
tensor columns all funnel through ``NodeFeatureSchema`` so they stay aligned.
Changing ``TREE_ENCODER_FEATURE_NAMES`` is a format-breaking change.
"""

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

    def vectorize_tensor(
        self,
        scalar_features: Mapping[str, float],
        *,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Vectorize and wrap as a 1-D tensor with ``self.dtype``.

        Args:
            scalar_features: name → value mapping; may omit any feature.
            device: target device for the result tensor; defaults to current default.
        """
        resolved_device = None if device is None else torch.device(device)
        return torch.tensor(self.vectorize_tuple(scalar_features), dtype=self.dtype, device=resolved_device)

    def vectorize(self, scalar_features: Mapping[str, float]) -> Tuple[float, ...]:
        """Backwards-compat alias for ``vectorize_tuple``."""
        return self.vectorize_tuple(scalar_features)


# Canonical encoder feature vector: scalar value estimate, WDL triple, and
# WDL variance. Changing this list breaks every saved checkpoint and packed
# pretrain dataset on disk.
TREE_ENCODER_FEATURE_NAMES: Tuple[str, ...] = ("value", "wdl_win", "wdl_draw", "wdl_loss", "wdl_var")

# Optional extra columns for node targets GNN pretraining (``v5+``): per-node value gap and policy drift.
# Appended after ``TREE_ENCODER_FEATURE_NAMES`` when packing with ``node_targets: true``.
# We selectively extract only the "policy_drift" head for pretraining.
TEACHER_NODETARGETS_FEATURE_NAMES: Tuple[str, ...] = ("policy_drift",)

# Column order for flat ``nodetargets_targets`` supervision tensors.
NODETARGETS_TARGET_FEATURE_NAMES: Tuple[str, ...] = TEACHER_NODETARGETS_FEATURE_NAMES

# Metadata value stored in node-pretrain checkpoints for target scaling.
NODETARGETS_TARGET_SCALING_VALUE_GAP_CP = "raw_value_gap_cp"
NODETARGETS_TARGET_SCALING_LOG1P_VISITS = "log1p_visits_raw_value_gap_cp"
NODETARGETS_TARGET_SCALING_LOG1P_VISITS_NODES_BELOW = "log1p_visits_nodes_below"


def nodetargets_target_feature_names() -> Tuple[str, ...]:
    """Return ordered node targets supervision columns."""
    return NODETARGETS_TARGET_FEATURE_NAMES


def tree_encoder_feature_schema(*, node_targets: bool = False) -> NodeFeatureSchema:
    """Column order for packed ``node_features`` (5 baseline, +1 teacher node target if ``node_targets``)."""
    names = TREE_ENCODER_FEATURE_NAMES
    if node_targets:
        names = names + TEACHER_NODETARGETS_FEATURE_NAMES
    return NodeFeatureSchema.from_ordered_features(names)


def require_tree_encoder_scalar_features(scalar_features: Mapping[str, float], *, context: str = "node") -> None:
    """Fail loudly if the encoder's required features are missing.

    Catches the common case of training on data generated by an older
    provider that didn't emit WDL features. The error names the missing
    features and the fix.

    Args:
        scalar_features: feature dict to check.
        context: short label (e.g. "root", "child") interpolated into the
            error message so the caller knows which dict is malformed.
    """
    missing = [feature for feature in TREE_ENCODER_FEATURE_NAMES if feature not in scalar_features]
    if missing:
        raise ValueError(
            f"{context} is missing required tree encoder features {missing}. "
            "Regenerate the teacher trees with WDL valuehead features."
        )
