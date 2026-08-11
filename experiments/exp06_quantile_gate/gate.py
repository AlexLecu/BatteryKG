"""Quantile-referenced abstention gate — see src.models.quantile_gate.

The implementation lived here while the rule was a candidate. It moved to
src/models/quantile_gate.py when it became the deployed gate that
src.models.train_final writes into the serving artifacts: src must not import
from experiments/. This module keeps the original import path so the
experiment's entry points and its reported numbers stay reproducible.
"""
from __future__ import annotations

from src.models.quantile_gate import (  # noqa: F401
    BEHAVIOR_VIEW,
    Q_STAR,
    fold_loo_distributions,
    fold_thresholds,
    gate_decisions,
    keep_mask_for_q,
    loo_coverage,
    threshold_at_q,
)

__all__ = ["BEHAVIOR_VIEW", "Q_STAR", "fold_loo_distributions", "fold_thresholds",
           "gate_decisions", "keep_mask_for_q", "loo_coverage", "threshold_at_q"]
