from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DoseLimits:
    target_mg_dl: float = 110.0
    correction_divisor_mg_dl_per_u: float = 30.0
    max_bolus_u: float = 4.0
    max_iob_reduction: float = 0.65
    min_glucose_block_mg_dl: float = 90.0
    low_glucose_reduction_mg_dl: float = 110.0


@dataclass
class DoseResult:
    cho_u: float
    correction_u: float
    raw_u: float
    pk_active_u: float
    pk_att: float
    final_u: float
    blocked: bool
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "cho_u": self.cho_u,
            "correction_u": self.correction_u,
            "raw_u": self.raw_u,
            "pk_active_u": self.pk_active_u,
            "pk_att": self.pk_att,
            "final_u": self.final_u,
            "blocked": self.blocked,
            "reasons": self.reasons,
        }


def _clamp(v: float, vmin: float, vmax: float) -> float:
    return max(vmin, min(vmax, v))


def compute_dose(
    *,
    glucose_mg_dl: float,
    cho_g: float,
    sensitivity_u_per_g: float,
    pk_active_u: float,
    limits: DoseLimits | None = None,
) -> DoseResult:
    limits = limits or DoseLimits()
    reasons: list[str] = []

    cho = max(0.0, float(cho_g))
    glucose = max(0.0, float(glucose_mg_dl))
    sens = _clamp(float(sensitivity_u_per_g), 0.01, 0.30)
    iob = max(0.0, float(pk_active_u))

    cho_u = max(0.0, sens * cho)
    correction_u = (glucose - limits.target_mg_dl) / max(limits.correction_divisor_mg_dl_per_u, 1e-6)
    raw_u = max(0.0, cho_u + correction_u)

    pk_att = _clamp(iob / 6.0, 0.0, limits.max_iob_reduction)
    post_pk = raw_u * (1.0 - pk_att)

    blocked = False
    if glucose < limits.min_glucose_block_mg_dl:
        blocked = True
        reasons.append("glucose_below_block_threshold")
    elif glucose < limits.low_glucose_reduction_mg_dl:
        post_pk *= 0.75
        reasons.append("low_glucose_soft_reduction")

    if raw_u > limits.max_bolus_u:
        reasons.append("max_bolus_clamped")

    final_u = 0.0 if blocked else _clamp(post_pk, 0.0, limits.max_bolus_u)

    return DoseResult(
        cho_u=cho_u,
        correction_u=correction_u,
        raw_u=raw_u,
        pk_active_u=iob,
        pk_att=pk_att,
        final_u=final_u,
        blocked=blocked,
        reasons=reasons,
    )


def simulate_scenarios(
    *,
    glucose_mg_dl: float,
    cho_effective_g: float,
    sensitivity_u_per_g: float,
    pk_active_u: float,
    limits: DoseLimits | None = None,
) -> dict[str, dict[str, Any]]:
    limits = limits or DoseLimits()
    base = compute_dose(
        glucose_mg_dl=glucose_mg_dl,
        cho_g=cho_effective_g,
        sensitivity_u_per_g=sensitivity_u_per_g,
        pk_active_u=pk_active_u,
        limits=limits,
    )
    conservative = compute_dose(
        glucose_mg_dl=glucose_mg_dl,
        cho_g=cho_effective_g * 0.85,
        sensitivity_u_per_g=sensitivity_u_per_g,
        pk_active_u=pk_active_u,
        limits=limits,
    )
    aggressive = compute_dose(
        glucose_mg_dl=glucose_mg_dl,
        cho_g=cho_effective_g * 1.15,
        sensitivity_u_per_g=sensitivity_u_per_g,
        pk_active_u=pk_active_u,
        limits=limits,
    )
    return {
        "conservative": conservative.to_dict(),
        "base": base.to_dict(),
        "aggressive": aggressive.to_dict(),
    }
