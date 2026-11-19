"""v2.17 photo-quality and classification-consistency guardrail: should the user retake this photo?

It runs after the analysis and reads only what the analysis already produced, so it never changes a wire, a class,
a CRS or the verdict; it decides whether the result is shown straight away, shown with a note, or held behind a
"retake / proceed anyway" prompt.

Severity per issue: "info" < "attention" < "retake". The status is:
    acceptable            no issue above info
    needs_attention       >= 1 attention issue
    retake_recommended    >= 1 retake issue                 -> requires_retake
    highly_uncertain      >= 2 retake issues                -> requires_retake
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

ORDER = {"info": 0, "attention": 1, "retake": 2}


@dataclass(frozen=True)
class GuardrailConfig:
    min_wires: int = 3
    conflict_min_share: float = 0.35
    conflict_retake_n: int = 2
    uncertain_attention: float = 0.35
    uncertain_retake: float = 0.65
    fragment_attention: float = 0.45
    fragment_retake: float = 0.75
    structure_attention: float = 0.40
    structure_retake: float = 0.70


def _issue(
    code: str,
    severity: str,
    message: str,
    value: float | None = None,
    threshold: float | None = None,
    strands: list[int] | None = None,
) -> dict[str, Any]:
    return {
        "code": code,
        "severity": severity,
        "message": message,
        "value": None if value is None else round(float(value), 4),
        "threshold": threshold,
        "affected_strand_ids": strands or [],
    }


def _level(value: float, attention: float, retake: float) -> str | None:
    if value >= retake:
        return "retake"
    if value >= attention:
        return "attention"
    return None


def assess_guardrail(
    wires: list[dict[str, Any]],
    *,
    flags: list[str] | None = None,
    poles_detected: bool | None = None,
    quality_grade: str | None = None,
    quality_reason: str | None = None,
    suppressed: list[dict[str, Any]] | None = None,
    cfg: GuardrailConfig | None = None,
) -> dict[str, Any]:
    """Evaluate photo quality and classification consistency guardrail on analysis output."""
    gcfg = cfg or GuardrailConfig()
    all_flags = list(flags or [])
    n = len(wires)
    issues: list[dict[str, Any]] = []

    if quality_grade == "Retake recommended":
        issues.append(_issue("image_quality", "retake", quality_reason or "The photo is too hard to read for wires."))
    if any(f.startswith("outside_envelope:focal_length") or f == "lens_outside_envelope" for f in all_flags):
        issues.append(
            _issue(
                "lens",
                "retake",
                "The photo was taken with an ultra-wide lens, which bends straight wires; retake with the main (1x) camera.",
            )
        )

    conflicts: list[dict[str, Any]] = []
    uncertain: list[int] = []
    short_ids: list[int] = []

    for idx, w in enumerate(wires, start=1):
        w_idx = int(w.get("id", w.get("index", idx)))
        w_flags = list(w.get("flags") or [])
        if "too_short_to_fit" in w_flags or w.get("status") == "too_short":
            short_ids.append(w_idx)
        if "class_uncertain" in w_flags:
            uncertain.append(w_idx)
            share = w.get("class_evidence") or {}
            top = sorted(share.items(), key=lambda kv: -kv[1])[:2]
            if len(top) == 2 and top[1][1] >= gcfg.conflict_min_share:
                conflicts.append({"strand_id": w_idx, "evidence": {k: round(v, 3) for k, v in top}})

    if conflicts:
        sev = "retake" if len(conflicts) >= gcfg.conflict_retake_n else "attention"
        issues.append(
            _issue(
                "class_conflict",
                sev,
                f"{len(conflicts)} wire{'s' if len(conflicts) != 1 else ''} show strong evidence for two different wire types.",
                len(conflicts),
                gcfg.conflict_retake_n,
                [c["strand_id"] for c in conflicts],
            )
        )

    if n >= gcfg.min_wires:
        u = len(uncertain) / n
        if lv := _level(u, gcfg.uncertain_attention, gcfg.uncertain_retake):
            issues.append(
                _issue(
                    "uncertain_types",
                    lv,
                    f"The type of {len(uncertain)} of {n} wires could not be told apart.",
                    u,
                    gcfg.uncertain_retake,
                    uncertain,
                )
            )
        f = len(short_ids) / n
        if lv := _level(f, gcfg.fragment_attention, gcfg.fragment_retake):
            issues.append(
                _issue(
                    "fragmentation",
                    lv,
                    f"{len(short_ids)} of {n} wires were found only in short pieces and could not be measured.",
                    f,
                    gcfg.fragment_retake,
                    short_ids,
                )
            )

    structure = [s for s in (suppressed or []) if s.get("reason_code") in ("structure_edge", "ground_structure")]
    if structure and n + len(structure) >= gcfg.min_wires:
        s_ratio = len(structure) / (n + len(structure))
        if lv := _level(s_ratio, gcfg.structure_attention, gcfg.structure_retake):
            issues.append(
                _issue(
                    "structure",
                    lv,
                    f"{len(structure)} line-like structure edges were found and ignored.",
                    s_ratio,
                    gcfg.structure_retake,
                )
            )

    if poles_detected is False:
        issues.append(_issue("poles", "info", "No utility pole was found; findings are per wire, not per pole."))

    worst = max((ORDER[i["severity"]] for i in issues), default=0)
    n_retake = sum(i["severity"] == "retake" for i in issues)
    status = (
        "highly_uncertain"
        if n_retake >= 2
        else "retake_recommended"
        if n_retake
        else "needs_attention"
        if worst == ORDER["attention"]
        else "acceptable"
    )
    requires = status in ("retake_recommended", "highly_uncertain")
    affected = sorted({s for i in issues for s in i["affected_strand_ids"]})
    return {
        "version": "v2.17",
        "requires_retake": requires,
        "quality_status": status,
        "proceed_with_warning": requires,
        "quality_issues": issues,
        "quality_reasons": [i["message"] for i in issues if i["severity"] != "info"],
        "classification_conflicts": conflicts,
        "uncertain_strand_count": len(uncertain),
        "unassessed_strand_count": len(short_ids),
        "affected_strand_ids": affected,
    }
