"""Cálculo determinístico do score: o modelo só observa, o código pontua."""

from __future__ import annotations

from atest_ai.config import RubricConfig
from atest_ai.domain.models import Scope, VisualReading

LIMITATION_NOTICE = "Visual evidence score; does not prove operation, authenticity, date or specific branch."


def validate_reading(scope: Scope, reading: VisualReading) -> None:
    """Validação semântica da leitura do modelo contra o escopo.

    Verifica se cada critério aparece exatamente uma vez e se uma imagem
    inutilizável não sustenta nenhuma conclusão.

    Args:
        scope: Escopo avaliado.
        reading: Leitura visual devolvida pelo modelo.

    Raises:
        ValueError: Se a leitura for incoerente com o escopo.
    """
    expected = {c.id for c in scope.criteria}
    received = [o.criterion_id for o in reading.observations]
    if len(received) != len(set(received)) or set(received) != expected:
        raise ValueError("The response must contain each criterion exactly once.")
    if reading.legibility == "unusable" and any(
        o.state != "not_verifiable" for o in reading.observations
    ):
        raise ValueError("An unusable image cannot support a visual conclusion.")


def compute_result(scope: Scope, reading: VisualReading, rubric: RubricConfig) -> dict:
    """Calcula o score determinístico e o status da evidência visual.

    Args:
        scope: Escopo avaliado.
        reading: Leitura visual devolvida pelo modelo.
        rubric: Fatores por estado, limiar de aderência e versão da rubrica.

    Returns:
        Dicionário com score, status, pendências essenciais, detalhes por
        critério e justificativa textual.

    Raises:
        ValueError: Se a leitura for incoerente com o escopo.
    """
    validate_reading(scope, reading)

    by_id = {o.criterion_id: o for o in reading.observations}
    items = []
    pending = []
    for c in scope.criteria:
        o = by_id[c.id]
        points = c.weight * rubric.factors[o.state]
        items.append({**c.model_dump(), **o.model_dump(), "points": points})
        if c.essential and o.state != "met":
            pending.append(c.id)
    score = sum(i["points"] for i in items)
    not_verifiable = sum(i["weight"] for i in items if i["state"] == "not_verifiable")
    if reading.legibility == "unusable" or not_verifiable == 100:
        status = "inconclusive"
    elif score >= rubric.compliance_threshold and not pending and reading.legibility == "adequate":
        status = "visually_compliant"
    elif score > 0:
        status = "partial_evidence"
    else:
        status = "insufficient_evidence"

    rationale = " ".join(
        f"{i['criterion_id']}: {i['points']:g}/{i['weight']} points "
        f"({i['state']}). {i['visible_evidence']}"
        + (f" Limitation: {i['limitation']}" if i['limitation'] else "")
        for i in items
    )
    return {
        "scope": scope.model_dump(),
        "visual_compliance_score": score,
        "status": status,
        "legibility": reading.legibility,
        "not_verifiable_weight": not_verifiable,
        "pending_essential_criteria": pending,
        "criteria": items,
        "score_rationale": rationale,
        "rubric_version": rubric.version,
        "limitation_notice": LIMITATION_NOTICE,
    }
