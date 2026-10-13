import pytest

from atest_ai.domain.models import Scope, VisualReading
from atest_ai.domain.scoring import compute_result


def test_all_met(cfg, scope, make_reading):
    """Todos os critérios atendidos resultam em 100 e visually_compliant."""
    r = compute_result(scope, make_reading(["met"] * 4), cfg.rubric)
    assert (r["visual_compliance_score"], r["status"]) == (100, "visually_compliant")


def test_missing_essential_brand(cfg, scope, make_reading):
    """Marca essencial não verificável impede aderência mesmo com 85."""
    r = compute_result(scope, make_reading(["met", "met", "met", "not_verifiable"]), cfg.rubric)
    assert (r["visual_compliance_score"], r["status"]) == (85, "partial_evidence")


def test_essential_object_contradicted(cfg, scope, make_reading):
    """Objeto essencial contraditado impede aderência."""
    r = compute_result(scope, make_reading(["not_met", "met", "met", "met"]), cfg.rubric)
    assert r["status"] != "visually_compliant"


def test_unusable_image(cfg, scope, make_reading):
    """Imagem inutilizável resulta em score 0 e inconclusive."""
    r = compute_result(scope, make_reading(["not_verifiable"] * 4, "unusable"), cfg.rubric)
    assert (r["visual_compliance_score"], r["status"]) == (0, "inconclusive")


def test_partial(cfg, scope, make_reading):
    """Estado partial vale metade do peso do critério."""
    r = compute_result(scope, make_reading(["partial", "not_verifiable", "not_verifiable", "not_verifiable"]), cfg.rubric)
    assert r["visual_compliance_score"] == 20


def test_rejects_missing_id(cfg, scope, make_reading):
    """Leitura sem um dos critérios é rejeitada."""
    reading = make_reading(["met"] * 4)
    reading.observations.pop()
    with pytest.raises(ValueError):
        compute_result(scope, reading, cfg.rubric)


def test_rejects_duplicate_id(cfg, scope, make_reading):
    """Leitura com critério duplicado é rejeitada."""
    reading = make_reading(["met"] * 4)
    reading.observations[3].criterion_id = "object"
    with pytest.raises(ValueError):
        compute_result(scope, reading, cfg.rubric)


def test_rejects_model_score(make_reading):
    """Score enviado pelo modelo é rejeitado pelo contrato."""
    data = make_reading(["met"] * 4).model_dump()
    data["score"] = 100
    with pytest.raises(ValueError):
        VisualReading.model_validate(data)


def test_rejects_wrong_weights(scope):
    """Escopo com pesos que não somam 100 é rejeitado."""
    data = scope.model_dump()
    data["criteria"][0]["weight"] = 39
    with pytest.raises(ValueError):
        Scope.model_validate(data)


def test_configurable_threshold(cfg, scope, make_reading):
    """O limiar de aderência vem da configuração da rubrica."""
    rubric = cfg.rubric.model_copy(update={"compliance_threshold": 101})
    r = compute_result(scope, make_reading(["met"] * 4), rubric)
    assert r["status"] == "partial_evidence"
