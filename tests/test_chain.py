import json

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from atest_ai.chains.visual_reading import output_json_schema, run_chain
from atest_ai.domain.scoring import compute_result
from atest_ai.services.analysis import analyze, read_image_with_llm
from tests.conftest import FakeBedrockClient

STOP_REASONS = {"end_turn"}


def test_langchain_pipeline_uses_native_structured_output(cfg, scope, image, fake_client):
    """Pipeline envia imagem e outputConfig com JSON Schema, sem tool forçado, e gera score."""
    reading, telemetry = read_image_with_llm(scope, image, cfg, client=fake_client)

    assert compute_result(scope, reading, cfg.rubric)["visual_compliance_score"] == 100
    request = fake_client.request
    assert request["messages"][0]["content"][1]["image"]["source"]["bytes"] == image
    assert "toolConfig" not in request
    schema = json.loads(request["outputConfig"]["textFormat"]["structure"]["jsonSchema"]["schema"])
    assert set(schema["required"]) == {"legibility", "observations"}
    assert telemetry["aws_request_id"] == "local-test"
    assert telemetry["prompt_version"] == cfg.prompt.version


def test_analyze_end_to_end(cfg, scope, image, fake_client):
    """analyze recebe bytes + escopo e devolve score, proveniência e metadados do modelo."""
    result = analyze(image, scope, cfg, file_name="foto.jpg", client=fake_client)
    assert (result["visual_compliance_score"], result["status"]) == (100, "visually_compliant")
    assert result["image"]["file_name"] == "foto.jpg"
    assert result["model"]["model_id"] == cfg.llm.model_id


def test_refusal_yields_no_score(cfg, scope, image, make_reading):
    """Recusa do modelo (stopReason refusal) não gera score."""
    client = FakeBedrockClient(make_reading(["met"] * 4).model_dump(), stop_reason="refusal")
    with pytest.raises(ValueError):
        read_image_with_llm(scope, image, cfg, client=client)


def test_reading_out_of_scope_yields_no_score(cfg, scope, image, make_reading):
    """Leitura com critério fora do escopo enviado é rejeitada."""
    data = make_reading(["met"] * 4).model_dump()
    data["observations"][0]["criterion_id"] = "unknown"
    with pytest.raises(ValueError):
        read_image_with_llm(scope, image, cfg, client=FakeBedrockClient(data))


def test_truncated_response_yields_no_score(scope, make_reading):
    """Resposta truncada (max_tokens) não gera score."""
    reading = make_reading(["met"] * 4)
    chain = RunnableLambda(lambda _: {
        "parsed": reading.model_dump(), "parsing_error": None,
        "raw": AIMessage(content="", response_metadata={"stopReason": "max_tokens"}),
    })
    with pytest.raises(ValueError):
        run_chain(chain, scope, b"test", STOP_REASONS)


def test_parser_error_yields_no_score(scope):
    """Erro de parsing da saída estruturada não gera score."""
    chain = RunnableLambda(lambda _: {
        "parsed": None, "parsing_error": ValueError("Invalid response"),
        "raw": AIMessage(content="", response_metadata={"stopReason": "end_turn"}),
    })
    with pytest.raises(ValueError):
        run_chain(chain, scope, b"test", STOP_REASONS)


def test_output_schema_has_no_unsupported_constraints():
    """O schema enviado ao Bedrock não contém restrições de tamanho/intervalo."""
    serialized = json.dumps(output_json_schema())
    for keyword in ("minLength", "maxLength", "minItems", "maxItems", "minimum", "maximum"):
        assert keyword not in serialized
