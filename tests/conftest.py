import io
import json

import pytest
from PIL import Image

from atest_ai.config import PROJECT_ROOT, AppConfig, LLMConfig
from atest_ai.domain.models import Observation, Scope, VisualReading

EXAMPLE_SCOPE = PROJECT_ROOT / "examples" / "scope.example.json"


@pytest.fixture
def cfg() -> AppConfig:
    """Configuração tipada padrão, independente de settings.toml e credenciais."""
    return AppConfig(llm=LLMConfig(
        # Este ID é apenas configuração do mock; nenhuma chamada real é feita.
        model_id="global.anthropic.claude-opus-5-5", region="us-east-1",
    ))


@pytest.fixture
def scope() -> Scope:
    """Escopo de exemplo (examples/scope.example.json), como enviado por um cliente."""
    return Scope.model_validate_json(EXAMPLE_SCOPE.read_text(encoding="utf-8"))


@pytest.fixture
def image() -> bytes:
    """Imagem JPEG mínima e válida."""
    buffer = io.BytesIO()
    Image.new("RGB", (20, 20), "white").save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture
def make_reading(scope):
    """Fábrica de leituras visuais com um estado por critério do escopo."""
    def _make(states, legibility="adequate"):
        """Cria uma VisualReading com os estados e a legibilidade informados."""
        return VisualReading(legibility=legibility, observations=[
            Observation(
                criterion_id=c.id, state=s, visible_evidence="Observação de teste.",
                limitation="Limitação de teste." if s in ("partial", "not_verifiable") else "",
            )
            for c, s in zip(scope.criteria, states)
        ])
    return _make


class FakeBedrockClient:
    """Cliente bedrock-runtime simulado que responde com texto JSON (saída estruturada)."""

    def __init__(self, reading: dict, stop_reason: str = "end_turn"):
        """Guarda a leitura e o stopReason que o cliente simulado devolverá."""
        self.reading = reading
        self.stop_reason = stop_reason

    def converse(self, **kwargs):
        """Simula a API Converse do Bedrock e registra o pedido recebido."""
        self.request = kwargs
        return {
            "output": {"message": {"role": "assistant", "content": [
                {"text": json.dumps(self.reading, ensure_ascii=False)}
            ]}},
            "stopReason": self.stop_reason,
            "usage": {"inputTokens": 100, "outputTokens": 40, "totalTokens": 140},
            "metrics": {"latencyMs": 1},
            "ResponseMetadata": {"RequestId": "local-test"},
        }


@pytest.fixture
def fake_client(make_reading):
    """Cliente simulado que devolve todos os critérios atendidos."""
    return FakeBedrockClient(make_reading(["met"] * 4).model_dump())
