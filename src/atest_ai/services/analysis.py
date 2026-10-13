"""Orquestração da análise: imagem + escopo -> LLM -> score determinístico.

Independente da origem das entradas: o main.py lê arquivos locais e a futura
API repassará o upload e o escopo recebidos na requisição.
"""

from __future__ import annotations

import uuid
from contextlib import nullcontext
from datetime import datetime, timezone

from atest_ai.chains.visual_reading import build_chain, run_chain
from atest_ai.config import AppConfig
from atest_ai.domain.models import Scope, VisualReading
from atest_ai.domain.scoring import compute_result
from atest_ai.llm.bedrock import create_model
from atest_ai.services.image import prepare_image


def _tracing_context(enabled: bool):
    """Devolve o contexto de tracing conforme a configuração.

    Args:
        enabled: Se o tracing externo está habilitado.

    Returns:
        Context manager nulo, ou um que desliga o tracing do LangSmith.
    """
    if enabled:
        return nullcontext()
    from langsmith import tracing_context

    # Não exportar imagens/prompts para tracing externo por padrão.
    return tracing_context(enabled=False)


def read_image_with_llm(scope: Scope, image: bytes, cfg: AppConfig, client=None) -> tuple[VisualReading, dict]:
    """Obtém do LLM a leitura visual da imagem contra o escopo.

    Args:
        scope: Escopo a avaliar.
        image: Imagem preparada em JPEG.
        cfg: Configuração da aplicação.
        client: Cliente bedrock-runtime opcional (ex.: simulado em testes).

    Returns:
        Tupla (leitura visual validada, metadados do modelo e telemetria).
    """
    model = create_model(cfg.llm, cfg.aws, client=client, network=cfg.network)
    chain = build_chain(model, cfg.prompt.version)
    with _tracing_context(cfg.observability.tracing_enabled):
        reading, telemetry = run_chain(chain, scope, image, cfg.llm.valid_stop_reasons)
    return reading, {
        "model_id": cfg.llm.model_id, "region": cfg.llm.region,
        "reasoning_effort": cfg.llm.reasoning_effort,
        "prompt_version": cfg.prompt.version,
        "integration": "langchain_aws.ChatBedrockConverse",
        **telemetry,
    }


def analyze(image: bytes, scope: Scope, cfg: AppConfig, file_name: str = "", client=None) -> dict:
    """Executa a análise completa de uma imagem contra um escopo.

    Args:
        image: Bytes da imagem original (JPEG, PNG ou WEBP).
        scope: Escopo com os critérios a verificar.
        cfg: Configuração da aplicação.
        file_name: Nome do arquivo de origem, apenas para proveniência.
        client: Cliente bedrock-runtime opcional (ex.: simulado em testes).

    Returns:
        Resultado com score, status, proveniência da imagem, modelo,
        run_id e data de criação.

    Raises:
        ValueError: Se a imagem ou a resposta do modelo forem inválidas.
    """
    prepared, provenance = prepare_image(image, cfg.image, file_name)
    reading, model = read_image_with_llm(scope, prepared, cfg, client=client)
    result = compute_result(scope, reading, cfg.rubric)
    result.update({
        "run_id": str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "image": provenance,
        "model": model,
    })
    return result
