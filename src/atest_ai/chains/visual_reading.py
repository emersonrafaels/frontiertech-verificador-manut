"""Chain LCEL: prompt multimodal -> modelo com saída estruturada (VisualReading).

Usa a saída estruturada nativa do Bedrock (outputConfig.textFormat com JSON
Schema) em vez de tool calling: os modelos Claude atuais (ex.: Opus 5.5)
não aceitam tool_choice forçado, então o modelo poderia responder sem chamar
o tool. A validação completa continua sendo feita pelo Pydantic.
"""

from __future__ import annotations

import base64
from collections.abc import Iterable
from typing import Any

from atest_ai.domain.models import Scope, VisualReading
from atest_ai.domain.scoring import validate_reading
from atest_ai.prompts.loader import build_prompt

# Restrições de JSON Schema não aceitas pela saída estruturada nativa;
# são removidas do schema enviado e reaplicadas pelo Pydantic na validação.
_UNSUPPORTED_KEYWORDS = frozenset({
    "minLength", "maxLength", "minItems", "maxItems",
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
})


def _strip_unsupported(node: Any) -> Any:
    """Remove recursivamente as restrições não suportadas do JSON Schema.

    Args:
        node: Nó do JSON Schema (dicionário, lista ou valor escalar).

    Returns:
        Cópia do nó sem as chaves de _UNSUPPORTED_KEYWORDS.
    """
    if isinstance(node, dict):
        return {k: _strip_unsupported(v) for k, v in node.items() if k not in _UNSUPPORTED_KEYWORDS}
    if isinstance(node, list):
        return [_strip_unsupported(v) for v in node]
    return node


def output_json_schema() -> dict:
    """Gera o JSON Schema de VisualReading aceito pela saída estruturada do Bedrock.

    Returns:
        Schema com título e descrição, sem restrições de tamanho/intervalo.
    """
    return _strip_unsupported(VisualReading.model_json_schema())


def build_chain(model, prompt_version: str):
    """Compõe prompt e modelo com saída estruturada nativa em JSON.

    Args:
        model: ChatBedrockConverse de um modelo com suporte a imagem e saída estruturada.
        prompt_version: Versão do prompt a usar.

    Returns:
        Runnable que devolve {"raw", "parsed", "parsing_error"}.
    """
    return build_prompt(prompt_version) | model.with_structured_output(
        output_json_schema(), method="json_schema", include_raw=True,
    )


def run_chain(chain, scope: Scope, image: bytes, valid_stop_reasons: Iterable[str]):
    """Executa a chain e aplica os guardrails da resposta.

    Args:
        chain: Runnable criado por build_chain (ou equivalente em testes).
        scope: Escopo a avaliar.
        image: Imagem preparada em JPEG.
        valid_stop_reasons: stopReasons aceitos como resposta completa.

    Returns:
        Tupla (leitura visual validada, telemetria da chamada).

    Raises:
        ValueError: Se a resposta for inválida, truncada, recusada ou
            incoerente com o escopo; nesse caso não há score.
    """
    response = chain.invoke({
        "scope_json": scope.model_dump_json(),
        "image_base64": base64.b64encode(image).decode("ascii"),
    })
    raw = response.get("raw")
    if raw is None or raw.response_metadata.get("stopReason") not in set(valid_stop_reasons):
        raise ValueError("Model did not complete a regular response; analysis without score.")
    if response.get("parsing_error") is not None or response.get("parsed") is None:
        raise ValueError("Invalid structured response; analysis without score.")
    reading = VisualReading.model_validate(response["parsed"])
    validate_reading(scope, reading)
    return reading, {
        "usage": raw.usage_metadata or {},
        "aws_request_id": raw.response_metadata.get("ResponseMetadata", {}).get("RequestId"),
        "metrics": raw.response_metadata.get("metrics", {}),
        "stop_reason": raw.response_metadata.get("stopReason"),
    }
