"""Processamento em lote (MVP D+1): um parecer por foto, um JSON por chamado.

Cada foto é analisada de forma independente: uma falha (rede, recusa,
resposta inválida) é registrada no item e não interrompe o lote.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from atest_ai.config import AppConfig
from atest_ai.domain.models import AnalysisRequest, Manifest
from atest_ai.services.analysis import analyze


def load_manifest(path: Path) -> Manifest:
    """Lê e valida o manifesto do lote.

    Args:
        path: Caminho do manifesto JSON.

    Returns:
        Manifesto validado.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
        pydantic.ValidationError: Se o manifesto ou algum escopo for inválido.
    """
    return Manifest.model_validate_json(path.read_text(encoding="utf-8"))


def _error_item(exc: Exception, expose_details: bool) -> dict:
    """Monta o registro de erro de uma foto, sem score.

    Args:
        exc: Exceção capturada.
        expose_details: Se a mensagem da exceção pode ser incluída.

    Returns:
        Dicionário com status, tipo do erro e, opcionalmente, a mensagem.
    """
    item = {"status": "analysis_not_completed", "error_type": type(exc).__name__}
    if expose_details:
        item["error_message"] = str(exc)[:1000]
    return item


def analyze_request(request: AnalysisRequest, base_dir: Path, cfg: AppConfig, client=None) -> dict:
    """Analisa todas as fotos de um chamado contra o escopo do chamado.

    Args:
        request: Chamado com escopo e fotos.
        base_dir: Pasta base para resolver os caminhos relativos das fotos.
        cfg: Configuração da aplicação.
        client: Cliente bedrock-runtime opcional (ex.: simulado em testes).

    Returns:
        Resultado do chamado com um item por foto (parecer ou erro).
    """
    items = []
    for image in request.images:
        path = (base_dir / image.file).resolve()
        logger.info("Analisando {} / {}", request.request_id, image.file)
        try:
            result = analyze(path.read_bytes(), request.scope, cfg, file_name=path.name, client=client)
        except Exception as exc:
            logger.warning("Falha em {} / {}: {}", request.request_id, image.file, type(exc).__name__)
            result = _error_item(exc, cfg.observability.expose_error_details)
        result["file"] = image.file
        if image.expected_compliant is not None:
            result["expected_compliant"] = image.expected_compliant
            if "visual_compliance_score" in result:
                result["matches_expected"] = (result["status"] == "visually_compliant") == image.expected_compliant
        items.append(result)
    return {"request_id": request.request_id, "scope": request.scope.model_dump(), "images": items}


def summarize(results: list[dict]) -> dict:
    """Resume o lote: status por foto e concordância com o esperado.

    Args:
        results: Resultados por chamado (saída de analyze_request).

    Returns:
        Totais do lote e uma linha por foto.
    """
    rows = []
    for request in results:
        for item in request["images"]:
            rows.append({
                "request_id": request["request_id"],
                "file": item["file"],
                "status": item["status"],
                "score": item.get("visual_compliance_score"),
                "pending_essential_criteria": item.get("pending_essential_criteria"),
                "expected_compliant": item.get("expected_compliant"),
                "matches_expected": item.get("matches_expected"),
            })
    labeled = [r for r in rows if r["matches_expected"] is not None]
    return {
        "images_total": len(rows),
        "images_failed": sum(r["status"] == "analysis_not_completed" for r in rows),
        "labeled_total": len(labeled),
        "labeled_matches": sum(r["matches_expected"] for r in labeled),
        "images": rows,
    }


def run_batch(manifest_path: Path, output_dir: Path, cfg: AppConfig, client=None) -> dict:
    """Executa o lote do manifesto e grava os resultados em disco.

    Grava <output_dir>/<run_id>/<request_id>.json (um por chamado, como o
    outbox da arquitetura) e summary.json com os totais do lote.

    Args:
        manifest_path: Caminho do manifesto JSON.
        output_dir: Pasta base de saída.
        cfg: Configuração da aplicação.
        client: Cliente bedrock-runtime opcional (ex.: simulado em testes).

    Returns:
        Resumo do lote, incluindo run_id e pasta de saída.
    """
    manifest = load_manifest(manifest_path)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for request in manifest.requests:
        result = analyze_request(request, manifest_path.parent, cfg, client=client)
        (run_dir / f"{request.request_id}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        results.append(result)

    summary = {
        "run_id": run_id,
        "output_dir": str(run_dir),
        "model_id": cfg.llm.model_id,
        "region": cfg.llm.region,
        "prompt_version": cfg.prompt.version,
        "rubric_version": cfg.rubric.version,
        **summarize(results),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
