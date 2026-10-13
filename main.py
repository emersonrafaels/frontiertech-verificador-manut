"""
Atest.AI — capacidade: aderência visual de imagens ao escopo do chamado.

Modo lote (padrão, MVP D+1): lê o manifesto de execution.manifest, com os
chamados, o escopo de cada um e suas fotos, e grava um JSON por chamado e um
summary.json em execution.output_dir/<run_id>/.

Modo avulso: main(Path("foto.jpg"), Path("escopo.json")) analisa uma imagem
contra um escopo e imprime o resultado em stdout.
"""

import json
import sys
from pathlib import Path

from loguru import logger

from atest_ai.config import AppConfig, load_config
from atest_ai.domain.models import Scope
from atest_ai.services.analysis import analyze
from atest_ai.services.batch import run_batch


def load_inputs(image_path: Path, scope_path: Path) -> tuple[bytes, Scope]:
    """Lê a imagem e o escopo de arquivos locais.

    Args:
        image_path: Caminho da imagem (JPEG, PNG ou WEBP).
        scope_path: Caminho do escopo em JSON (ver examples/scope.example.json).

    Returns:
        Tupla (bytes da imagem, escopo validado).

    Raises:
        FileNotFoundError: Se algum dos arquivos não existir.
        pydantic.ValidationError: Se o escopo for inválido.
    """
    image = image_path.read_bytes()
    scope = Scope.model_validate_json(scope_path.read_text(encoding="utf-8"))
    return image, scope


def run_single(cfg: AppConfig, image_path: Path, scope_path: Path) -> dict:
    """Analisa uma imagem avulsa contra um escopo e imprime o resultado.

    Args:
        cfg: Configuração da aplicação.
        image_path: Caminho da imagem.
        scope_path: Caminho do escopo JSON.

    Returns:
        Resultado da análise.
    """
    image, scope = load_inputs(image_path, scope_path)
    result = analyze(image, scope, cfg, file_name=image_path.name)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def run(cfg: AppConfig, image_path: Path | None = None, scope_path: Path | None = None) -> dict:
    """Executa o modo avulso (imagem + escopo) ou, sem eles, o lote do manifesto.

    Args:
        cfg: Configuração da aplicação.
        image_path: Caminho de uma imagem avulsa (exige scope_path).
        scope_path: Caminho do escopo JSON da imagem avulsa.

    Returns:
        Resultado da imagem avulsa ou resumo do lote.

    Raises:
        ValueError: Se só um de image_path/scope_path for informado, ou se o
            lote não tiver manifesto e pasta de saída configurados.
    """
    if image_path or scope_path:
        if not (image_path and scope_path):
            raise ValueError("Single mode requires both an image and a scope.")
        return run_single(cfg, image_path, scope_path)
    if not cfg.execution.manifest or not cfg.execution.output_dir:
        raise ValueError("Set execution.manifest and execution.output_dir.")
    summary = run_batch(cfg.execution.manifest, cfg.execution.output_dir, cfg)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def configure_logging(level: str) -> None:
    """Configura o loguru para escrever o progresso em stderr.

    O stdout fica reservado ao JSON de resultado. As mensagens de log não
    incluem conteúdo bruto do modelo nem credenciais.

    Args:
        level: Nível mínimo de log (ex.: "DEBUG", "INFO", "WARNING").
    """
    logger.remove()
    logger.add(
        sys.stderr, level=level.upper(), colorize=None,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    )


def main(image_path: Path | None = None, scope_path: Path | None = None) -> None:
    """Ponto de entrada: carrega a configuração e executa a análise.

    Em erro, imprime JSON sanitizado em stderr (sem conteúdo bruto nem
    credenciais; a mensagem só aparece com observability.expose_error_details)
    e encerra com exit code 1, sem score. No lote, encerra com 1 também se
    alguma foto não foi analisada.

    Args:
        image_path: Caminho de uma imagem avulsa; sem ele, roda o lote.
        scope_path: Caminho do escopo JSON da imagem avulsa.
    """
    expose_details = False
    try:
        cfg = load_config()
        configure_logging(cfg.observability.log_level)
        expose_details = cfg.observability.expose_error_details
        result = run(cfg, image_path, scope_path)
    except Exception as exc:
        # Sem conteúdo bruto/credenciais/log do SDK. Detalhe sanitizado em monitoramento interno.
        error = {"status": "analysis_not_completed", "error_type": type(exc).__name__}
        if expose_details:
            error["error_message"] = str(exc)[:1000]
        print(json.dumps(error, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1)
    if result.get("images_failed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
