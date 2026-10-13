"""Configuração via Dynaconf (config/settings.toml + config/.secrets.toml).

O Dynaconf é lido apenas aqui; o restante do código recebe objetos tipados
(Pydantic), o que facilita testes e evita acesso global a `settings`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from dynaconf import Dynaconf
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"

ReasoningEffort = Literal["low", "medium", "high", "xhigh", "max"]


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExecutionConfig(_Section):
    """Entradas locais do main.py (a futura API recebe-as na requisição)."""
    manifest: Path | None = None
    output_dir: Path | None = None

    @field_validator("manifest", "output_dir", mode="before")
    @classmethod
    def _empty_as_none(cls, value):
        """Converte string vazia do TOML em None nos campos opcionais.

        Args:
            value: Valor bruto recebido do Dynaconf.

        Returns:
            None se o valor for "", caso contrário o próprio valor.
        """
        return None if value == "" else value


class LLMConfig(_Section):
    provider: Literal["bedrock_converse"] = "bedrock_converse"
    model_id: str = ""
    region: str = ""
    max_tokens: int = Field(16000, ge=1)
    reasoning_effort: ReasoningEffort | None = "medium"
    connect_timeout: int = Field(10, ge=1)
    read_timeout: int = Field(300, ge=1)
    max_attempts: int = Field(3, ge=1)
    valid_stop_reasons: frozenset[str] = frozenset({"end_turn"})

    @field_validator("reasoning_effort", mode="before")
    @classmethod
    def _empty_effort_as_none(cls, value):
        """Converte string vazia em None (não envia esforço de raciocínio).

        Args:
            value: Valor bruto recebido do Dynaconf.

        Returns:
            None se o valor for "", caso contrário o próprio valor.
        """
        return None if value == "" else value


class PromptConfig(_Section):
    version: str = "visual-langchain-0.3"


class ObservabilityConfig(_Section):
    tracing_enabled: bool = False
    expose_error_details: bool = False
    log_level: Literal["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    @field_validator("log_level", mode="before")
    @classmethod
    def _upper_log_level(cls, value):
        """Aceita o nível de log em qualquer caixa (ex.: "info").

        Args:
            value: Valor bruto recebido do Dynaconf.

        Returns:
            O nível em maiúsculas, se for string.
        """
        return value.upper() if isinstance(value, str) else value


class NetworkConfig(_Section):
    use_system_trust_store: bool = False


class ImageConfig(_Section):
    max_original_bytes: int = 20_000_000
    max_pixels: int = 30_000_000
    max_side: int = 1568
    jpeg_quality: int = Field(90, ge=1, le=95)
    max_prepared_bytes: int = 3_500_000
    accepted_formats: frozenset[str] = frozenset({"JPEG", "PNG", "WEBP"})


class RubricConfig(_Section):
    version: str = "visual-0.1-experimental"
    compliance_threshold: float = Field(85, ge=0, le=100)
    factors: dict[str, float] = {
        "met": 1.0, "partial": 0.5,
        "not_met": 0.0, "not_verifiable": 0.0,
    }


class AWSConfig(_Section):
    profile: str = ""
    access_key_id: SecretStr = SecretStr("")
    secret_access_key: SecretStr = SecretStr("")
    session_token: SecretStr = SecretStr("")


class AppConfig(_Section):
    execution: ExecutionConfig = ExecutionConfig()
    llm: LLMConfig = LLMConfig()
    prompt: PromptConfig = PromptConfig()
    observability: ObservabilityConfig = ObservabilityConfig()
    network: NetworkConfig = NetworkConfig()
    image: ImageConfig = ImageConfig()
    rubric: RubricConfig = RubricConfig()
    aws: AWSConfig = AWSConfig()


def _normalize(value: Any) -> Any:
    """Coloca as chaves em minúsculas, recursivamente.

    Variáveis de ambiente (ex.: ATEST_LLM__MODEL_ID) chegam com chaves em
    maiúsculas; os modelos Pydantic esperam minúsculas.

    Args:
        value: Dicionário, lista ou valor escalar vindo do Dynaconf.

    Returns:
        Estrutura equivalente com todas as chaves de dicionário em minúsculas.
    """
    if isinstance(value, dict):
        return {str(k).lower(): _normalize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize(v) for v in value]
    return value


def _resolve_path(path: Path | None) -> Path | None:
    """Resolve caminhos relativos a partir da raiz do projeto.

    Args:
        path: Caminho configurado, absoluto ou relativo, ou None.

    Returns:
        Caminho absoluto, ou None se nenhum caminho foi configurado.
    """
    if path is None or path.is_absolute():
        return path
    return PROJECT_ROOT / path


def create_settings() -> Dynaconf:
    """Cria a instância Dynaconf com settings.toml e .secrets.toml.

    O ambiente ativo é escolhido por ATEST_ENV (padrão: development) e
    qualquer chave pode ser sobrescrita por variáveis com prefixo ATEST_.

    Returns:
        Objeto Dynaconf com as configurações mescladas.
    """
    return Dynaconf(
        envvar_prefix="ATEST",
        root_path=str(CONFIG_DIR),
        settings_files=["settings.toml", ".secrets.toml"],
        environments=True,
        env_switcher="ATEST_ENV",
        default_env="default",
        env="development",
        merge_enabled=True,
    )


@lru_cache(maxsize=1)
def load_config() -> AppConfig:
    """Carrega e valida a configuração da aplicação (com cache).

    Returns:
        AppConfig imutável, com caminhos já resolvidos para absolutos.

    Raises:
        pydantic.ValidationError: Se alguma chave estiver ausente, com tipo
            inválido ou não for reconhecida.
    """
    settings = create_settings()
    sections = {name: settings.get(name, {}) for name in AppConfig.model_fields}
    config = AppConfig.model_validate(_normalize(sections))
    execution = config.execution.model_copy(update={
        "manifest": _resolve_path(config.execution.manifest),
        "output_dir": _resolve_path(config.execution.output_dir),
    })
    return config.model_copy(update={"execution": execution})
