from atest_ai.config import PROJECT_ROOT, load_config
from atest_ai.prompts.loader import read_template


def _reload():
    """Limpa o cache e recarrega a configuração."""
    load_config.cache_clear()
    return load_config()


def test_loads_settings_toml(monkeypatch):
    """Valores de settings.toml são carregados e caminhos resolvidos."""
    monkeypatch.delenv("ATEST_ENV", raising=False)
    cfg = _reload()
    assert cfg.llm.model_id == "global.anthropic.claude-opus-5-5"
    assert cfg.llm.reasoning_effort == "medium"
    assert cfg.observability.log_level == "INFO"
    assert cfg.rubric.factors["partial"] == 0.5
    assert cfg.llm.region == "sa-east-1"
    assert cfg.execution.manifest == PROJECT_ROOT / "data/input/manifest.json"
    assert cfg.execution.output_dir == PROJECT_ROOT / "data/output"
    assert cfg.network.use_system_trust_store is True  # [development]


def test_environment_variable_override(monkeypatch):
    """Variáveis ATEST_* sobrescrevem chaves sem perder as demais."""
    monkeypatch.setenv("ATEST_LLM__MODEL_ID", "model-x")
    monkeypatch.setenv("ATEST_RUBRIC__COMPLIANCE_THRESHOLD", "90")
    cfg = _reload()
    assert cfg.llm.model_id == "model-x"
    assert cfg.llm.max_tokens == 16000  # merge preserva demais chaves
    assert cfg.rubric.compliance_threshold == 90


def test_production_environment(monkeypatch):
    """ATEST_ENV=production usa os defaults seguros (sem detalhes de erro nem trust store)."""
    monkeypatch.setenv("ATEST_ENV", "production")
    cfg = _reload()
    assert cfg.observability.expose_error_details is False
    assert cfg.network.use_system_trust_store is False


def test_versioned_prompt_template():
    """O template da versão configurada é encontrado."""
    assert read_template(load_config().prompt.version, "system").startswith("Você avalia UMA imagem")


def teardown_module():
    """Limpa o cache para não vazar configuração entre módulos."""
    load_config.cache_clear()
