# Atest.AI — Verificador visual de manutenção

Capacidade 1: aderência visual de **uma** imagem ao escopo do serviço (LangChain + Amazon Bedrock Converse).
O LLM apenas **observa** cada critério (saída estruturada `VisualReading`); o **score é calculado de forma determinística** pelo código.

## Estrutura

```
main.py                    # ponto de entrada: lote do manifesto ou imagem + escopo avulsos
examples/scope.example.json # modelo de escopo (JSON)
config/
  settings.toml            # configurações por ambiente (Dynaconf)
  .secrets.toml            # segredos locais — NÃO versionado (ver .secrets.toml.example)
src/atest_ai/
  config.py                # Dynaconf -> objetos tipados (Pydantic)
  domain/                  # regras de negócio puras, sem LLM
    models.py              #   contratos: Scope, Criterion, Observation, VisualReading
    scoring.py             #   validação semântica + score determinístico
  prompts/
    loader.py              # monta o ChatPromptTemplate multimodal
    templates/<versao>/    # prompts versionados (system.md, human.md)
  llm/bedrock.py           # fábrica do ChatBedrockConverse (cliente injetável)
  chains/visual_reading.py # chain LCEL + guardrails da resposta
  services/                # image, analysis (pronta para a futura API), batch (MVP D+1)
tests/                     # pytest; nenhuma chamada externa
data/input/manifest.json   # lote de teste: chamados, escopos e fotos
data/input/images/         # fotos (ignoradas pelo git)
data/output/<run_id>/      # um JSON por chamado + summary.json (ignorado pelo git)
docs/architecture/         # arquitetura alvo (MVP batch D+1 e versão via API)
```

## Instalação

```bash
python -m pip install -e ".[dev]"
cp config/.secrets.toml.example config/.secrets.toml   # opcional
```

## Execução

O produto recebe **imagens e o escopo do chamado** (critérios com pesos somando 100); não há
escopos fixos no código. Hoje a entrada é o `main.py`; a futura API chamará
`atest_ai.services.analysis.analyze(image_bytes, scope, cfg)` da mesma forma.

### Teste real (passo a passo)

```bash
aws sso login --profile PowerUserAccess-755670014224   # renova a sessão SSO (expira periodicamente)
.venv\Scripts\python -m pip install -e ".[dev]"        # inclui truststore
.venv\Scripts\python main.py                           # roda o lote de data/input/manifest.json
```

- **Lote (padrão, MVP D+1):** `data/input/manifest.json` lista os chamados, o escopo de cada um e
  suas fotos (caminhos relativos ao manifesto). `expected_compliant` é opcional e serve só para
  medir acertos no `summary.json`; nunca é enviado ao modelo.
- **Saída:** `data/output/<run_id>/<request_id>.json` (um parecer por foto) e `summary.json`
  (status, score e concordância por foto). O resumo também sai em stdout e o progresso em stderr.
- **Erros:** uma foto com falha vira `analysis_not_completed` (sem score) e não interrompe o lote;
  o processo termina com exit code 1 se alguma foto falhar. Em `development`, o erro inclui
  `error_message` (`observability.expose_error_details`).
- **Avulso:** `main.main(Path("foto.jpg"), Path("escopo.json"))` (modelo em `examples/scope.example.json`).

### Bedrock

Configurado em `[default.llm]` de `config/settings.toml`:

- `model_id = "global.anthropic.claude-opus-5-5"` e `region = "sa-east-1"`. Em São Paulo só existem
  perfis de inferência `global.` para o Claude: a inferência pode ser roteada para outras regiões AWS.
- `reasoning_effort = "medium"`: o thinking do Opus 5.5 não pode ser desligado; o esforço controla profundidade e custo.
- A saída estruturada usa o JSON Schema nativo do Bedrock (`outputConfig`), não tool calling: os modelos
  Claude atuais rejeitam `tool_choice` forçado.
- Credenciais: `aws.profile` em `config/.secrets.toml` (perfil SSO) ou a cadeia padrão AWS. A identidade
  precisa de `bedrock:InvokeModel` no perfil de inferência e nos modelos de destino.
- Rede: em `development`, `network.use_system_trust_store = true` valida TLS pelo repositório do Windows,
  necessário quando um antivírus (ex.: Avast) ou proxy inspeciona HTTPS com CA própria.

Sobrescritas sem editar arquivos:

```bash
ATEST_ENV=production
ATEST_EXECUTION__MANIFEST=data/input/outro_manifesto.json
ATEST_LLM__MODEL_ID=global.anthropic.claude-sonnet-5-5
ATEST_LLM__REASONING_EFFORT=high
```

## Testes

```bash
python -m pytest
```

## Limites

Score de evidência visual, não probabilidade, atestação funcional, autenticidade, data de execução
ou confirmação do local específico. Pesos, estados e faixas (`rubric`) são hipóteses de piloto, não
valores calibrados. Tracing externo fica desligado por padrão (`observability.tracing_enabled = false`).
