"""Fábrica do modelo de chat (Bedrock Converse via LangChain)."""

from __future__ import annotations

from atest_ai.config import AWSConfig, LLMConfig, NetworkConfig


def apply_network_config(network: NetworkConfig) -> None:
    """Aplica ajustes de rede do processo antes de criar clientes AWS.

    Com use_system_trust_store, o TLS passa a validar certificados pelo
    repositório do sistema operacional (necessário quando um antivírus ou
    proxy corporativo inspeciona HTTPS com uma CA própria).

    Args:
        network: Configuração de rede.
    """
    if network.use_system_trust_store:
        import truststore

        truststore.inject_into_ssl()


def create_aws_session(aws: AWSConfig):
    """Cria a sessão boto3.

    Credenciais explícitas só são usadas se definidas em .secrets.toml;
    caso contrário vale a cadeia padrão AWS (perfil SSO / IAM role).

    Args:
        aws: Configuração AWS (perfil e credenciais opcionais).

    Returns:
        Sessão boto3 pronta para criar clientes.
    """
    import boto3

    kwargs = {}
    if aws.profile:
        kwargs["profile_name"] = aws.profile
    if aws.access_key_id.get_secret_value():
        kwargs["aws_access_key_id"] = aws.access_key_id.get_secret_value()
        kwargs["aws_secret_access_key"] = aws.secret_access_key.get_secret_value()
        if aws.session_token.get_secret_value():
            kwargs["aws_session_token"] = aws.session_token.get_secret_value()
    return boto3.Session(**kwargs)


def create_model(llm: LLMConfig, aws: AWSConfig, client=None, network: NetworkConfig | None = None):
    """Cria o ChatBedrockConverse usado pela chain.

    O modelo precisa suportar imagem + saída estruturada nativa (JSON Schema).

    Args:
        llm: Configuração do modelo (ID, região, esforço, timeouts, retries).
        aws: Configuração de credenciais AWS.
        client: Cliente bedrock-runtime opcional (ex.: simulado em testes).
        network: Configuração de rede aplicada antes de criar o cliente real.

    Returns:
        Instância de ChatBedrockConverse.

    Raises:
        ValueError: Se model_id ou region não estiverem configurados.
    """
    from langchain_aws import ChatBedrockConverse

    if not llm.model_id or not llm.region:
        raise ValueError("Set llm.model_id and llm.region (settings.toml or ATEST_LLM__*).")
    if client is None:
        from botocore.config import Config

        if network is not None:
            apply_network_config(network)
        client = create_aws_session(aws).client(
            "bedrock-runtime", region_name=llm.region, config=Config(
                connect_timeout=llm.connect_timeout, read_timeout=llm.read_timeout,
                retries={"mode": "standard", "total_max_attempts": llm.max_attempts},
            ),
        )
    # Sem temperature: os modelos Claude atuais rejeitam parâmetros de amostragem.
    return ChatBedrockConverse(
        client=client, model_id=llm.model_id, region_name=llm.region,
        max_tokens=llm.max_tokens, reasoning_effort=llm.reasoning_effort,
    )
