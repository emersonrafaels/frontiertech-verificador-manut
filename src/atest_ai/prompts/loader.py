"""Prompts versionados em arquivos (templates/<versao>/{system,human}.md).

Os templates usam sintaxe f-string do LangChain: chaves literais devem ser
escapadas como {{ }}. Placeholders são interpolados uma única vez.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

TEMPLATES_DIR = Path(__file__).parent / "templates"


@lru_cache
def read_template(version: str, name: str) -> str:
    """Lê um template de prompt versionado (com cache).

    Args:
        version: Versão do prompt (nome do diretório em templates/).
        name: Nome do template, sem extensão (ex.: "system", "human").

    Returns:
        Conteúdo do template, sem espaços nas bordas.

    Raises:
        FileNotFoundError: Se o template não existir.
    """
    path = TEMPLATES_DIR / version / f"{name}.md"
    if not path.is_file():
        raise FileNotFoundError(f"Prompt template not found: {version}/{name}.md")
    return path.read_text(encoding="utf-8").strip()


def build_prompt(version: str):
    """Monta o prompt multimodal: system + (texto com escopo, imagem em base64).

    Args:
        version: Versão do prompt (nome do diretório em templates/).

    Returns:
        ChatPromptTemplate com os placeholders scope_json e image_base64.
    """
    from langchain_core.prompts import ChatPromptTemplate

    return ChatPromptTemplate.from_messages([
        ("system", read_template(version, "system")),
        ("human", [
            {"type": "text", "text": read_template(version, "human")},
            {"type": "image_url", "image_url": {
                "url": "data:image/jpeg;base64,{image_base64}"
            }},
        ]),
    ])
