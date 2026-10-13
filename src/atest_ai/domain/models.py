"""Contratos de dados (entrada de negócio e saída estruturada do modelo)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

State = Literal["met", "partial", "not_met", "not_verifiable"]
Legibility = Literal["adequate", "limited", "unusable"]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Criterion(Contract):
    id: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=1, max_length=1000)
    weight: int = Field(ge=1, le=100)
    essential: bool = False


class Scope(Contract):
    description: str = Field(min_length=1, max_length=4000)
    criteria: list[Criterion] = Field(min_length=1, max_length=12)

    @model_validator(mode="after")
    def validate_criteria(self):
        """Garante IDs de critério únicos e pesos somando 100.

        Returns:
            O próprio escopo, se válido.

        Raises:
            ValueError: Se houver IDs duplicados ou a soma dos pesos for diferente de 100.
        """
        ids = [c.id for c in self.criteria]
        if len(ids) != len(set(ids)):
            raise ValueError("Criterion IDs must be unique.")
        if sum(c.weight for c in self.criteria) != 100:
            raise ValueError("Criterion weights must sum to 100.")
        return self


class Observation(Contract):
    criterion_id: str
    state: State
    visible_evidence: str = Field(min_length=1, max_length=2000)
    limitation: str = Field(max_length=2000)

    @model_validator(mode="after")
    def validate_limitation(self):
        """Exige limitação preenchida para estados partial e not_verifiable.

        Returns:
            A própria observação, se válida.

        Raises:
            ValueError: Se o estado exigir limitação e ela estiver vazia.
        """
        if self.state in ("partial", "not_verifiable") and not self.limitation.strip():
            raise ValueError("Partial/not verifiable criterion requires a limitation.")
        return self


class VisualReading(Contract):
    """Observações visuais por critério, sem score calculado pelo modelo."""
    legibility: Legibility
    observations: list[Observation] = Field(min_length=1, max_length=12)


class ImageItem(Contract):
    """Foto de um chamado; expected_compliant é rótulo opcional para medir acertos."""
    file: str = Field(min_length=1, max_length=500)
    expected_compliant: bool | None = None


class AnalysisRequest(Contract):
    """Chamado (SETA/SISR) com seu escopo e as fotos enviadas pelo fornecedor."""
    request_id: str = Field(min_length=1, max_length=100)
    scope: Scope
    images: list[ImageItem] = Field(min_length=1, max_length=50)


class Manifest(Contract):
    """Manifesto do lote diário (MVP D+1): chamados a analisar."""
    requests: list[AnalysisRequest] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_request_ids(self):
        """Garante request_id único no lote.

        Returns:
            O próprio manifesto, se válido.

        Raises:
            ValueError: Se houver request_id duplicado.
        """
        ids = [r.request_id for r in self.requests]
        if len(ids) != len(set(ids)):
            raise ValueError("Request IDs must be unique in the manifest.")
        return self
