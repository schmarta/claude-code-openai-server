"""``GET /v1/models`` — advertise the Claude models this server exposes.

We list both the short aliases the CLI accepts (``opus``/``sonnet``/…) and the
current concrete ids, so an OpenAI client can pick either. The list is static;
the actual model used per request is whatever the client sends, resolved through
``config.resolve_model``.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.openai_models import ModelCard, ModelList

router = APIRouter()

# Short aliases first (handy as a client `default_model`), then concrete ids.
_MODEL_IDS = [
    "opus",
    "sonnet",
    "haiku",
    "fable",
    "claude-opus-4-8",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
    "claude-fable-5",
]


@router.get("/v1/models", response_model=ModelList)
async def list_models() -> ModelList:
    return ModelList(data=[ModelCard(id=mid) for mid in _MODEL_IDS])


@router.get("/v1/models/{model_id}", response_model=ModelCard)
async def get_model(model_id: str) -> ModelCard:
    return ModelCard(id=model_id)
