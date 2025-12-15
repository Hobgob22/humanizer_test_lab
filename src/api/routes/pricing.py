from __future__ import annotations

from fastapi import APIRouter, Depends

from ..dependencies import verify_api_key
from ...pricing import MODEL_PRICING

router = APIRouter()


@router.get("/")
async def list_model_pricing(api_key: str = Depends(verify_api_key)):
    return MODEL_PRICING

