from __future__ import annotations

import math
from typing import Any, Dict, Optional

MODEL_PRICING: Dict[str, Dict[str, Any]] = {
    # OpenAI models
    "gpt-5": {"input_per_mtok": 1.25, "output_per_mtok": 10.0, "provider": "openai", "model_id": "gpt-5"},
    "gpt-5-chat-latest": {"input_per_mtok": 1.25, "output_per_mtok": 10.0, "provider": "openai", "model_id": "gpt-5-chat-latest"},
    "gpt-5.1": {"input_per_mtok": 1.25, "output_per_mtok": 10.0, "provider": "openai", "model_id": "gpt-5.1"},
    "gpt-5.1-none": {"input_per_mtok": 1.25, "output_per_mtok": 10.0, "provider": "openai", "model_id": "gpt-5.1"},
    "gpt-5.1-chat-latest": {"input_per_mtok": 1.25, "output_per_mtok": 10.0, "provider": "openai", "model_id": "gpt-5.1-chat-latest"},
    "gpt-5-mini": {"input_per_mtok": 0.25, "output_per_mtok": 2.0, "provider": "openai", "model_id": "gpt-5-mini"},
    "gpt-5-nano": {"input_per_mtok": 0.05, "output_per_mtok": 0.4, "provider": "openai", "model_id": "gpt-5-nano"},
    "gpt-4.1": {"input_per_mtok": 2.0, "output_per_mtok": 8.0, "provider": "openai", "model_id": "gpt-4.1"},
    "gpt-4.1-mini": {"input_per_mtok": 0.8, "output_per_mtok": 3.2, "provider": "openai", "model_id": "gpt-4.1-mini"},
    "gpt-4o": {"input_per_mtok": 2.5, "output_per_mtok": 10.0, "provider": "openai", "model_id": "gpt-4o"},
    "gpt-o4-mini": {"input_per_mtok": 0.15, "output_per_mtok": 0.6, "provider": "openai", "model_id": "gpt-o4-mini"},
    "gpt-o3": {"input_per_mtok": 10.0, "output_per_mtok": 40.0, "provider": "openai", "model_id": "o3"},

    # Anthropic Claude models
    "claude-sonnet-4.5": {"input_per_mtok": 3.0, "output_per_mtok": 15.0, "provider": "claude", "model_id": "claude-sonnet-4-5"},
    "claude-haiku-4.5": {"input_per_mtok": 1.0, "output_per_mtok": 5.0, "provider": "claude", "model_id": "claude-haiku-4-5"},
    "claude-sonnet-4": {"input_per_mtok": 3.0, "output_per_mtok": 15.0, "provider": "claude", "model_id": "claude-sonnet-4"},
    "claude-sonnet-3.7": {"input_per_mtok": 3.0, "output_per_mtok": 15.0, "provider": "claude", "model_id": "claude-3-7-sonnet-latest"},
    "claude-sonnet-3.5": {"input_per_mtok": 3.0, "output_per_mtok": 15.0, "provider": "claude", "model_id": "claude-3-5-sonnet-latest"},
    "claude-haiku-3.5": {"input_per_mtok": 0.8, "output_per_mtok": 4.0, "provider": "claude", "model_id": "claude-3-5-haiku-latest"},

    # Groq / Moonshot AI
    "kimi-k2": {"input_per_mtok": 1.0, "output_per_mtok": 3.0, "provider": "groq", "model_id": "moonshotai/kimi-k2-instruct-0905"},

    # Google Gemini
    "gemini-2.5-flash": {"input_per_mtok": 0.3, "output_per_mtok": 2.5, "provider": "gemini", "model_id": "gemini-2.5-flash"},
    "gemini-2.5-flash-lite": {"input_per_mtok": 0.1, "output_per_mtok": 0.4, "provider": "gemini", "model_id": "gemini-2.5-flash-lite"},
    "gemini-2.0-flash": {"input_per_mtok": 0.1, "output_per_mtok": 0.4, "provider": "gemini", "model_id": "gemini-2.0-flash"},
    "gemini-2.5-pro": {"input_per_mtok": 1.25, "output_per_mtok": 10.0, "provider": "gemini", "model_id": "gemini-2.5-pro-latest"},
    "gemini-3-pro": {"input_per_mtok": 1.5, "output_per_mtok": 12.0, "provider": "gemini", "model_id": "gemini-3-pro"},
}


def estimate_token_usage(text: str, *, thinking_tokens: int = 0) -> Dict[str, int]:
    """
    Roughly convert characters to tokens using the heuristic 4 chars ≈ 1 token.
    Also estimate output tokens as 35% of input with a minimum floor so that
    long JSON responses are accounted for.
    """
    clean_text = text or ""
    input_tokens = max(1, math.ceil(len(clean_text) / 4))
    output_tokens = max(1200, math.ceil(input_tokens * 0.35))
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "thinking_tokens": max(0, thinking_tokens),
    }


def estimate_cost(model_id: str, *, input_tokens: int, output_tokens: int, thinking_tokens: int = 0) -> Optional[Dict[str, Any]]:
    pricing = MODEL_PRICING.get(model_id)
    if not pricing:
        return None

    input_cost = (input_tokens / 1_000_000) * pricing["input_per_mtok"]
    output_cost = ((output_tokens + thinking_tokens) / 1_000_000) * pricing["output_per_mtok"]
    total_cost = input_cost + output_cost

    return {
        "model": model_id,
        "unit_rates": {"input_per_mtok": pricing["input_per_mtok"], "output_per_mtok": pricing["output_per_mtok"]},
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "thinking_tokens": thinking_tokens,
        "estimated_cost": round(total_cost, 6),
        "provider": pricing.get("provider"),
    }

