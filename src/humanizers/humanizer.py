# src/humanizers/humanizer.py
"""
Provider-agnostic humanizer wrapper.

Supports:
• OpenAI Chat (standard & fine-tuned)
• Google Gemini
• Anthropic Claude

v1.4
────
• System prompts properly implemented for all providers
• Added Claude support with proper rate limiting
• Global, thread-safe rate-limiting for all providers
• Auto-retry for rate limits and transient errors
"""

from __future__ import annotations
from typing import Literal, Any

import random
import time

from openai import OpenAI
from google import genai
from google.genai import types
import anthropic

from ..config import (
    OPENAI_API_KEY,
    HUMANIZER_OPENAI_API_KEY,
    GEMINI_API_KEY,
    CLAUDE_API_KEY,
)
from ..models import MODEL_REGISTRY
from ..prompts import (
    DEFAULT_DOC_SYSTEM_PROMPT,
    DEFAULT_PARA_SYSTEM_PROMPT,
    FINETUNED_DOC_SYSTEM_PROMPT,
    FINETUNED_PARA_SYSTEM_PROMPT,
)
from ..rate_limiter import wait as _rate_wait


# ────────────────────────── clients ────────────────────────────
_openai_std = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
_openai_ft = OpenAI(api_key=HUMANIZER_OPENAI_API_KEY) if HUMANIZER_OPENAI_API_KEY else None
_claude_client = anthropic.Anthropic(api_key=CLAUDE_API_KEY) if CLAUDE_API_KEY else None

if GEMINI_API_KEY:
    _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
else:
    _gemini_client = None


# ─────────────────────── helper functions ─────────────────────
def _openai_call(text: str, model: str, api: OpenAI, system_prompt: str) -> str:
    """
    One-shot call with:
      • Rate limiting
      • 3 automatic retries with exponential back-off (2 s → 4 s → 8 s).
      • System prompt properly set in messages
    """
    for attempt in range(1, 4):
        try:
            _rate_wait("openai")  # Apply rate limiting
            
            resp = api.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": text},
                ],
                temperature=1.0,
                max_tokens=2048,
                timeout=300,  # 5 minute timeout
            )
            return resp.choices[0].message.content.strip()

        except Exception as exc:
            if attempt == 3:
                raise    # bubble up after final failure
            time.sleep(2 ** attempt)   # back-off 2 / 4 / 8 s


def _claude_call(text: str, model: str, system_prompt: str) -> str:
    """
    Claude API call with:
      • Rate limiting
      • 3 automatic retries with exponential back-off
      • System prompt properly set
    """
    for attempt in range(1, 4):
        try:
            _rate_wait("claude")  # Apply rate limiting
            
            response = _claude_client.messages.create(
                model=model,
                max_tokens=2048,
                temperature=1.0,
                system=system_prompt,  # System prompt properly set
                messages=[
                    {"role": "user", "content": text}
                ]
            )
            
            # Extract text from response
            if hasattr(response.content[0], 'text'):
                return response.content[0].text.strip()
            else:
                return str(response.content).strip()
                
        except Exception as exc:
            msg = str(exc).lower()
            # Check for rate limit errors
            if ("rate" in msg or "429" in msg) and attempt < 3:
                time.sleep(2 ** attempt)  # back-off 2 / 4 / 8 s
                continue
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def _gemini_generate(model_id: str, text: str, system_prompt: str, *, max_retries: int = 10):
    """
    Gemini wrapper that:
    • honours the global rate-limit (700 req/min)
    • retries after a 429/Quota error with exponential back-off
    • Uses system_instruction properly
    """
    delay = 5  # start with smaller delay due to higher rate limit

    for attempt in range(1, max_retries + 1):
        _rate_wait("gemini")  # ⇠ blocks until a token is free

        try:
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=text)],
                )
            ]
            
            resp = _gemini_client.models.generate_content(
                model=model_id,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=1.0,
                    system_instruction=system_prompt,  # System prompt properly set
                ),
            )
            return resp
            
        except Exception as exc:
            msg = str(exc).lower()
            if ("quota" in msg or "rate" in msg or "429" in msg) and attempt < max_retries:
                time.sleep(delay + random.uniform(0, 2))  # jitter
                delay = min(delay * 1.5, 60)             # cap at 60s
                continue
            raise


def _gemini_call(text: str, model: str, system_prompt: str) -> str:
    """Call Gemini with proper system instructions."""
    resp = _gemini_generate(model, text, system_prompt)
    return resp.text.strip()


def _select_prompt(prompt_id: str, mode: Literal["doc", "para"]) -> str:
    """Select the appropriate prompt based on prompt_id and mode."""
    if prompt_id == "default":
        return DEFAULT_DOC_SYSTEM_PROMPT if mode == "doc" else DEFAULT_PARA_SYSTEM_PROMPT
    if prompt_id == "finetuned":
        return FINETUNED_DOC_SYSTEM_PROMPT if mode == "doc" else FINETUNED_PARA_SYSTEM_PROMPT
    raise ValueError(f"Unknown prompt_id '{prompt_id}'")


# ───────────────────────── public API ──────────────────────────
def humanize(
    text: str,
    display_name: str,
    mode: Literal["doc", "para"] = "para",
    **kwargs: Any,
) -> str:
    """
    Rewrite *text* using the model identified by *display_name*.

    Extra keyword arguments are accepted and ignored so that
    upstream callers can pass contextual data (e.g. `log=…`)
    without breaking the interface.
    
    All providers now properly use system prompts/instructions.
    """
    # Silently discard unrecognised kwargs (e.g. log callbacks)
    kwargs.pop("log", None)

    meta = MODEL_REGISTRY[display_name]
    provider = meta["provider"]
    model_id = meta["model"]
    prompt_id = meta["prompt_id"]

    system_prompt = _select_prompt(prompt_id, mode)

    if provider == "openai":
        if not _openai_std:
            raise ValueError("OpenAI API key not configured")
        return _openai_call(text, model_id, _openai_std, system_prompt)
    
    if provider == "openai_ft":
        if not _openai_ft:
            raise ValueError("Humanizer OpenAI API key not configured")
        return _openai_call(text, model_id, _openai_ft, system_prompt)
    
    if provider == "claude":
        if not _claude_client:
            raise ValueError("Claude API key not configured")
        return _claude_call(text, model_id, system_prompt)
    
    if provider == "gemini":
        if not _gemini_client:
            raise ValueError("Gemini API key not configured")
        return _gemini_call(text, model_id, system_prompt)

    raise ValueError(f"Unknown provider '{provider}' for model {display_name}")