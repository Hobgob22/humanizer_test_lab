"""
Provider-agnostic humanizer wrapper.

Supports:
• OpenAI Chat (standard & fine-tuned)
• Google Gemini

v1.2
────
• Fixed OpenAI API timeout parameter (was request_timeout, now timeout)
• Global, thread-safe rate-limiting and auto-retry for Gemini so we
  never exceed the free-tier quota.
"""

from __future__ import annotations
from typing import Literal

import random
import time

from openai import OpenAI
import google.generativeai as genai

from ..config import (
    OPENAI_API_KEY,
    HUMANIZER_OPENAI_API_KEY,
    GEMINI_API_KEY,
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
_openai_std = OpenAI(api_key=OPENAI_API_KEY)
_openai_ft = OpenAI(api_key=HUMANIZER_OPENAI_API_KEY)

genai.configure(api_key=GEMINI_API_KEY)


# ─────────────────────── helper functions ─────────────────────
def _openai_call(text: str, model: str, api: OpenAI, system_prompt: str) -> str:
    """
    One-shot call with:
      • 60 s hard timeout,
      • 3 automatic retries with exponential back-off (2 s → 4 s → 8 s).
    """
    for attempt in range(1, 4):
        try:
            resp = api.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": text},
                ],
                temperature=1.0,
                max_tokens=2048,
                timeout=60,    # correct parameter name for OpenAI client
            )
            return resp.choices[0].message.content.strip()

        except Exception as exc:
            if attempt == 3:
                raise    # bubble up after final failure
            time.sleep(2 ** attempt)   # back-off 2 / 4 / 8 s



def _gemini_generate(model_id: str, parts: list[str], *, max_retries: int = 10):
    """
    Wrapper that
    • honours the global rate-limit
    • retries after a 429/Quota error with exponential back-off
    """
    gmodel = genai.GenerativeModel(model_id)
    delay = 60  # start with a full-window wait

    for attempt in range(1, max_retries + 1):
        _rate_wait("gemini")  # ⇠ blocks until a token is free

        try:
            return gmodel.generate_content(parts, generation_config={"temperature": 1.0})
        except Exception as exc:
            msg = str(exc).lower()
            if ("quota" in msg or "rate" in msg or "429" in msg) and attempt < max_retries:
                time.sleep(delay + random.uniform(0, 2))  # jitter
                delay = min(delay * 1.5, 300)             # cap at 5 min
                continue
            raise


def _gemini_call(text: str, model: str, system_prompt: str) -> str:
    resp = _gemini_generate(model, [system_prompt, text])
    return resp.text.strip()


def _select_prompt(prompt_id: str, mode: Literal["doc", "para"]) -> str:
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
) -> str:
    """
    Rewrite *text* using the model identified by *display_name*.
    """
    meta = MODEL_REGISTRY[display_name]
    provider = meta["provider"]
    model_id = meta["model"]
    prompt_id = meta["prompt_id"]

    system_prompt = _select_prompt(prompt_id, mode)

    if provider == "openai":
        return _openai_call(text, model_id, _openai_std, system_prompt)
    if provider == "openai_ft":
        return _openai_call(text, model_id, _openai_ft, system_prompt)
    if provider == "gemini":
        return _gemini_call(text, model_id, system_prompt)

    raise ValueError(f"Unknown provider '{provider}' for model {display_name}")