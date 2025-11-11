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
    LEGACY_FINETUNED_DOC_SYSTEM_PROMPT,
    LEGACY_FINETUNED_PARA_SYSTEM_PROMPT,
    FINETUNED_DOC_SYSTEM_PROMPT1,
    FINETUNED_DOC_SYSTEM_PROMPT2,
    FINETUNED_PARA_SYSTEM_PROMPT1,
    FINETUNED_PARA_SYSTEM_PROMPT2,
    # Dynamic prompts
    MINIMAL_DOC_SYSTEM_PROMPT,
    MINIMAL_PARA_SYSTEM_PROMPT,
    COMPACT_DOC_SYSTEM_PROMPT,
    COMPACT_PARA_SYSTEM_PROMPT,
    RICH_SYSTEM_PROMPT_STANDARD_DOC,
    RICH_SYSTEM_PROMPT_STANDARD_PARA,
    RICH_SYSTEM_PROMPT_WITH_COUNTER_EXAMPLES_DOC,
    RICH_SYSTEM_PROMPT_WITH_COUNTER_EXAMPLES_PARA,
    RICH_SYSTEM_PROMPT_WITH_NEGATIVE_EXAMPLES_DOC,
    RICH_SYSTEM_PROMPT_WITH_NEGATIVE_EXAMPLES_PARA,
    RICH_SYSTEM_PROMPT_WITH_FOCUS_AREAS_DOC,
    RICH_SYSTEM_PROMPT_WITH_FOCUS_AREAS_PARA,
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


# ─────────────────────── prompt override handling ─────────────────────
#   display_name  →  "v1" | "v2"
PROMPT_OVERRIDES: dict[str, str] = {}

def set_prompt_override(model_name: str, variant: str) -> None:
    """Register a single prompt variant for *model_name*."""
    if variant not in ("v1", "v2"):
        raise ValueError("variant must be 'v1' or 'v2'")
    PROMPT_OVERRIDES[model_name] = variant

def set_prompt_overrides(mapping: dict[str, str]) -> None:
    """Replace the entire override map in one go."""
    PROMPT_OVERRIDES.clear()
    PROMPT_OVERRIDES.update(mapping)

# ─────────────────────── helper functions ─────────────────────
def _openai_call(text: str, model: str, api: OpenAI, system_prompt: str) -> str:
    """
    One-shot call with:
      • Rate limiting
      • 3 automatic retries with exponential back-off (2 s → 4 s → 8 s).
      • System prompt properly set in messages
      • Retry on empty/blank responses
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
            result = resp.choices[0].message.content.strip()
            
            # Check for empty or blank response
            if not result or result.isspace():
                if attempt < 3:
                    print(f"⚠️ Empty response from OpenAI (attempt {attempt}/3), retrying...")
                    time.sleep(2 ** attempt)
                    continue
                else:
                    # If all retries failed, return original text as fallback
                    print(f"❌ All retries failed for OpenAI, returning original text")
                    return text
            
            return result

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
      • Retry on empty/blank responses
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
                result = response.content[0].text.strip()
            else:
                result = str(response.content).strip()
            
            # Check for empty or blank response
            if not result or result.isspace():
                if attempt < 3:
                    print(f"⚠️ Empty response from Claude (attempt {attempt}/3), retrying...")
                    time.sleep(2 ** attempt)
                    continue
                else:
                    # If all retries failed, return original text as fallback
                    print(f"❌ All retries failed for Claude, returning original text")
                    return text
            
            return result
                
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
    """Call Gemini with proper system instructions and retry on empty responses."""
    for attempt in range(1, 4):
        resp = _gemini_generate(model, text, system_prompt)
        result = resp.text.strip()
        
        # Check for empty or blank response
        if not result or result.isspace():
            if attempt < 3:
                print(f"⚠️ Empty response from Gemini (attempt {attempt}/3), retrying...")
                time.sleep(2 ** attempt)
                continue
            else:
                # If all retries failed, return original text as fallback
                print(f"❌ All retries failed for Gemini, returning original text")
                return text
        
        return result


def _select_prompt(
    prompt_id: str,
    mode: Literal["doc", "para"],
    *,
    variant: str | None = None,
    system_prompt: str | None = None,
) -> str:
    """
    Resolve the system-prompt to use.

    • Non-fine-tuned models → always default prompts
    • Fine-tuned models → allow user-selected "v1" or "v2"
                         (fallback to legacy prompt if none supplied)
    • Dynamic models → use system_prompt parameter to select appropriate prompt
    """
    if prompt_id == "legacy-finetuned":
        return (
            LEGACY_FINETUNED_PARA_SYSTEM_PROMPT
            if mode == "doc"
            else LEGACY_FINETUNED_DOC_SYSTEM_PROMPT
        )
    if prompt_id == "default":
        return (
            DEFAULT_DOC_SYSTEM_PROMPT
            if mode == "doc"
            else DEFAULT_PARA_SYSTEM_PROMPT
        )

    if prompt_id == "finetuned":
        if variant == "v1":
            return (
                FINETUNED_DOC_SYSTEM_PROMPT1
                if mode == "doc"
                else FINETUNED_PARA_SYSTEM_PROMPT1
            )
        if variant == "v2":
            return (
                FINETUNED_DOC_SYSTEM_PROMPT2
                if mode == "doc"
                else FINETUNED_PARA_SYSTEM_PROMPT2
            )
        # Fallback – keeps old behaviour for unattended runs
        return (
            FINETUNED_DOC_SYSTEM_PROMPT1
            if mode == "doc"
            else FINETUNED_PARA_SYSTEM_PROMPT1
        )

    if prompt_id == "dynamic":
        # Select prompt based on system_prompt configuration
        if system_prompt == "none":
            # No system prompt
            return ""
        elif system_prompt == "minimal_prompt":
            return MINIMAL_DOC_SYSTEM_PROMPT if mode == "doc" else MINIMAL_PARA_SYSTEM_PROMPT
        elif system_prompt == "compact_prompt":
            return COMPACT_DOC_SYSTEM_PROMPT if mode == "doc" else COMPACT_PARA_SYSTEM_PROMPT
        elif system_prompt == "rich_prompt_standard":
            return RICH_SYSTEM_PROMPT_STANDARD_DOC if mode == "doc" else RICH_SYSTEM_PROMPT_STANDARD_PARA
        elif system_prompt == "rich_prompt_with_counter_examples":
            return RICH_SYSTEM_PROMPT_WITH_COUNTER_EXAMPLES_DOC if mode == "doc" else RICH_SYSTEM_PROMPT_WITH_COUNTER_EXAMPLES_PARA
        elif system_prompt == "rich_prompt_with_negative_examples":
            return RICH_SYSTEM_PROMPT_WITH_NEGATIVE_EXAMPLES_DOC if mode == "doc" else RICH_SYSTEM_PROMPT_WITH_NEGATIVE_EXAMPLES_PARA
        elif system_prompt == "rich_prompt_with_focus_areas":
            return RICH_SYSTEM_PROMPT_WITH_FOCUS_AREAS_DOC if mode == "doc" else RICH_SYSTEM_PROMPT_WITH_FOCUS_AREAS_PARA
        else:
            raise ValueError(f"Unknown system_prompt '{system_prompt}' for dynamic model")

    raise ValueError(f"Unknown prompt_id '{prompt_id}'")



# ───────────────────────── public API ──────────────────────────
def _build_user_prompt(
    text: str,
    scores_in_prompt: str | None = None,
    initial_score: float | None = None,
) -> str:
    """
    Build user prompt based on scores_in_prompt configuration.

    • none: simple rewrite request
    • both_raw: include raw AI score with target goal
    • both_binned: include binned level (Low/Medium/High) with target goal
    • ai_score_raw: include only raw AI score, no target
    • ai_score_binned: include only binned level, no target
    """
    if not scores_in_prompt or scores_in_prompt == "none":
        # Simple prompt without scores
        return f"Please rewrite the following text to sound more human and natural while keeping all facts, meaning, and citations exactly the same:\n\n{text}"

    # Default score if not provided (for testing)
    if initial_score is None:
        initial_score = 0.5

    # Determine level for binned prompts
    if initial_score < 0.1:
        level = "Low"
    elif initial_score < 0.4:
        level = "Medium"
    else:
        level = "High"

    if scores_in_prompt == "both_raw":
        return f"""This text's AI level is currently classified as **{initial_score:.2f}**.
'''
{text}
'''
Rewrite the text more natural to lower AI score to 0.01."""

    elif scores_in_prompt == "both_binned":
        return f"""This text AI score is {level}.
'''
{text}
'''
Rewrite it so the level becomes **Low**."""

    elif scores_in_prompt == "ai_score_raw":
        return f"""This text's AI level is currently classified as **{initial_score:.2f}**.
'''
{text}
'''
Rewrite it more natural so the text sounds human-like."""

    elif scores_in_prompt == "ai_score_binned":
        return f"""This text AI score is {level}.
'''
{text}
'''
Rewrite it more natural so the text sounds human-like."""

    else:
        raise ValueError(f"Unknown scores_in_prompt value: {scores_in_prompt}")


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

    # Extract initial_score if provided (for dynamic prompts)
    initial_score = kwargs.pop("initial_score", None)

    meta = MODEL_REGISTRY[display_name]
    provider = meta["provider"]
    model_id = meta["model"]
    prompt_id = meta["prompt_id"]

    variant = PROMPT_OVERRIDES.get(display_name)  # may be None

    # Get system_prompt config for dynamic models
    system_prompt_type = meta.get("system_prompt")
    scores_in_prompt = meta.get("scores_in_prompt")

    # Select system prompt
    system_prompt = _select_prompt(prompt_id, mode, variant=variant, system_prompt=system_prompt_type)

    # Build user prompt (dynamic models may include scores)
    if prompt_id == "dynamic":
        user_prompt = _build_user_prompt(text, scores_in_prompt, initial_score)
    else:
        user_prompt = f"Please rewrite the following text to sound more human and natural while keeping all facts, meaning, and citations exactly the same:\n\n{text}"

    if provider == "openai":
        if not _openai_std:
            raise ValueError("OpenAI API key not configured")
        return _openai_call(user_prompt, model_id, _openai_std, system_prompt)

    if provider == "openai_ft":
        if not _openai_ft:
            raise ValueError("Humanizer OpenAI API key not configured")
        return _openai_call(user_prompt, model_id, _openai_ft, system_prompt)

    if provider == "openai_dynamic":
        # Use base model with dynamic prompts
        if not _openai_std:
            raise ValueError("OpenAI API key not configured")
        return _openai_call(user_prompt, model_id, _openai_std, system_prompt)

    if provider == "claude":
        if not _claude_client:
            raise ValueError("Claude API key not configured")
        return _claude_call(user_prompt, model_id, system_prompt)

    if provider == "gemini":
        if not _gemini_client:
            raise ValueError("Gemini API key not configured")
        return _gemini_call(user_prompt, model_id, system_prompt)

    raise ValueError(f"Unknown provider '{provider}' for model {display_name}")