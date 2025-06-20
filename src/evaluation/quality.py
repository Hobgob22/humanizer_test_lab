"""
Paragraph-quality checker with Gemini semantic validation.

v3.3 - Enhanced citation handling for specific formats
────
• Improved citation detection for APA/Harvard, MLA, and Ref-style citations
• Updated validation logic for citation preservation
• Maintains all existing functionality
• Improved error handling and debugging
"""

from __future__ import annotations

import json
import random
import re
import time
from typing import Dict, List

from google import genai
from google.genai import types

from ..config import GEMINI_API_KEY
from ..prompts import EVALUATION_PROMPT
from ..rate_limiter import wait as _rate_wait

# ─────────────────────────── Gemini client ────────────────────────────
if not GEMINI_API_KEY:
    print("[quality] WARNING: GEMINI_API_KEY is not set! Quality checks will use fallback mode.")
    # create a dummy client so code paths still run
    client = genai.Client(api_key="dummy-key-for-fallback")
else:
    print(f"[quality] Gemini API key configured (length: {len(GEMINI_API_KEY)})")
    client = genai.Client(api_key=GEMINI_API_KEY)

# ─────────────────────────── Schema definition ────────────────────────────
QUALITY_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "same_meaning": types.Schema(type=types.Type.BOOLEAN),
        "same_lang": types.Schema(type=types.Type.BOOLEAN),
        "no_missing_info": types.Schema(type=types.Type.BOOLEAN),
        "citation_preserved": types.Schema(type=types.Type.BOOLEAN),
    },
    required=["same_meaning", "same_lang", "no_missing_info", "citation_preserved"],
)


def _gemini_generate(model_id: str, system_prompt: str, text_pair: str, *, max_retries: int = 10):
    """
    Helper to invoke Gemini with structured output, rate-limiting, and retry logic.
    Rate limit: 700 requests/minute (shared with humanizer)
    """
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not configured")

    print(f"[quality._gemini_generate] Starting with model_id={model_id}")
    delay = 5  # Start with smaller delay since we have higher rate limit

    for attempt in range(1, max_retries + 1):
        print(f"[quality._gemini_generate] Attempt {attempt}/{max_retries}")
        _rate_wait("gemini")

        try:
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=text_pair)],
                )
            ]
            resp = client.models.generate_content(
                model=model_id,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema=QUALITY_SCHEMA,
                    system_instruction=system_prompt,  # System prompt properly set
                ),
            )
            print(f"[quality._gemini_generate] Success on attempt {attempt}")
            return resp

        except Exception as exc:
            msg = str(exc).lower()
            print(f"[quality._gemini_generate] Exception on attempt {attempt}: {exc!r}")

            # Abort immediately on quota errors
            if "quota" in msg:
                print("[quality._gemini_generate] Quota exhausted – aborting")
                raise

            # Retry on rate-limit errors
            if ("rate" in msg or "429" in msg) and attempt < max_retries:
                backoff = delay + random.uniform(0, 2)
                print(f"[quality._gemini_generate] Rate limit hit – waiting {backoff:.1f}s")
                time.sleep(backoff)
                delay = min(delay * 1.5, 60)  # Cap at 60s since we have higher rate limit
                continue

            # Non-retriable or out of retries
            raise


# ───────────────────────────── helpers ───────────────────────────────
_ALL_FLAGS = (
    "length_ok",
    "same_meaning",
    "same_lang",
    "no_missing_info",
    "citation_preserved",
    "citation_content_ok",
)

_GEMINI_FLAGS = (
    "same_meaning",
    "same_lang",
    "no_missing_info",
    "citation_preserved",
)

# Enhanced citation regex to capture various citation formats
# Matches content in parentheses up to 100 chars (APA/Harvard, MLA, and Ref-style)
_CITATION_RE = re.compile(r"\(([^()]{1,100}?)\)")

# More specific patterns for validation (not used for extraction, just for logging)
_APA_HARVARD_PATTERN = re.compile(
    r"[A-Z][a-zA-Z'-]+(?:\s+et\s+al\.)?(?:\s*,\s*\d{4})|"  # Smith, 2021 or Smith et al., 2021
    r"[A-Z][a-zA-Z'-]+\s*&\s*[A-Z][a-zA-Z'-]+(?:\s*,\s*\d{4})?"  # Brown & Garcia, 2018
)
_MLA_PATTERN = re.compile(
    r"[A-Z][a-zA-Z'-]+(?:\s+(?:and|et\s+al\.))?\s+\d+(?:–\d+)?"  # Smith 23 or Smith et al. 117
)
_REF_PATTERN = re.compile(r"Ref-[fus]\d{6}")  # Ref-f123456, Ref-u999999, Ref-s000001


def _citations(text: str) -> List[str]:
    """Return the raw citation strings (without parentheses)."""
    return _CITATION_RE.findall(text)


def _is_valid_citation(cite: str) -> bool:
    """Check if a citation matches expected formats (for debugging)."""
    return bool(
        _APA_HARVARD_PATTERN.search(cite) or
        _MLA_PATTERN.search(cite) or
        _REF_PATTERN.search(cite)
    )


def _parse_gemini_response(resp) -> Dict[str, bool]:
    """Parse Gemini's structured response into our expected format."""
    try:
        # support both direct JSON and candidate formats
        if hasattr(resp, "text"):
            content = resp.text
        else:
            content = resp.candidates[0].content.parts[0].text

        print(f"[quality._parse_gemini_response] Raw content: {content}")
        data = json.loads(content) if isinstance(content, str) else content
        print(f"[quality._parse_gemini_response] Parsed data: {data}")

        result: Dict[str, bool] = {}
        for flag in _GEMINI_FLAGS:
            result[flag] = bool(data.get(flag, False))
        return result

    except Exception as e:
        print(f"[quality._parse_gemini_response] Error parsing response: {e!r}")
        # On parse failure, mark all false
        return {flag: False for flag in _GEMINI_FLAGS}


# ───────────────────────────── public API ────────────────────────────
def quality(original: str, humanized: str) -> Dict[str, bool]:
    """Return quality-flags for a single paragraph/block."""
    print("\n" + "=" * 60)
    print("[quality] Starting quality check")
    print(f"[quality] Original length: {len(original)} chars, {len(original.split())} words")
    print(f"[quality] Humanized length: {len(humanized)} chars, {len(humanized.split())} words")
    print(f"[quality] Original preview: {original[:100]}...")
    print(f"[quality] Humanized preview: {humanized[:100]}...")

    # 1. Deterministic checks
    len_orig = len(original.split())
    len_hum = len(humanized.split())
    word_delta = len_hum - len_orig
    length_ok = -30 <= word_delta <= 10

    orig_citations = _citations(original)
    hum_citations = _citations(humanized)
    
    # Check if all original citations are preserved in humanized text
    # We check the full citation including parentheses
    citation_content_ok = all(f"({c})" in humanized for c in orig_citations)

    print(f"[quality] Length check: delta={word_delta}, ok={length_ok}")
    print(
        f"[quality] Citations: original={len(orig_citations)}, "
        f"humanized={len(hum_citations)}, preserved={citation_content_ok}"
    )
    if orig_citations:
        print(f"[quality] Original citations: {orig_citations[:3]}...")
        # Log citation types for debugging
        for cite in orig_citations[:5]:  # Check first 5
            if _is_valid_citation(cite):
                print(f"[quality]   - Valid citation format: ({cite})")
            else:
                print(f"[quality]   - Unknown citation format: ({cite})")

    # 2. Gemini semantic checks
    print("[quality] Preparing Gemini request")
    text_pair = f"ORIGINAL:\n{original}\n\nHUMANIZED:\n{humanized}"

    try:
        if not GEMINI_API_KEY:
            raise ValueError("No GEMINI_API_KEY configured")

        resp = _gemini_generate("gemini-2.0-flash", EVALUATION_PROMPT, text_pair)
        gem_flags = _parse_gemini_response(resp)
        print(f"[quality] Gemini evaluation: {gem_flags}")

    except Exception as e:
        print(f"[quality] Gemini evaluation failed: {e}")
        gem_flags = {
            "same_meaning": abs(word_delta) < 50,
            "same_lang": True,
            "no_missing_info": len_hum >= len_orig * 0.7,
            "citation_preserved": citation_content_ok,
        }
        print(f"[quality] Using fallback evaluation: {gem_flags}")
        if not GEMINI_API_KEY:
            print("[quality] Note: GEMINI_API_KEY not set - using heuristic evaluation")

    # 3. Combine all checks
    result = {
        "length_ok": length_ok,
        "citation_content_ok": citation_content_ok,
        **gem_flags,
    }

    passed = sum(1 for v in result.values() if v)
    print(f"[quality] Final result: {passed}/{len(result)} checks passed")
    for flag, val in result.items():
        print(f"[quality]   - {flag}: {'✓' if val else '✗'}")
    print("=" * 60 + "\n")

    return result