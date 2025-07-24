# src/evaluation/quality.py
"""
Paragraph-quality checker with Gemini semantic validation.

v4.0 - Enhanced with grammar dimension
─────
• Added grammar quality scoring (1-10) and error detection
• Maintains all existing functionality and backward compatibility
• Enhanced citation handling for specific formats
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
from ..prompts import build_evaluation_prompt
from ..rate_limiter import wait as _rate_wait

# ─────────────────────────── Gemini client ────────────────────────────
if not GEMINI_API_KEY:
    print("[quality] WARNING: GEMINI_API_KEY is not set! Quality checks will use fallback mode.")
    # create a dummy client so code paths still run
    client = genai.Client(api_key="dummy-key-for-fallback")
else:
    print(f"[quality] Gemini API key configured (length: {len(GEMINI_API_KEY)})")
    client = genai.Client(api_key=GEMINI_API_KEY)

# ─────────────────────────── Schema definitions ───────────────────────────
# Schema for texts WITH Ref-ID citations
QUALITY_SCHEMA_WITH_CITATIONS = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "same_meaning": types.Schema(
            type=types.Type.OBJECT,
            properties={
                "level":   types.Schema(type=types.Type.INTEGER),   # 0-10 enum
                "details": types.Schema(type=types.Type.STRING),
            },
            required=["level", "details"],
        ),
        "same_lang": types.Schema(
            type=types.Type.OBJECT,
            properties={
                "consistent":        types.Schema(type=types.Type.BOOLEAN),
                "originalLanguage":  types.Schema(type=types.Type.STRING),
                "humanisedLanguage": types.Schema(type=types.Type.STRING),
            },
            required=["consistent", "originalLanguage", "humanisedLanguage"],
        ),
        "missing_information": types.Schema(
            type=types.Type.OBJECT,
            properties={
                "level": types.Schema(type=types.Type.INTEGER),      # 0-10 enum
                "missingInfo": types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(type=types.Type.STRING),
                ),
                "addedInfo": types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(type=types.Type.STRING),
                ),
            },
            required=["level", "missingInfo", "addedInfo"],
        ),
        "citation_preserved": types.Schema(
            type=types.Type.OBJECT,
            properties={
                "preserved":        types.Schema(type=types.Type.BOOLEAN),
                "originalCount":    types.Schema(type=types.Type.INTEGER),
                "humanisedCount":   types.Schema(type=types.Type.INTEGER),
                "missingCitations": types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(type=types.Type.STRING),
                ),
            },
            required=["preserved", "originalCount", "humanisedCount", "missingCitations"],
        ),
        "grammar": types.Schema(
            type=types.Type.OBJECT,
            properties={
                "level": types.Schema(type=types.Type.INTEGER),      # 0-10 enum
                "errors": types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(type=types.Type.STRING),
                ),
            },
            required=["level", "errors"],
        ),
    },
    required=[
        "same_meaning",
        "same_lang",
        "missing_information",
        "citation_preserved",
        "grammar",
    ],
)

# Schema for texts WITHOUT Ref-ID citations (no citation_preserved field)
QUALITY_SCHEMA_WITHOUT_CITATIONS = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "same_meaning": types.Schema(
            type=types.Type.OBJECT,
            properties={
                "level":   types.Schema(type=types.Type.INTEGER),   # 0-10 enum
                "details": types.Schema(type=types.Type.STRING),
            },
            required=["level", "details"],
        ),
        "same_lang": types.Schema(
            type=types.Type.OBJECT,
            properties={
                "consistent":        types.Schema(type=types.Type.BOOLEAN),
                "originalLanguage":  types.Schema(type=types.Type.STRING),
                "humanisedLanguage": types.Schema(type=types.Type.STRING),
            },
            required=["consistent", "originalLanguage", "humanisedLanguage"],
        ),
        "missing_information": types.Schema(
            type=types.Type.OBJECT,
            properties={
                "level": types.Schema(type=types.Type.INTEGER),      # 0-10 enum
                "missingInfo": types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(type=types.Type.STRING),
                ),
                "addedInfo": types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(type=types.Type.STRING),
                ),
            },
            required=["level", "missingInfo", "addedInfo"],
        ),
        "grammar": types.Schema(
            type=types.Type.OBJECT,
            properties={
                "level": types.Schema(type=types.Type.INTEGER),      # 0-10 enum
                "errors": types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(type=types.Type.STRING),
                ),
            },
            required=["level", "errors"],
        ),
    },
    required=[
        "same_meaning",
        "same_lang",
        "missing_information",
        "grammar",
    ],
)


def _gemini_generate(model_id: str, system_prompt: str, text_pair: str, schema: types.Schema, *, max_retries: int = 10):
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
                    response_schema=schema,  # Use provided schema
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

# Enhanced citation regex to capture all content in parentheses up to 100 chars
# Note: _citations() function filters this to only return Ref-ID citations
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
    """Return the raw citation strings (without parentheses), filtered to only include Ref-ID citations."""
    # First extract all content in parentheses
    broad_matches = _CITATION_RE.findall(text)
    # Then filter to only keep Ref-ID citations (Ref-f######, Ref-u######, Ref-s######)
    ref_citations = [match for match in broad_matches if _REF_PATTERN.match(match)]
    return ref_citations


def _is_valid_citation(cite: str) -> bool:
    """Check if a citation matches expected formats (for debugging)."""
    return bool(
        _APA_HARVARD_PATTERN.search(cite) or
        _MLA_PATTERN.search(cite) or
        _REF_PATTERN.search(cite)
    )


def _parse_gemini_response(resp, has_citations: bool = True) -> Dict:
    """
    Parse Gemini output and coerce it to the **new** quality layout.

    The evaluator now returns numeric *level* fields (0-10) instead of
    boolean flags for *same_meaning* and *missing_information*.
    We normalise everything here so downstream code can consume a
    predictable structure (booleans derived for backward-compatibility,
    numeric levels exposed for the new dashboards).
    
    Args:
        resp: Gemini response object
        has_citations: Whether the original text had Ref-ID citations
    """
    try:
        # Raw JSON (streaming) vs structured response
        content = resp.text if hasattr(resp, "text") else resp.candidates[0].content.parts[0].text
        print(f"[quality._parse_gemini_response] Raw content: {content}")

        raw = json.loads(content) if isinstance(content, str) else content
        print(f"[quality._parse_gemini_response] Parsed data: {raw}")

        # Helper with sensible defaults
        def _g(key, default=None):
            return raw.get(key, default if default is not None else {})

        same_meaning = _g("same_meaning", {})
        same_lang    = _g("same_lang",    {})
        missing_inf  = _g("missing_information", {})
        grammar      = _g("grammar", {})

        data = {
            "same_meaning": {
                "level":   int(same_meaning.get("level", 0)),
                "details": same_meaning.get("details", ""),
            },
            "same_lang": {
                "consistent":        bool(same_lang.get("consistent", False)),
                "originalLanguage":  same_lang.get("originalLanguage", "unknown"),
                "humanisedLanguage": same_lang.get("humanisedLanguage", "unknown"),
            },
            "missing_information": {
                "level":       int(missing_inf.get("level", 10)),
                "missingInfo": missing_inf.get("missingInfo", []),
                "addedInfo":   missing_inf.get("addedInfo", []),
            },
            "grammar": {
                "level":  int(grammar.get("level", 5)),
                "errors": grammar.get("errors", []),
            },
        }

        # Only include citation data if original had Ref-ID citations
        if has_citations:
            citation = _g("citation_preserved", {})
            data["citation_preserved"] = {
                "preserved":        bool(citation.get("preserved", False)),
                "originalCount":    int(citation.get("originalCount", 0)),
                "humanisedCount":   int(citation.get("humanisedCount", 0)),
                "missingCitations": citation.get("missingCitations", []),
            }
        else:
            # Return None for all citation fields when no citations exist
            data["citation_preserved"] = None

        return data

    except Exception as exc:
        print(f"[quality._parse_gemini_response] Error parsing response: {exc!r}")
        # Safe fall-back
        fallback_data = {
            "same_meaning": {"level": 5, "details": "Parse error"},
            "same_lang": {"consistent": False, "originalLanguage": "unknown", "humanisedLanguage": "unknown"},
            "missing_information": {"level": 5, "missingInfo": [], "addedInfo": []},
            "grammar": {"level": 5, "errors": ["Parse error – could not evaluate"]},
        }
        
        # Add citation fallback only if original had citations
        if has_citations:
            fallback_data["citation_preserved"] = {
                "preserved": False, "originalCount": 0, "humanisedCount": 0, "missingCitations": []
            }
        else:
            fallback_data["citation_preserved"] = None
            
        return fallback_data


# ───────────────────────────── public API ────────────────────────────
def quality(original: str, humanized: str) -> Dict[str, bool | int | List[str]]:
    """
    Evaluate *humanized* against *original* and return a mixed
    dict with both backward-compatible booleans **and** the new
    numeric levels required by v5 of the dashboard.

    • same_meaning_level ...... 0-10 scale
    • missing_info_level ...... 0-10 scale
    • grammar_level ........... 0-10 scale

    Boolean flags (*same_meaning*, *no_missing_info*) are derived
    heuristically from the numeric levels (≥7 and ≤2, respectively)
    so the rest of the pipeline keeps working until it is refactored.
    
    Citation metrics are only evaluated if the original text contains Ref-ID citations.
    """
    print("\n" + "=" * 60)
    print("[quality] Starting quality check")
    print(f"[quality] Original length: {len(original)} chars, {len(original.split())} words")
    print(f"[quality] Humanized length: {len(humanized)} chars, {len(humanized.split())} words")
    print(f"[quality] Original preview: {original[:100]}…")
    print(f"[quality] Humanized preview: {humanized[:100]}…")

    # ── 1 · deterministic checks ──────────────────────────────────────────
    word_delta = len(humanized.split()) - len(original.split())
    length_ok  = -15 <= word_delta <= 15

    orig_citations = _citations(original)
    hum_citations  = _citations(humanized)
    has_ref_citations = len(orig_citations) > 0
    
    print(f"[quality] Ref-ID citations found: {len(orig_citations)} in original, {len(hum_citations)} in humanized")
    if has_ref_citations:
        print(f"[quality] Original citations: {orig_citations}")
        print(f"[quality] Humanized citations: {hum_citations}")
    
    # Citation content check logic:
    # - If no original citations: None (not applicable)  
    # - If original has citations but humanized has none: False
    # - If original has citations and humanized has some: check if ALL original citations preserved
    if orig_citations:
        if not hum_citations:
            # Original had citations but humanized has none -> False
            citation_content_ok = False
            print(f"[quality] Citation content FAILED: Original had {len(orig_citations)} citations, humanized has none")
        else:
            # Check if ALL original citations are preserved in humanized text
            citation_content_ok = all(f"({c})" in humanized for c in orig_citations)
            preserved_count = sum(1 for c in orig_citations if f"({c})" in humanized)
            print(f"[quality] Citation content check: {preserved_count}/{len(orig_citations)} preserved -> {citation_content_ok}")
    else:
        citation_content_ok = None  # Not applicable when no original citations

    # ── 2 · Gemini semantic check ─────────────────────────────────────────
    text_pair = f"ORIGINAL:\n{original}\n\nHUMANISED:\n{humanized}"
    
    # Use appropriate prompt and schema based on citation presence
    evaluation_prompt = build_evaluation_prompt(has_ref_citations)
    schema = QUALITY_SCHEMA_WITH_CITATIONS if has_ref_citations else QUALITY_SCHEMA_WITHOUT_CITATIONS
    
    try:
        if not GEMINI_API_KEY:
            raise ValueError("No GEMINI_API_KEY configured")
        resp      = _gemini_generate("gemini-2.0-flash", evaluation_prompt, text_pair, schema)
        gem_data  = _parse_gemini_response(resp, has_citations=has_ref_citations)
    except Exception as exc:
        print(f"[quality] Gemini evaluation failed: {exc}")
        gem_data = {
            "same_meaning": {"level": 5, "details": "Fallback"},
            "same_lang": {"consistent": True, "originalLanguage": "unknown", "humanisedLanguage": "unknown"},
            "missing_information": {"level": 2, "missingInfo": [], "addedInfo": []},
            "grammar": {"level": 10, "errors": []},
        }
        
        # Add citation fallback only if original had citations
        if has_ref_citations:
            gem_data["citation_preserved"] = {
                "preserved": citation_content_ok,
                "originalCount": len(orig_citations),
                "humanisedCount": len(hum_citations),
                "missingCitations": []
            }
        else:
            gem_data["citation_preserved"] = None

    # ── 3 · derive booleans for legacy consumers ─────────────────────────
    same_meaning_bool   = gem_data["same_meaning"]["level"] >= 7
    no_missing_info_bool = gem_data["missing_information"]["level"] <= 2
    
    # Citation preservation logic: only for paragraphs with citations
    if has_ref_citations and gem_data["citation_preserved"] is not None:
        citation_preserved_bool = (
            gem_data["citation_preserved"]["preserved"] and 
            not gem_data["citation_preserved"]["missingCitations"]
        )
    else:
        citation_preserved_bool = None  # Not applicable

    # ── 4 · final payload ────────────────────────────────────────────────
    result = {
        # deterministic
        "length_ok":            length_ok,
        "citation_content_ok":  citation_content_ok,  # None if no citations

        # legacy booleans
        "same_meaning":         same_meaning_bool,
        "no_missing_info":      no_missing_info_bool,
        "same_lang":            gem_data["same_lang"]["consistent"],
        "citation_preserved":   citation_preserved_bool,  # None if no citations

        # new numeric metrics
        "same_meaning_level":   gem_data["same_meaning"]["level"],
        "missing_info_level":   gem_data["missing_information"]["level"],
        "grammar_level":        gem_data["grammar"]["level"],

        # NEW ▸ expose full meaning object & details
        "same_meaning_details": gem_data["same_meaning"]["details"],
        "same_meaning_obj":     gem_data["same_meaning"],   # nested schema preserved

        # extras for detailed UIs
        "grammar_errors":       gem_data["grammar"]["errors"],
        "missing_items":        gem_data["missing_information"]["missingInfo"],
        "added_items":          gem_data["missing_information"]["addedInfo"],
        "original_lang":        gem_data["same_lang"]["originalLanguage"],
        "humanised_lang":       gem_data["same_lang"]["humanisedLanguage"],
    }

    # Add citation-specific fields only if citations exist
    if has_ref_citations and gem_data["citation_preserved"] is not None:
        result["missing_citations"] = gem_data["citation_preserved"]["missingCitations"]
    else:
        result["missing_citations"] = None  # Not applicable

    # diagnostic summary - exclude None values from boolean count
    bool_flags = {k: v for k, v in result.items()
                  if isinstance(v, bool) and k not in ("citation_content_ok",)}
    passed = sum(bool_flags.values())
    total_applicable = len(bool_flags)
    
    print(f"[quality] Boolean checks passed: {passed}/{total_applicable}")
    print(f"[quality] same_meaning_level={result['same_meaning_level']} "
          f"missing_info_level={result['missing_info_level']} "
          f"grammar_level={result['grammar_level']}")
    if has_ref_citations:
        print(f"[quality] Citation evaluation enabled (found {len(orig_citations)} Ref-ID citations)")
    else:
        print("[quality] Citation evaluation skipped (no Ref-ID citations found)")
    print("=" * 60 + "\n")
    return result
