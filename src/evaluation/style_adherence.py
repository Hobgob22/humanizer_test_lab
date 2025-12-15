# src/evaluation/style_adherence.py
"""
Style adherence checker with Gemini evaluation.

Evaluates how well humanized text follows a given writing profile.
"""

from __future__ import annotations

import json
import random
import time
from typing import Dict, Any

from google import genai
from google.genai import types

from ..config import GEMINI_API_KEY
from ..rate_limiter import wait as _rate_wait

# ─────────────────────────── Gemini client ────────────────────────────
if not GEMINI_API_KEY:
    print("[style_adherence] WARNING: GEMINI_API_KEY is not set! Style checks will use fallback mode.")
    client = genai.Client(api_key="dummy-key-for-fallback")
else:
    print(f"[style_adherence] Gemini API key configured (length: {len(GEMINI_API_KEY)})")
    client = genai.Client(api_key=GEMINI_API_KEY)

# ─────────────────────────── Schema definition ───────────────────────────
STYLE_ADHERENCE_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "overall_adherence": types.Schema(
            type=types.Type.OBJECT,
            properties={
                "score": types.Schema(type=types.Type.INTEGER),  # 0-10 scale
                "summary": types.Schema(type=types.Type.STRING),
            },
            required=["score", "summary"],
        ),
        "hedging": types.Schema(
            type=types.Type.OBJECT,
            properties={
                "score": types.Schema(type=types.Type.INTEGER),  # 0-10 scale
                "details": types.Schema(type=types.Type.STRING),
            },
            required=["score", "details"],
        ),
        "formality": types.Schema(
            type=types.Type.OBJECT,
            properties={
                "score": types.Schema(type=types.Type.INTEGER),  # 0-10 scale
                "details": types.Schema(type=types.Type.STRING),
            },
            required=["score", "details"],
        ),
        "vocabulary": types.Schema(
            type=types.Type.OBJECT,
            properties={
                "score": types.Schema(type=types.Type.INTEGER),  # 0-10 scale
                "details": types.Schema(type=types.Type.STRING),
            },
            required=["score", "details"],
        ),
        "sentence_structure": types.Schema(
            type=types.Type.OBJECT,
            properties={
                "score": types.Schema(type=types.Type.INTEGER),  # 0-10 scale
                "details": types.Schema(type=types.Type.STRING),
            },
            required=["score", "details"],
        ),
        "strengths": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(type=types.Type.STRING),
        ),
        "weaknesses": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(type=types.Type.STRING),
        ),
    },
    required=[
        "overall_adherence",
        "hedging",
        "formality",
        "vocabulary",
        "sentence_structure",
        "strengths",
        "weaknesses",
    ],
)


STYLE_ADHERENCE_SYSTEM_PROMPT = """
You are an expert linguistic evaluator specializing in academic writing style analysis.

Your task is to evaluate how well a HUMANIZED TEXT adheres to a given WRITING PROFILE.

# CONTEXT

You will receive:
1. **WRITING PROFILE**: The target style specifications
2. **ORIGINAL TEXT**: The source text that was humanized (for context only)
3. **HUMANIZED TEXT**: The rewritten version to evaluate

**Important**: The ORIGINAL TEXT is provided for context to help you understand what content and constraints the humanization was working with. If certain profile elements (like specific terminology, technical density, or data) are not present in the humanized text because they were not in the original, this is acceptable and should NOT be penalized.

Focus your evaluation on: **How well does the humanized text adopt the stylistic characteristics from the profile, given what was in the original text?**

# EVALUATION CRITERIA

For each dimension (hedging, formality, vocabulary, sentence_structure), provide:
- **score** (0-10 scale):
  - 0-2: Poor adherence, multiple major violations of the style profile
  - 3-4: Weak adherence, several notable violations
  - 5-6: Moderate adherence, some inconsistencies
  - 7-8: Good adherence, minor deviations, stylistic goals mostly achieved
  - 9-10: Excellent adherence, profile characteristics successfully adopted

- **details**: Specific evidence from the text explaining the score (cite examples)

## Hedging
Evaluate modal verbs, epistemic verbs, probability markers, and hedging patterns against the profile. Consider whether the original text provided opportunities for hedging.

## Formality
Evaluate overall formality level, nominalization, personal pronouns, passive voice, and contractions.

## Vocabulary
Evaluate lexical choices, word variety, and register consistency. Note: If the original text lacked technical content, the absence of technical vocabulary in the humanized version is acceptable.

## Sentence Structure
Evaluate sentence length, complexity, clause preferences, and syntactic patterns that could be applied to the given content.

# OUTPUT REQUIREMENTS

Provide:
1. **overall_adherence**: Aggregate score (0-10) and summary of how well the humanized text adopts the target style
2. **hedging**: Score and details for hedging dimension
3. **formality**: Score and details for formality dimension
4. **vocabulary**: Score and details for vocabulary dimension
5. **sentence_structure**: Score and details for sentence structure dimension
6. **strengths**: List of 2-5 aspects where the humanized text successfully matches the profile
7. **weaknesses**: List of 2-5 aspects where the humanized text could better match the profile (only cite deviations that were feasible given the original content)

Be specific and evidence-based. Quote examples from the humanized text to support your evaluation. Be fair - don't penalize for missing profile elements that weren't achievable given the original text's content.
""".strip()


def _gemini_generate(model_id: str, system_prompt: str, user_content: str, schema: types.Schema, *, max_retries: int = 10):
    """
    Helper to invoke Gemini with structured output, rate-limiting, and retry logic.
    """
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not configured")

    print(f"[style_adherence._gemini_generate] Starting with model_id={model_id}")
    delay = 5

    for attempt in range(1, max_retries + 1):
        print(f"[style_adherence._gemini_generate] Attempt {attempt}/{max_retries}")
        _rate_wait("gemini")

        try:
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=user_content)],
                )
            ]
            resp = client.models.generate_content(
                model=model_id,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema=schema,
                    system_instruction=system_prompt,
                ),
            )
            print(f"[style_adherence._gemini_generate] Success on attempt {attempt}")
            return resp

        except Exception as exc:
            msg = str(exc).lower()
            print(f"[style_adherence._gemini_generate] Exception on attempt {attempt}: {exc!r}")

            # Abort immediately on quota errors
            if "quota" in msg:
                print("[style_adherence._gemini_generate] Quota exhausted – aborting")
                raise

            # Retry on rate-limit errors
            if ("rate" in msg or "429" in msg) and attempt < max_retries:
                backoff = delay + random.uniform(0, 2)
                print(f"[style_adherence._gemini_generate] Rate limit hit – waiting {backoff:.1f}s")
                time.sleep(backoff)
                delay = min(delay * 1.5, 60)
                continue

            # Non-retriable or out of retries
            raise


def _parse_gemini_response(resp) -> Dict[str, Any]:
    """Parse Gemini structured output response."""
    try:
        content = resp.text if hasattr(resp, "text") else resp.candidates[0].content.parts[0].text
        print(f"[style_adherence._parse_gemini_response] Raw content: {content[:200]}...")

        raw = json.loads(content) if isinstance(content, str) else content
        print(f"[style_adherence._parse_gemini_response] Parsed data successfully")

        def _g(key, default=None):
            return raw.get(key, default if default is not None else {})

        overall = _g("overall_adherence", {})
        hedging = _g("hedging", {})
        formality = _g("formality", {})
        vocabulary = _g("vocabulary", {})
        sentence_structure = _g("sentence_structure", {})

        data = {
            "overall_adherence": {
                "score": int(overall.get("score", 5)),
                "summary": overall.get("summary", ""),
            },
            "hedging": {
                "score": int(hedging.get("score", 5)),
                "details": hedging.get("details", ""),
            },
            "formality": {
                "score": int(formality.get("score", 5)),
                "details": formality.get("details", ""),
            },
            "vocabulary": {
                "score": int(vocabulary.get("score", 5)),
                "details": vocabulary.get("details", ""),
            },
            "sentence_structure": {
                "score": int(sentence_structure.get("score", 5)),
                "details": sentence_structure.get("details", ""),
            },
            "strengths": _g("strengths", []),
            "weaknesses": _g("weaknesses", []),
        }

        return data

    except Exception as exc:
        print(f"[style_adherence._parse_gemini_response] Error parsing response: {exc!r}")
        # Safe fallback
        return {
            "overall_adherence": {"score": 5, "summary": "Parse error"},
            "hedging": {"score": 5, "details": "Parse error"},
            "formality": {"score": 5, "details": "Parse error"},
            "vocabulary": {"score": 5, "details": "Parse error"},
            "sentence_structure": {"score": 5, "details": "Parse error"},
            "strengths": [],
            "weaknesses": ["Parse error – could not evaluate"],
        }


def evaluate_style_adherence(writing_profile: Dict[str, Any], original_text: str, humanized_text: str) -> Dict[str, Any]:
    """
    Evaluate how well humanized_text adopts the stylistic patterns from the writing_profile.
    
    Args:
        writing_profile: The AcademicWritingProfile dict (from Writing Profile Lab)
        original_text: The original text before humanization (for context - to understand content constraints)
        humanized_text: The humanized text to evaluate
    
    Returns:
        Dict with scores (0-10) for each dimension, strengths, weaknesses, and overall adherence
    """
    print("\n" + "=" * 60)
    print("[style_adherence] Starting style adherence evaluation")
    print(f"[style_adherence] Original text length: {len(original_text)} chars, {len(original_text.split())} words")
    print(f"[style_adherence] Humanized text length: {len(humanized_text)} chars, {len(humanized_text.split())} words")
    print(f"[style_adherence] Profile keys: {list(writing_profile.keys())}")

    # Build the evaluation prompt
    profile_json = json.dumps(writing_profile, indent=2)
    
    user_content = f"""# WRITING PROFILE

{profile_json}

# ORIGINAL TEXT (for context - to understand what content was being humanized)

{original_text}

# HUMANIZED TEXT TO EVALUATE

{humanized_text}

# TASK

Evaluate how well the HUMANIZED TEXT adopts the stylistic characteristics from the WRITING PROFILE.

**Important Context**: The ORIGINAL TEXT is provided so you understand what content was being worked with. If certain profile elements (like technical terminology, specific data patterns, or certain rhetorical devices) are not present in the humanized text because they were not in the original, this is acceptable and should NOT be penalized. The goal was to humanize the original text while adopting the target style where feasible.

Focus on: Did the humanization successfully apply the stylistic patterns from the profile to the available content?

For each dimension (hedging, formality, vocabulary, sentence_structure), provide:
- A score (0-10) indicating how well the style was adopted
- Specific details with quoted examples from the humanized text

Also provide:
- overall_adherence: aggregate score and summary of style adoption
- strengths: list of 2-5 aspects where the humanized text successfully matches the profile
- weaknesses: list of 2-5 aspects where the style could be better applied (only cite feasible improvements)

Be fair and context-aware. Don't penalize for profile elements that couldn't be applied given the original text's content.
"""

    try:
        if not GEMINI_API_KEY:
            raise ValueError("No GEMINI_API_KEY configured")
        
        resp = _gemini_generate(
            "gemini-2.5-flash",
            STYLE_ADHERENCE_SYSTEM_PROMPT,
            user_content,
            STYLE_ADHERENCE_SCHEMA
        )
        result = _parse_gemini_response(resp)
    except Exception as exc:
        print(f"[style_adherence] Gemini evaluation failed: {exc}")
        result = {
            "overall_adherence": {"score": 5, "summary": "Evaluation failed"},
            "hedging": {"score": 5, "details": "Evaluation failed"},
            "formality": {"score": 5, "details": "Evaluation failed"},
            "vocabulary": {"score": 5, "details": "Evaluation failed"},
            "sentence_structure": {"score": 5, "details": "Evaluation failed"},
            "strengths": [],
            "weaknesses": [f"Evaluation failed: {str(exc)[:100]}"],
        }

    print(f"[style_adherence] Overall adherence score: {result['overall_adherence']['score']}/10")
    print(f"[style_adherence] Dimension scores: hedging={result['hedging']['score']}, "
          f"formality={result['formality']['score']}, vocabulary={result['vocabulary']['score']}, "
          f"sentence_structure={result['sentence_structure']['score']}")
    print(f"[style_adherence] Strengths: {len(result['strengths'])}, Weaknesses: {len(result['weaknesses'])}")
    print("=" * 60 + "\n")
    
    return result
