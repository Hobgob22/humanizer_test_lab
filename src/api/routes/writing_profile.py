from __future__ import annotations

import json
from typing import List, Optional, Dict

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from ..dependencies import verify_api_key
from ..models import (
    WritingProfileResponse,
    HumanizeRequest,
    HumanizeResponse,
    AIScoreRequest,
    AIScoreResponse,
    QualityCheckRequest,
    QualityCheckResponse,
    StyleAdherenceRequest,
    StyleAdherenceResponse,
)
from ...text_extraction import (
    ALLOWED_EXTENSIONS,
    extract_text_from_bytes,
    summarize_text,
    validate_extension,
)
from ...writing_profile import (
    generate_writing_profile,
    convert_profile_to_generation_prompt,
    AcademicWritingProfile,
)
from ...humanizers.humanizer import humanize
from ...detectors import gptzero
from ...evaluation.quality import quality
from ...evaluation.style_adherence import evaluate_style_adherence

MAX_FILES = 5
MAX_SINGLE_FILE_BYTES = 2 * 1024 * 1024  # 2 MB per file
MAX_TOTAL_BYTES = 8 * 1024 * 1024  # 8 MB per request
MAX_SAMPLE_CHAR = 16000

router = APIRouter()


def _prepare_profile_instruction(profile_payload: Dict) -> tuple[AcademicWritingProfile, str]:
    """
    Validate the provided profile payload and return both the parsed object
    and a generation prompt that can be embedded into downstream user prompts.
    """
    try:
        profile_obj = AcademicWritingProfile(**profile_payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid writing profile payload: {exc}") from exc
    
    instruction_text = convert_profile_to_generation_prompt(profile_obj)
    return profile_obj, instruction_text


@router.post("/generate", response_model=WritingProfileResponse)
async def create_writing_profile(
    model_id: str = Form(...),
    sample_text: Optional[str] = Form(None),
    reasoning_effort: Optional[str] = Form(None),
    thinking_mode: Optional[str] = Form(None),
    deep_think: Optional[bool] = Form(False),
    thinking_budget: Optional[str] = Form("5000"),
    files: Optional[List[UploadFile]] = File(None),
    api_key: str = Depends(verify_api_key),
):
    provided_texts: List[str] = []
    source_files: List[dict] = []
    total_bytes = 0

    if sample_text and sample_text.strip():
        provided_texts.append(sample_text.strip())

    upload_list = files or []
    if len(upload_list) > MAX_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files uploaded. Max allowed is {MAX_FILES}.",
        )

    parsed_thinking_budget: Optional[int] = None
    if thinking_budget not in (None, ""):
        try:
            parsed_thinking_budget = max(0, int(thinking_budget))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Thinking budget must be an integer.") from exc
    else:
        parsed_thinking_budget = None

    for upload in upload_list:
        filename = upload.filename or "unnamed"
        if not validate_extension(filename):
            allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
            raise HTTPException(
                status_code=400,
                detail=f"File '{filename}' has unsupported type. Allowed: {allowed}",
            )
        data = await upload.read()
        size = len(data)
        if size > MAX_SINGLE_FILE_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"File '{filename}' exceeds the {MAX_SINGLE_FILE_BYTES // (1024 * 1024)} MB limit.",
            )
        total_bytes += size
        if total_bytes > MAX_TOTAL_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"Combined upload exceeds {MAX_TOTAL_BYTES // (1024 * 1024)} MB.",
            )
        try:
            extracted = extract_text_from_bytes(data, filename)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to read '{filename}': {exc}") from exc
        if extracted.strip():
            provided_texts.append(extracted.strip())
            source_files.append({"name": filename, "size_bytes": size})

    if not provided_texts:
        raise HTTPException(status_code=400, detail="Provide at least one text sample or upload a file.")

    merged_sample = "\n\n---\n\n".join(provided_texts)
    if len(merged_sample) > MAX_SAMPLE_CHAR:
        merged_sample = merged_sample[:MAX_SAMPLE_CHAR]

    reasoning_options = {
        "reasoning_effort": reasoning_effort or "",
        "thinking_mode": thinking_mode or "",
        "deep_think": bool(deep_think),
        "thinking_budget": parsed_thinking_budget,
    }

    try:
        result = generate_writing_profile(model_id, merged_sample, reasoning=reasoning_options)
        preview = summarize_text([merged_sample], max_chars=800)
        response_payload = WritingProfileResponse(
            model=model_id,
            profile=result["profile"],
            raw_output=result["raw_output"],
            markdown_preview=result["markdown_preview"],
            system_prompt=result["system_prompt"],
            user_prompt=result["user_prompt"],
            reasoning={k: v for k, v in reasoning_options.items() if v not in (None, "", False)},
            sample_preview=preview,
            sources=source_files,
            pricing=result.get("pricing"),
            token_usage=result.get("token_usage"),
        )
        return response_payload
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except json.JSONDecodeError as exc:  # type: ignore[name-defined]
        raise HTTPException(status_code=502, detail="Model response was not valid JSON.") from exc
    except Exception as exc:
        import traceback
        error_detail = f"Failed to generate writing profile: {exc}"
        print(f"[ERROR] {error_detail}")
        print(f"[ERROR] Full traceback:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_detail) from exc


@router.post("/humanize", response_model=HumanizeResponse)
async def humanize_with_profile(
    payload: HumanizeRequest,
    api_key: str = Depends(verify_api_key),
):
    """Generate a single humanized draft using a previously extracted writing profile."""
    if not payload.text or not payload.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")
    
    profile_obj, instruction_text = _prepare_profile_instruction(payload.writing_profile)

    task_prompt = (
        "## Task\n"
        "Rewrite ONLY the text inside <original_text>...</original_text> so it follows every style rule above "
        "while preserving all meaning, facts, and (Ref-XXXX) citations.\n"
        "Return only the rewritten text without commentary.\n\n"
        "<original_text>\n"
        f"{payload.text.strip()}\n"
        "</original_text>\n"
    )

    profile_mode = (payload.profile_mode or "user").lower()
    if profile_mode not in {"user", "system"}:
        raise HTTPException(status_code=400, detail="profile_mode must be either 'user' or 'system'")

    user_prefix = instruction_text if profile_mode == "user" else None
    system_prefix = instruction_text if profile_mode == "system" else None
    rendered_user_prompt = f"{instruction_text}\n\n{task_prompt}" if profile_mode == "user" else task_prompt

    try:
        humanized = humanize(
            task_prompt,
            payload.model,
            "doc",
            system_prefix=system_prefix,
            user_prefix=user_prefix,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to humanize text: {exc}") from exc
    
    return HumanizeResponse(
        model=payload.model,
        humanized_text=humanized.strip(),
        instruction_preview=instruction_text,
        prompt_used=rendered_user_prompt,
        profile_summary=profile_obj.profile_summary,
        profile_mode=profile_mode,
    )


@router.post("/ai-score", response_model=AIScoreResponse)
async def check_ai_score(
    payload: AIScoreRequest,
    api_key: str = Depends(verify_api_key),
):
    """Run GPTZero on a single text snippet and return the raw detector output."""
    if not payload.text or not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    
    version = payload.version or "2025-11-28-base"
    try:
        raw = gptzero.detect_ai(payload.text, version=version, skip_cache=payload.skip_cache)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to run GPTZero: {exc}") from exc
    
    documents = raw.get("documents") or [{}]
    doc_payload = documents[0]
    score = doc_payload.get("completely_generated_prob")
    
    return AIScoreResponse(
        version=version,
        completely_generated_prob=score,
        raw_document=doc_payload,
    )


@router.post("/quality", response_model=QualityCheckResponse)
async def run_quality_check(
    payload: QualityCheckRequest,
    api_key: str = Depends(verify_api_key),
):
    """Evaluate a humanized draft against the original text."""
    if not payload.original_text or not payload.original_text.strip():
        raise HTTPException(status_code=400, detail="Original text cannot be empty.")
    if not payload.humanized_text or not payload.humanized_text.strip():
        raise HTTPException(status_code=400, detail="Humanized text cannot be empty.")
    
    try:
        result = quality(payload.original_text, payload.humanized_text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Quality check failed: {exc}") from exc
    
    return QualityCheckResponse(result=result)


@router.post("/style-adherence", response_model=StyleAdherenceResponse)
async def check_style_adherence(
    payload: StyleAdherenceRequest,
    api_key: str = Depends(verify_api_key),
):
    """Evaluate how well humanized text adopts stylistic patterns from a writing profile (original text for context)."""
    if not payload.original_text or not payload.original_text.strip():
        raise HTTPException(status_code=400, detail="Original text is required for context.")
    if not payload.humanized_text or not payload.humanized_text.strip():
        raise HTTPException(status_code=400, detail="Humanized text cannot be empty.")
    if not payload.writing_profile:
        raise HTTPException(status_code=400, detail="Writing profile cannot be empty.")
    
    try:
        result = evaluate_style_adherence(payload.writing_profile, payload.original_text, payload.humanized_text)
    except Exception as exc:
        import traceback
        error_detail = f"Style adherence check failed: {exc}"
        print(f"[ERROR] {error_detail}")
        print(f"[ERROR] Full traceback:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_detail) from exc
    
    return StyleAdherenceResponse(result=result)
