# src/pipeline.py
"""
AI-score extraction & N-run benchmark pipeline, now interrupt-aware
and safe to import in threaded environments (e.g. Streamlit).

• Documents are processed sequentially
• Within each document, ALL iterations are processed concurrently:
  - Document & paragraph humanizations run in parallel
  - Detector checks (GPTZero + Sapling) run concurrently
  - Gemini quality checks run concurrently
• Respects rate limits:
  - Gemini: 700 req/min (shared between humanizer and quality)
  - GPTZero: 500 req/min
  - Sapling: 120,000 chars/2min
  - Claude: 700 req/min
  - OpenAI: 1500 req/min
• Live logging shows progress at each stage
• Single Ctrl+C aborts immediately (when run in the true main thread)
"""

from __future__ import annotations

import hashlib
import re
import time
import signal
import threading
from contextlib import contextmanager
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
    wait,
    FIRST_COMPLETED,
)
from pathlib import Path
from typing import Callable, Dict, List, Tuple, DefaultDict
from collections import defaultdict

from requests.exceptions import RequestException
from openai import OpenAI

from .config import (
    REHUMANIZE_N,
    HUMANIZER_MAX_WORKERS, GEMINI_MAX_WORKERS, DETECTOR_MAX_WORKERS,
    SAPLING_MAX_CONCURRENT,
    OPENAI_API_KEY, HUMANIZER_OPENAI_API_KEY,
    MIN_WORDS_PARAGRAPH,
)
from .detectors import gptzero, sapling
from .docx_utils import extract_paragraphs_with_type
from .evaluation.quality import quality
from .humanizers.humanizer import humanize, _select_prompt
from .models import MODEL_REGISTRY

# Derive default list of models from registry display-names
DEFAULT_HUMANIZER_MODELS = list(MODEL_REGISTRY)

# ─────────────────────── Signal handling ────────────────────────
def _sigint_handler(sig, frame):
    raise KeyboardInterrupt

# Only install our SIGINT handler if we're truly in the main interpreter thread;
# avoids ValueError when imported under Streamlit's script-runner.
if threading.current_thread() is threading.main_thread():
    signal.signal(signal.SIGINT, _sigint_handler)

# ═══════════════ 1 · Generic helpers ════════════════════════════
@contextmanager
def _fast_pool(*args, **kwargs):
    pool = ThreadPoolExecutor(*args, **kwargs)
    try:
        yield pool
    finally:
        # Cancel pending on shutdown to abort quickly
        pool.shutdown(wait=False, cancel_futures=True)

_hash = lambda txt: hashlib.sha256(txt.encode("utf-8")).hexdigest()

def _maybe_log(message: str, cb: Callable[[str], None] | None = None):
    """
    Send the message to the UI logger callback (if provided)
    and also print it to the terminal with a timestamp.
    """
    timestamped = f"[{time.strftime('%H:%M:%S')}] {message}"
    if callable(cb):
        try:
            cb(timestamped)
        except Exception:
            # swallow UI-logging errors
            pass
    print(timestamped, flush=True)

def _stage(message: str, cb: Callable[[str], None] | None = None):
    """Stage boundary logging: prefixes with ▶️ and logs."""
    _maybe_log(f"▶️  {message}", cb)

# Global pool for Gemini calls (caps parallelism)
_GEMINI_POOL = ThreadPoolExecutor(max_workers=GEMINI_MAX_WORKERS)

# Global semaphore for Sapling concurrency control
_SAPLING_SEMAPHORE = threading.Semaphore(SAPLING_MAX_CONCURRENT)

# ═══════════════ 2 · Timeout helper (interrupt-aware) ════════════════
def _call_with_timeout(fn, *args, timeout: int = 300, **kwargs):
    """
    Run *fn* in a worker thread and raise RuntimeError if it takes
    longer than *timeout* seconds. Polls every second for interrupts.
    Default timeout increased to 300s (5 minutes) for robustness.
    """
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="wdog") as pool:
        fut = pool.submit(fn, *args, **kwargs)
        start = time.time()
        try:
            while True:
                done, _ = wait([fut], timeout=1, return_when=FIRST_COMPLETED)
                if fut in done:
                    return fut.result()
                elapsed = time.time() - start
                if elapsed >= timeout:
                    fut.cancel()
                    raise RuntimeError(f"Operation timed-out after {timeout}s")
                # Log progress for long-running operations
                if elapsed > 60 and int(elapsed) % 30 == 0:
                    _maybe_log(f"⏳ Still running... {elapsed:.0f}s elapsed", kwargs.get('log'))
        except KeyboardInterrupt:
            fut.cancel()
            raise

# ═══════════════ 3 · Detector helpers ════════════════════════════
_SENT_RE = re.compile(r"(?<=[.!?])\s+")
def _split_sentences(text: str) -> List[str]:
    parts = [t.strip() for t in _SENT_RE.split(text.strip()) if t.strip()]
    return parts or [text.strip()]

def _detect_gptzero(text: str, paragraphs: List[str], *, skip_cache: bool, log=None):
    if not skip_cache:
        cached = gptzero.get("gptzero", text)
        if cached is not None:
            _maybe_log("GPTZero: ✨ cache hit — scores retrieved", log)
            raw = cached
        else:
            _maybe_log("GPTZero: 🔄 cache miss — computing scores", log)
            raw = gptzero.detect_ai(text)
    else:
        _maybe_log("GPTZero: 🔄 cache miss — computing scores", log)
        raw = gptzero.detect_ai(text, skip_cache=True)

    doc_score = raw["documents"][0]["completely_generated_prob"]
    para_raw  = raw["documents"][0].get("paragraphs") or []
    if len(para_raw) == len(paragraphs):
        para_scores = [p["completely_generated_prob"] for p in para_raw]
    else:
        para_scores = [doc_score] * len(paragraphs)

    _maybe_log(f"GPTZero: doc_score={doc_score}", log)
    return doc_score, para_scores


def _detect_sapling(text: str, paragraphs: List[str], *, skip_cache: bool, log=None):
    # Acquire semaphore to limit Sapling concurrency
    with _SAPLING_SEMAPHORE:
        if not skip_cache:
            # Try cache first for original documents
            cached = sapling.get("sapling", text)
            if cached is not None:
                _maybe_log("Sapling: ✨ cache hit — scores retrieved", log)
                raw = cached
            else:
                _maybe_log("Sapling: 🔄 cache miss — computing scores", log)
                raw = sapling.detect_ai(text, skip_cache=True)
        else:
            # Skip cache for humanized drafts
            _maybe_log("Sapling: 🔄 computing scores (new draft)", log)
            raw = sapling.detect_ai(text, skip_cache=True)

        doc_score   = raw["score"]
        sent_scores = [s["score"] for s in raw.get("sentence_scores", [])]
        para_scores, idx = [], 0
        for para in paragraphs:
            n_sent = len(_split_sentences(para))
            if idx + n_sent <= len(sent_scores):
                chunk = sent_scores[idx:idx+n_sent]
                idx += n_sent
                para_scores.append(sum(chunk)/len(chunk))
            else:
                para_scores.append(doc_score)

        _maybe_log(f"Sapling: doc_score={doc_score}", log)
        return doc_score, para_scores


def _detect_both(text: str, paras: List[str], *, skip_cache: bool, log=None):
    """Run both detectors concurrently."""
    with _fast_pool(max_workers=2) as pool:
        # Both detectors respect skip_cache parameter
        fut_gz = pool.submit(_detect_gptzero, text, paras, skip_cache=skip_cache, log=log)
        fut_sp = pool.submit(_detect_sapling, text, paras, skip_cache=skip_cache, log=log)
        gz_doc, gz_par = fut_gz.result()
        sp_doc, sp_par = fut_sp.result()
    return {"g_doc": gz_doc, "s_doc": sp_doc,
            "g_par": gz_par, "s_par": sp_par}

# ═══════════════ 4 · Concurrent detector scoring ═══════════════════════
def _score_all_texts_concurrently(texts_paras: List[Tuple[str, List[str]]], log=None):
    uniq = {_hash(t): (t, p) for t, p in texts_paras}
    baseline_hash = _hash(texts_paras[0][0])  # first entry == original document
    baseline_text, baseline_paras = texts_paras[0]

    doc_scores_gz, doc_scores_sp = {}, {}
    para_scores_gz, para_scores_sp = {}, {}

    # Check if baseline is already cached
    baseline_cached = False
    if baseline_hash in uniq:
        # Try to get cached scores for the original document
        cached_gz = gptzero.get("gptzero", baseline_text)
        cached_sp = sapling.get("sapling", baseline_text)
        
        if cached_gz is not None and cached_sp is not None:
            # Extract scores directly from cache without API calls
            _maybe_log("Original document: ✨ using cached scores (no API calls)", log)
            
            # Extract GPTZero scores from cache
            doc_scores_gz[baseline_hash] = cached_gz["documents"][0]["completely_generated_prob"]
            para_raw = cached_gz["documents"][0].get("paragraphs") or []
            if len(para_raw) == len(baseline_paras):
                gz_para_scores = [p["completely_generated_prob"] for p in para_raw]
            else:
                gz_para_scores = [doc_scores_gz[baseline_hash]] * len(baseline_paras)
            
            # Extract Sapling scores from cache
            doc_scores_sp[baseline_hash] = cached_sp["score"]
            sent_scores = [s["score"] for s in cached_sp.get("sentence_scores", [])]
            sp_para_scores, idx = [], 0
            for para in baseline_paras:
                n_sent = len(_split_sentences(para))
                if idx + n_sent <= len(sent_scores):
                    chunk = sent_scores[idx:idx+n_sent]
                    idx += n_sent
                    sp_para_scores.append(sum(chunk)/len(chunk))
                else:
                    sp_para_scores.append(doc_scores_sp[baseline_hash])
            
            # Update paragraph scores
            para_scores_gz.update({_hash(pt): s for pt, s in zip(baseline_paras, gz_para_scores)})
            para_scores_sp.update({_hash(pt): s for pt, s in zip(baseline_paras, sp_para_scores)})
            
            baseline_cached = True
            # Remove baseline from work queue
            uniq.pop(baseline_hash, None)

    # Count only new texts that need scoring
    new_texts_count = len(uniq)
    if new_texts_count == 0:
        _stage("✓ Detector scoring complete (baseline cached)", log)
        return doc_scores_gz, doc_scores_sp, para_scores_gz, para_scores_sp

    _stage(f"Detector scoring phase • {new_texts_count} new texts to score", log)

    with _fast_pool(max_workers=DETECTOR_MAX_WORKERS) as pool:
        fut2h = {}
        for h, (t, p) in uniq.items():
            skip_cache = True  # All remaining texts are new drafts, skip cache
            fut = pool.submit(_detect_both, t, p, skip_cache=skip_cache, log=log)
            fut2h[fut] = h

        completed = 0
        for fut in as_completed(fut2h):
            completed += 1
            h = fut2h[fut]
            t, p = uniq[h]
            res = fut.result()
            doc_scores_gz[h] = res["g_doc"]
            doc_scores_sp[h] = res["s_doc"]
            para_scores_gz.update({_hash(pt): s for pt, s in zip(p, res["g_par"])})
            para_scores_sp.update({_hash(pt): s for pt, s in zip(p, res["s_par"])})
            _maybe_log(f"Detector progress: {completed}/{new_texts_count}", log)

    _stage("✓ Detector scoring complete", log)
    return doc_scores_gz, doc_scores_sp, para_scores_gz, para_scores_sp


# ═══════════════ 5 · Gemini quality helper ════════════════════════════
def _batch_quality_check(pairs: List[Tuple[str, str]], log=None):
    _stage(f"Gemini quality check • {len(pairs)} pair(s)", log)
    if not pairs:
        _stage("✓ Gemini check done (no pairs)", log)
        return {}

    unique = list(set(pairs))
    _maybe_log(f"Gemini quality on {len(unique)} unique pairs", log)

    out: Dict[Tuple[str, str], Dict] = {}
    with _fast_pool(max_workers=GEMINI_MAX_WORKERS) as pool:
        fut2key = {pool.submit(quality, o, h): (_hash(o), _hash(h)) for o, h in unique}
        done = 0
        try:
            for fut in as_completed(fut2key):
                k = fut2key[fut]
                out[k] = fut.result()
                done += 1
                if done % 5 == 0 or done == len(unique):
                    _maybe_log(f"Quality {done}/{len(unique)}", log)
        except KeyboardInterrupt:
            pool.shutdown(wait=False, cancel_futures=True)
            raise

    _stage("✓ Gemini check done", log)
    return out

# ═══════════════ 6 · Humaniser helpers ═══════════════════════════════
def _humanize_doc(text: str, model: str, log=None) -> str:
    _stage(f"Doc humanization START • {model}", log)
    start_time = time.time()
    # No timeout wrapper needed - humanizer has its own timeout and rate limiting
    out = humanize(text, model, "doc", log=log)
    elapsed = time.time() - start_time
    _stage(f"Doc humanization DONE • {model} • {elapsed:.1f}s", log)
    return out

def _humanize_paragraphs(paragraphs: List[str], model: str, log=None) -> Tuple[List[str], List[Dict]]:
    """
    Paragraph-wise humanisation with mismatch tracking and pair storage.
    Returns (humanized_paragraphs, paragraph_pair_info)
    
    paragraph_pair_info contains the actual original→humanized pairs for each paragraph,
    including mismatch information and quality evaluation pairing.
    """
    _stage(f"Para humanization START • {model} • {len(paragraphs)} paragraphs", log)
    start_time = time.time()
    if not paragraphs:
        return [], []

    max_workers = min(HUMANIZER_MAX_WORKERS, len(paragraphs))
    out = [None] * len(paragraphs)
    pair_info = [None] * len(paragraphs)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        fut2idx = {
            pool.submit(humanize, p, model, "para"): i
            for i, p in enumerate(paragraphs)
        }
        completed = 0
        try:
            for fut in as_completed(fut2idx):
                idx = fut2idx[fut]
                original_para = paragraphs[idx]
                humanized_text = fut.result()
                
                # Split humanized text into paragraphs (by double newlines or single newlines)
                received_paras = [p.strip() for p in humanized_text.split('\n\n') if p.strip()]
                if not received_paras:
                    # Fallback: split by single newlines
                    received_paras = [p.strip() for p in humanized_text.split('\n') if p.strip()]
                if not received_paras:
                    # Last fallback: treat as single paragraph
                    received_paras = [humanized_text.strip()] if humanized_text.strip() else [""]
                
                # Detect mismatch: sent 1 paragraph, received N paragraphs
                is_mismatch = len(received_paras) != 1
                
                # The text used for document assembly and quality evaluation is the same
                assembly_text = "\n\n".join(received_paras) if received_paras else ""

                # Store the actual pair information
                pair_info[idx] = {
                    "original_paragraph": original_para,
                    "original_index": idx,
                    "humanized_paragraphs": received_paras,
                    "is_mismatch": is_mismatch,
                    "sent_count": 1,
                    "received_count": len(received_paras),
                    # For quality evaluation, use the FULL received text
                    "quality_evaluation_text": assembly_text,
                    # For document assembly, also use the FULL received text
                    "document_assembly_text": assembly_text
                }
                
                # For output (document assembly), use appropriate text
                out[idx] = pair_info[idx]["document_assembly_text"]
                
                completed += 1
                if completed % 5 == 0 or completed == len(paragraphs):
                    _maybe_log(f"Para progress: {completed}/{len(paragraphs)} • {model}", log)
        except KeyboardInterrupt:
            pool.shutdown(wait=False, cancel_futures=True)
            raise

    elapsed = time.time() - start_time
    _stage(f"Para humanization DONE • {model} • {elapsed:.1f}s total", log)
    
    # Count mismatches
    total_mismatches = sum(1 for info in pair_info if info and info["is_mismatch"])
    if total_mismatches > 0:
        _maybe_log(f"⚠️ Para mode mismatches: {total_mismatches}/{len(paragraphs)} paragraphs", log)
    
    return out, pair_info

# ═══════════════ 7 · Paragraph helper ════════════════════════════════
def _merge_heading_content(para_objs, hum_content):
    out, idx = [], 0
    for p in para_objs:
        if p["type"] == "content":
            out.append(hum_content[idx]); idx += 1
        else:
            out.append(p["text"])
    return out

def _determine_paragraph_types(paragraphs: List[str]) -> List[str]:
    """
    Determine paragraph types for humanized text using the same logic as docx_utils.
    Returns list of 'content' or 'heading' for each paragraph.
    """
    types = []
    for para in paragraphs:
        text = para.strip()
        if not text:
            continue
        para_type = 'content' if len(text.split()) >= MIN_WORDS_PARAGRAPH else 'heading'
        types.append(para_type)
    return types

def _detect_paragraph_mismatch(orig_para_objs: List[Dict], hum_paras: List[str]) -> Tuple[bool, str]:
    """
    Detect paragraph structure mismatch between original and humanized documents.
    
    Returns:
        Tuple[bool, str]: (is_mismatch, reason)
        
    Checks:
        1. Count mismatch: Different number of paragraphs
        2. Structure mismatch: Different sequence of heading/content types
    """
    # Get original types
    orig_types = [p["type"] for p in orig_para_objs]
    
    # Determine humanized types
    hum_types = _determine_paragraph_types(hum_paras)
    
    # Check 1: Count mismatch
    if len(orig_types) != len(hum_types):
        return True, f"Count mismatch: original has {len(orig_types)} paragraphs, humanized has {len(hum_types)}"
    
    # Check 2: Structure mismatch (same count but different type sequence)
    if orig_types != hum_types:
        # Find first mismatch position for better error message
        mismatch_pos = -1
        for i, (orig_type, hum_type) in enumerate(zip(orig_types, hum_types)):
            if orig_type != hum_type:
                mismatch_pos = i
                break
        
        orig_structure = "->".join(orig_types)
        hum_structure = "->".join(hum_types)
        
        if mismatch_pos >= 0:
            reason = f"Structure mismatch at position {mismatch_pos + 1}: expected '{orig_types[mismatch_pos]}', got '{hum_types[mismatch_pos]}'."
        else:
            reason = f"Structure mismatch"
        
        return True, reason
    
    # No mismatch detected
    return False, "No mismatch"

# ═══════════════ 8 · Draft generator ════════════════════════════════
def _generate_single_draft(
    model: str,
    iteration: int,
    orig_text: str,
    para_objs,
    *,
    include_para: bool = True,
    include_doc: bool = True,  # NEW: control doc mode
    is_para_folder: bool = False,  # NEW: flag for para folders
    log=None,
):
    """
    Generate one draft. 
    - include_para: Whether to run paragraph mode (for regular folders)
    - include_doc: Whether to run document mode (optional for regular folders)
    - is_para_folder: True for ai_paras/human_paras folders (use para prompt for single para)
    """
    _stage(f"Starting draft generation • model={model} • iter={iteration+1}", log)
    
    specs = []
    
    # For para folders: use para mode prompt for the single paragraph
    if is_para_folder:
        # These folders contain single paragraphs, so use para mode
        hum_text = humanize(orig_text, model, "para", log=log)
        
        # Analyze the result for paragraph-level mismatch
        received_paras = [p.strip() for p in hum_text.split('\n\n') if p.strip()]
        if not received_paras:
            # Fallback: split by single newlines
            received_paras = [p.strip() for p in hum_text.split('\n') if p.strip()]
        if not received_paras:
            # Last fallback: treat as single paragraph
            received_paras = [hum_text.strip()] if hum_text.strip() else [""]
        
        # For para folders, we sent 1 paragraph and expect 1 back
        is_mismatch = len(received_paras) != 1
        para_level_mismatches = 1 if is_mismatch else 0
        has_para_mismatches = is_mismatch
        
        # Create single pair info for consistency
        para_pair_info = [{
            "original_paragraph": orig_text,
            "original_index": 0,
            "humanized_paragraphs": received_paras,
            "is_mismatch": is_mismatch,
            "sent_count": 1,
            "received_count": len(received_paras),
            "quality_evaluation_text": received_paras[0] if received_paras else "",
            "document_assembly_text": "\n\n".join(received_paras) if len(received_paras) > 1 else (received_paras[0] if received_paras else "")
        }]
        
        doc_paras = [received_paras[0] if received_paras else ""]  # Use first paragraph for document assembly
        _maybe_log(f"Para-folder humanization complete • {model}", log)
        
        if is_mismatch:
            _maybe_log(f"⚠️ Para folder mismatch: sent 1 paragraph, received {len(received_paras)}", log)
        
        specs.append({
            "model": model,
            "mode": "para",  # Mark as para mode for consistency
            "iter": iteration,
            "humanized_text": hum_text,
            "humanized_paras_resolved": doc_paras,
            "para_pair_info": para_pair_info,  # Store the pair info
            "para_level_mismatches": para_level_mismatches,  # Count of mismatched paragraphs
            "has_para_level_mismatches": has_para_mismatches,  # Boolean flag
        })
    else:
        # Regular folders (ai_texts, human_texts)
        
        # ── Doc-level (optional) ───────────────────────────────────────────────────
        if include_doc:
            hum_doc = _humanize_doc(orig_text, model, log)
            doc_paras = [p.strip() for p in hum_doc.splitlines() if p.strip()]
            _maybe_log(f"Doc-mode complete • {model} • {len(doc_paras)} paragraphs", log)

            specs.append({
                "model": model,
                "mode": "doc",
                "iter": iteration,
                "humanized_text": hum_doc,
                "humanized_paras_resolved": doc_paras,
            })

        # ── Paragraph-mode ────────────────────────────────────────────────────
        if include_para:
            content_paras = [p["text"] for p in para_objs if p["type"] == "content"]
            if content_paras:
                hum_para_content, para_pair_info = _humanize_paragraphs(content_paras, model, log)
                hum_para_paras   = _merge_heading_content(para_objs, hum_para_content)
                _maybe_log(f"Para-mode complete • {model} • {len(hum_para_paras)} paragraphs", log)

                # Check if any paragraphs are mismatched
                para_level_mismatches = sum(1 for info in para_pair_info if info and info["is_mismatch"])
                has_para_mismatches = para_level_mismatches > 0
                
                specs.append({
                    "model": model,
                    "mode": "para",
                    "iter": iteration,
                    "humanized_paras": hum_para_paras,
                    "humanized_paras_resolved": hum_para_paras,
                    "humanized_text": "\n\n".join(hum_para_paras),
                    "para_pair_info": para_pair_info,  # Store actual original→humanized pairs
                    "para_level_mismatches": para_level_mismatches,  # Count of mismatched paragraphs
                    "has_para_level_mismatches": has_para_mismatches,  # Boolean flag
                })

    _stage(f"✓ Draft generation done • model={model} • iter={iteration+1}", log)
    return specs


def _generate_all_drafts(models, iterations, orig_text, para_objs,
                         log=None, *, include_para: bool = True, 
                         include_doc: bool = True, is_para_folder: bool = False):
    out: List[Dict] = []
    total_tasks = len(models) * iterations
    max_workers = min(HUMANIZER_MAX_WORKERS, total_tasks)

    if is_para_folder:
        modes_lbl = "para-folder drafts"
    elif include_doc and include_para:
        modes_lbl = "draft pairs"
    elif include_doc:
        modes_lbl = "doc-only drafts"
    else:
        modes_lbl = "para-only drafts"
        
    _stage(f"Generating {total_tasks} {modes_lbl}", log)
    
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        fut_to_info = {}
        for m in models:
            for i in range(iterations):
                fut = pool.submit(
                    _generate_single_draft,
                    m, i, orig_text, para_objs,
                    include_para=include_para, 
                    include_doc=include_doc,
                    is_para_folder=is_para_folder,
                    log=log,
                )
                fut_to_info[fut] = (m, i)

        completed = 0
        try:
            for fut in as_completed(fut_to_info):
                completed += 1
                model, iter_num = fut_to_info[fut]
                _maybe_log(f"Progress: {completed}/{total_tasks} • {model} iter {iter_num+1}", log)
                out.extend(fut.result())
        except KeyboardInterrupt:
            pool.shutdown(wait=False, cancel_futures=True)
            raise

    _stage(f"✓ All {len(out)} drafts generated", log)
    return out

# ═══════════════ 9 · Assembly helpers ═══════════════════════════════
_EXPECTED_FLAGS = (
    "length_ok","same_meaning","same_lang","no_missing_info",
    "citation_preserved","citation_content_ok",
)

def _assemble_scores_from_batch(
    doc_text: str, paragraphs: List[str],
    doc_scores_gz: Dict[str,float], doc_scores_sp: Dict[str,float],
    para_scores_gz: Dict[str,float], para_scores_sp: Dict[str,float],
) -> Dict[str, Dict]:
    g_doc = doc_scores_gz.get(_hash(doc_text))
    s_doc = doc_scores_sp.get(_hash(doc_text))
    g_ind_par = [para_scores_gz.get(_hash(p)) for p in paragraphs]
    s_ind_par = [para_scores_sp.get(_hash(p)) for p in paragraphs]
    counts    = [len(p.split()) for p in paragraphs]

    def _weighted(scores, default):
        good = [(s, c) for s, c in zip(scores, counts) if isinstance(s, (int, float))]
        return (sum(s * c for s, c in good) / sum(c for _, c in good)) if good else default

    g_ind_doc = _weighted(g_ind_par, g_doc)
    s_ind_doc = _weighted(s_ind_par, s_doc)
    pad = lambda lst, fb: [fb if x is None else x for x in lst]

    return {
        "group_doc": {"gptzero": g_doc, "sapling": s_doc},
        "ind_doc":   {"gptzero": g_ind_doc, "sapling": s_ind_doc},
        "group_par": {"gptzero": pad(g_ind_par, g_doc), "sapling": pad(s_ind_par, s_doc)},
        "ind_par":   {"gptzero": pad(g_ind_par, g_doc), "sapling": pad(s_ind_par, s_doc)},
    }

def _assemble_per_para_stats(
    orig: List[str], hum: List[str],
    ai_before: Dict[str, List[float]], ai_after: Dict[str, List[float]],
    quality_results: Dict[Tuple[str, str], Dict[str, bool]],
    para_objs: List[Dict[str, str]],  # paragraph objects with type info
    para_pair_info: List[Dict] = None,  # NEW: actual original→humanized pairs
):
    """
    Assemble per-paragraph statistics with proper heading/content separation.
    
    - Content paragraphs: included in quality stats and metrics
    - Headings: included in details for display but excluded from aggregated quality stats
    - Length deviations calculated at both paragraph and draft level
    - Enhanced citation preservation metrics
    - Paragraph-level mismatch tracking (for para mode)
    """
    flags_total = {k: 0 for k in _EXPECTED_FLAGS}
    
    # New numeric level aggregators
    same_meaning_levels = []
    missing_info_levels = []
    grammar_scores = []
    all_grammar_errors = []
    
    # Length deviation tracking
    para_length_deviations = []  # percentage deviations for content paragraphs
    
    # Citation tracking for enhanced metrics
    paragraphs_with_citations_orig = 0  # count of original paragraphs with citations
    paragraphs_citations_preserved = 0   # count where ALL citations preserved (Gemini)
    paragraphs_citation_content_ok = 0   # count where citation content is regex-preserved
    total_citations_humanized = 0       # total citations in humanized version
    exact_match_citations = 0           # citations with exact ID match
    
    # Details for UI display (includes both headings and content)
    details: List[Dict] = []
    content_para_count = 0  # track actual content paragraphs

    for idx, (o, h) in enumerate(zip(orig, hum)):
        para_type = para_objs[idx]["type"] if idx < len(para_objs) else "content"
        raw = quality_results.get((_hash(o), _hash(h)), {})
        
        # Word counts for length deviation
        wc_before = len(o.split())
        wc_after = len(h.split())
        
        if para_type == "content":
            content_para_count += 1
            
            # Extract boolean flags (backward compatible)
            p_flags = {}
            for k in _EXPECTED_FLAGS:
                value = raw.get(k)
                if value is None:
                    # For citation metrics, None means not applicable (no citations)
                    p_flags[k] = None
                else:
                    p_flags[k] = bool(value)
                    # Only count in totals if not None
                    if p_flags[k] is True:
                        flags_total[k] += 1
            
            # Extract new numeric levels
            same_meaning_level = raw.get("same_meaning_level")
            missing_info_level = raw.get("missing_info_level") 
            grammar_level = raw.get("grammar_level")
            
            if same_meaning_level is not None:
                same_meaning_levels.append(same_meaning_level)
            if missing_info_level is not None:
                missing_info_levels.append(missing_info_level)
            if grammar_level is not None:
                grammar_scores.append(grammar_level)
            
            # Grammar errors
            grammar_errors = raw.get("grammar_errors", [])
            all_grammar_errors.extend(grammar_errors)
            
            # Length deviation (percentage) for content paragraphs
            if wc_before > 0:
                length_deviation = ((wc_after - wc_before) / wc_before) * 100
                para_length_deviations.append(length_deviation)
            
            # Citation preservation analysis for content paragraphs
            orig_citations = _citations(o)
            hum_citations = _citations(h)
            
            if orig_citations:  # This paragraph has citations originally
                paragraphs_with_citations_orig += 1
                
                # Check citation_preserved from quality results (Gemini evaluation)
                if p_flags.get("citation_preserved") is True:
                    paragraphs_citations_preserved += 1
                
                # Check citation_content_ok from quality results (regex check)
                if p_flags.get("citation_content_ok") is True:
                    paragraphs_citation_content_ok += 1
            
            # Count humanized citations for exact match analysis
            if hum_citations:
                total_citations_humanized += len(hum_citations)
                # Check exact matches (simplified - could be enhanced with regex)
                for hum_cite in hum_citations:
                    if hum_cite in orig_citations:
                        exact_match_citations += 1
            
        else:
            # Heading: empty quality flags for display but don't affect aggregates
            p_flags = {k: None for k in _EXPECTED_FLAGS}  # None indicates no check performed
            same_meaning_level = None
            missing_info_level = None
            grammar_level = None
            grammar_errors = []

        # Get paragraph mismatch info for this paragraph (content paragraphs only)
        content_para_idx = sum(1 for i in range(idx) if para_objs[i]["type"] == "content") if para_type == "content" else -1
        para_mismatch_data = None
        if para_pair_info and para_type == "content" and content_para_idx < len(para_pair_info):
            para_mismatch_data = para_pair_info[content_para_idx]

        # Add to details (for both headings and content)
        detail_entry = {
            "paragraph": idx + 1,
            "type": para_type,
            "wc_before": wc_before, 
            "wc_after": wc_after,
            "ai_before": {d: ai_before.get(d, [None]*len(orig))[idx] for d in ("gptzero","sapling")},
            "ai_after":  {d: ai_after.get(d, [None]*len(orig))[idx] for d in ("gptzero","sapling")},
            "flags": p_flags,
            "same_meaning_level": same_meaning_level,
            "missing_info_level": missing_info_level, 
            "grammar_level": grammar_level,
            "grammar_errors": grammar_errors,
            # Additional details from quality check
            "missing_items": raw.get("missing_items", []) if para_type == "content" else [],
            "added_items": raw.get("added_items", []) if para_type == "content" else [],
            "same_meaning_details": raw.get("same_meaning_details", "") if para_type == "content" else "",
        }
        
        # Add paragraph mismatch information for content paragraphs
        if para_type == "content" and para_mismatch_data:
            detail_entry.update({
                "para_is_mismatch": para_mismatch_data["is_mismatch"],
                "para_sent_count": para_mismatch_data["sent_count"],
                "para_received_count": para_mismatch_data["received_count"],
                "para_received_paragraphs": para_mismatch_data["humanized_paragraphs"],  # Updated key
                "para_original_paragraph": para_mismatch_data["original_paragraph"],  # Store original for comparison
                "para_quality_evaluation_text": para_mismatch_data["quality_evaluation_text"],  # Text used for quality eval
            })
        else:
            detail_entry.update({
                "para_is_mismatch": False,
                "para_sent_count": 1,
                "para_received_count": 1,
                "para_received_paragraphs": [],
                "para_original_paragraph": "",
                "para_quality_evaluation_text": "",
            })
        
        details.append(detail_entry)
    
    # Calculate aggregate metrics based on CONTENT paragraphs only
    # Add backward-compatible aggregate scores
    flags_total["grammar_score"] = sum(grammar_scores) / len(grammar_scores) if grammar_scores else None
    flags_total["grammar_errors"] = all_grammar_errors
    
    # NEW: Add numeric level averages
    flags_total["same_meaning_level_avg"] = sum(same_meaning_levels) / len(same_meaning_levels) if same_meaning_levels else None
    flags_total["missing_info_level_avg"] = sum(missing_info_levels) / len(missing_info_levels) if missing_info_levels else None
    
    # NEW: Add length deviation metrics
    flags_total["para_length_deviation_avg"] = sum(para_length_deviations) / len(para_length_deviations) if para_length_deviations else 0
    flags_total["para_length_deviations"] = para_length_deviations  # for distribution analysis
    
    # NEW: Enhanced citation preservation metrics  
    flags_total["paragraph_citation_preservation_rate"] = (
        (paragraphs_citations_preserved / paragraphs_with_citations_orig * 100) 
        if paragraphs_with_citations_orig > 0 else 100  # 100% if no citations to preserve
    )
    flags_total["paragraph_citation_content_ok_rate"] = (
        (paragraphs_citation_content_ok / paragraphs_with_citations_orig * 100)
        if paragraphs_with_citations_orig > 0 else 100  # 100% if no citations to check
    )
    flags_total["citation_exact_match_rate"] = (
        (paragraphs_citation_content_ok / paragraphs_with_citations_orig * 100)
        if paragraphs_with_citations_orig > 0 else 100  # 100% if no citations to check
    )
    
    # Additional metadata
    flags_total["content_paragraph_count"] = content_para_count
    flags_total["total_segments"] = len(orig)
    
    return details, flags_total


def _assemble_per_para_stats_from_pairs(
    ai_before: Dict[str, List[float]], ai_after: Dict[str, List[float]],
    quality_results: Dict[Tuple[str, str], Dict[str, bool]],
    para_objs: List[Dict[str, str]],  # paragraph objects with type info
    para_pair_info: List[Dict],  # actual original→humanized pairs
    para_scores_gz: Dict[str, float], para_scores_sp: Dict[str, float]
):
    """
    Assemble per-paragraph statistics using para_pair_info for cases with document-level mismatches.
    This ensures we show paragraph analysis even when document structure doesn't match.
    """
    flags_total = {k: 0 for k in _EXPECTED_FLAGS}
    
    # New numeric level aggregators
    same_meaning_levels = []
    missing_info_levels = []
    grammar_scores = []
    all_grammar_errors = []
    
    # Length deviation tracking
    para_length_deviations = []  # percentage deviations for content paragraphs
    
    # Citation tracking for enhanced metrics
    paragraphs_with_citations_orig = 0  # count of original paragraphs with citations
    paragraphs_citations_preserved = 0   # count where ALL citations preserved (Gemini)
    paragraphs_citation_content_ok = 0   # count where citation content is regex-preserved
    total_citations_humanized = 0       # total citations in humanized version
    exact_match_citations = 0           # citations with exact ID match
    
    # Details for UI display (includes both headings and content)
    details: List[Dict] = []
    content_para_count = 0  # track actual content paragraphs
    content_pair_idx = 0    # track position in para_pair_info

    # Go through all original paragraphs and create details
    for idx, para_obj in enumerate(para_objs):
        para_type = para_obj["type"]
        original_text = para_obj["text"]
        
        if para_type == "content":
            # For content paragraphs, use para_pair_info
            if content_pair_idx < len(para_pair_info) and para_pair_info[content_pair_idx]:
                pair_info = para_pair_info[content_pair_idx]
                original_para = pair_info["original_paragraph"]
                humanized_text = pair_info["quality_evaluation_text"]
                is_mismatch = pair_info["is_mismatch"]
                
                # Get quality results for this exact pair
                raw = quality_results.get((_hash(original_para), _hash(humanized_text)), {})
                
                # Word counts
                wc_before = len(original_para.split())
                wc_after = len(humanized_text.split())
                
                content_para_count += 1
                
                # Extract boolean flags
                p_flags = {}
                for k in _EXPECTED_FLAGS:
                    value = raw.get(k)
                    if value is None:
                        p_flags[k] = None
                    else:
                        p_flags[k] = bool(value)
                        if p_flags[k] is True:
                            flags_total[k] += 1
                
                # Extract new numeric levels
                same_meaning_level = raw.get("same_meaning_level")
                missing_info_level = raw.get("missing_info_level") 
                grammar_level = raw.get("grammar_level")
                
                if same_meaning_level is not None:
                    same_meaning_levels.append(same_meaning_level)
                if missing_info_level is not None:
                    missing_info_levels.append(missing_info_level)
                if grammar_level is not None:
                    grammar_scores.append(grammar_level)
                
                # Grammar errors
                grammar_errors = raw.get("grammar_errors", [])
                all_grammar_errors.extend(grammar_errors)
                
                # Length deviation
                if wc_before > 0:
                    length_deviation = ((wc_after - wc_before) / wc_before) * 100
                    para_length_deviations.append(length_deviation)
                
                # Citation preservation analysis
                orig_citations = _citations(original_para)
                hum_citations = _citations(humanized_text)
                
                if orig_citations:
                    paragraphs_with_citations_orig += 1
                    if p_flags.get("citation_preserved") is True:
                        paragraphs_citations_preserved += 1
                    if p_flags.get("citation_content_ok") is True:
                        paragraphs_citation_content_ok += 1
                
                if hum_citations:
                    total_citations_humanized += len(hum_citations)
                    for hum_cite in hum_citations:
                        if hum_cite in orig_citations:
                            exact_match_citations += 1
                
                # Get AI scores for this paragraph
                gz_before = para_scores_gz.get(_hash(original_para))
                sp_before = para_scores_sp.get(_hash(original_para))
                gz_after = para_scores_gz.get(_hash(humanized_text))
                sp_after = para_scores_sp.get(_hash(humanized_text))
                
                content_pair_idx += 1
            else:
                # No pair info available for this content paragraph
                wc_before = len(original_text.split())
                wc_after = wc_before  # assume no change
                p_flags = {k: None for k in _EXPECTED_FLAGS}
                same_meaning_level = None
                missing_info_level = None
                grammar_level = None
                grammar_errors = []
                raw = {}
                is_mismatch = False
                gz_before = gz_after = sp_before = sp_after = None
        else:
            # For headings, create basic entry
            wc_before = len(original_text.split())
            wc_after = wc_before  # headings typically don't change
            p_flags = {k: None for k in _EXPECTED_FLAGS}
            same_meaning_level = None
            missing_info_level = None
            grammar_level = None
            grammar_errors = []
            raw = {}
            is_mismatch = False
            
            # Get AI scores for heading
            gz_before = para_scores_gz.get(_hash(original_text))
            sp_before = para_scores_sp.get(_hash(original_text))
            gz_after = gz_before  # headings typically don't change
            sp_after = sp_before

        # Create detail entry
        detail_entry = {
            "paragraph": idx + 1,
            "type": para_type,
            "wc_before": wc_before, 
            "wc_after": wc_after,
            "ai_before": {"gptzero": gz_before, "sapling": sp_before},
            "ai_after":  {"gptzero": gz_after, "sapling": sp_after},
            "flags": p_flags,
            "same_meaning_level": same_meaning_level,
            "missing_info_level": missing_info_level, 
            "grammar_level": grammar_level,
            "grammar_errors": grammar_errors,
            "missing_items": raw.get("missing_items", []) if para_type == "content" else [],
            "added_items": raw.get("added_items", []) if para_type == "content" else [],
            "same_meaning_details": raw.get("same_meaning_details", "") if para_type == "content" else "",
        }
        
        # Add paragraph mismatch information for content paragraphs
        if para_type == "content":
            # For content paragraphs, use the pair info we just processed
            if content_pair_idx > 0 and (content_pair_idx - 1) < len(para_pair_info):
                pair_data = para_pair_info[content_pair_idx - 1]
                if pair_data:
                    detail_entry.update({
                        "para_is_mismatch": pair_data["is_mismatch"],
                        "para_sent_count": pair_data["sent_count"],
                        "para_received_count": pair_data["received_count"],
                        "para_received_paragraphs": pair_data["humanized_paragraphs"],
                        "para_original_paragraph": pair_data["original_paragraph"],
                        "para_quality_evaluation_text": pair_data["quality_evaluation_text"],
                    })
                else:
                    detail_entry.update({
                        "para_is_mismatch": False,
                        "para_sent_count": 1,
                        "para_received_count": 1,
                        "para_received_paragraphs": [],
                        "para_original_paragraph": "",
                        "para_quality_evaluation_text": "",
                    })
            else:
                detail_entry.update({
                    "para_is_mismatch": False,
                    "para_sent_count": 1,
                    "para_received_count": 1,
                    "para_received_paragraphs": [],
                    "para_original_paragraph": "",
                    "para_quality_evaluation_text": "",
                })
        else:
            # For headings, no pair info
            detail_entry.update({
                "para_is_mismatch": False,
                "para_sent_count": 1,
                "para_received_count": 1,
                "para_received_paragraphs": [],
                "para_original_paragraph": "",
                "para_quality_evaluation_text": "",
            })
        
        details.append(detail_entry)
    
    # Calculate aggregate metrics based on CONTENT paragraphs only
    flags_total["grammar_score"] = sum(grammar_scores) / len(grammar_scores) if grammar_scores else None
    flags_total["grammar_errors"] = all_grammar_errors
    flags_total["same_meaning_level_avg"] = sum(same_meaning_levels) / len(same_meaning_levels) if same_meaning_levels else None
    flags_total["missing_info_level_avg"] = sum(missing_info_levels) / len(missing_info_levels) if missing_info_levels else None
    flags_total["para_length_deviation_avg"] = sum(para_length_deviations) / len(para_length_deviations) if para_length_deviations else 0
    flags_total["para_length_deviations"] = para_length_deviations
    
    # Citation preservation metrics  
    flags_total["paragraph_citation_preservation_rate"] = (
        (paragraphs_citations_preserved / paragraphs_with_citations_orig * 100) 
        if paragraphs_with_citations_orig > 0 else 100
    )
    flags_total["paragraph_citation_content_ok_rate"] = (
        (paragraphs_citation_content_ok / paragraphs_with_citations_orig * 100)
        if paragraphs_with_citations_orig > 0 else 100
    )
    flags_total["citation_exact_match_rate"] = (
        (paragraphs_citation_content_ok / paragraphs_with_citations_orig * 100)
        if paragraphs_with_citations_orig > 0 else 100
    )
    
    # Additional metadata
    flags_total["content_paragraph_count"] = content_para_count
    flags_total["total_segments"] = len(details)
    
    return details, flags_total


# Import citation extraction from quality module
from .evaluation.quality import _citations

# ═══════════════ 10 · Main runner ════════════════════════════════
def run_test(doc_path: Path, models: List[str]|None=None,
             logger: Callable[[str],None]|None=None,
             iterations: int = REHUMANIZE_N,
             max_retries: int = 5,
             include_doc_mode: bool = True):  # NEW parameter
    _stage("[Pipeline] run_test START", logger)
    _maybe_log("="*60, logger)
    _maybe_log(f"Processing document: {doc_path.name}", logger)
    _maybe_log("="*60, logger)

    # Extract paragraphs
    _stage("extracting paragraphs", logger)
    try:
        para_objs = extract_paragraphs_with_type(doc_path)
        _maybe_log(f"Extracted {len(para_objs)} paragraphs", logger)
    except Exception as exc:
        _maybe_log(f"❌ paragraph extraction error: {exc}", logger)
        return {"document": doc_path.name, "runs": [], "paragraph_count": 0, "error": str(exc)}

    if not para_objs:
        _maybe_log("– SKIP (no paragraphs)", logger)
        return {"document": doc_path.name, "runs": [], "paragraph_count": 0, "empty": True}

    orig_paras = [p["text"] for p in para_objs]
    orig_full  = "\n\n".join(orig_paras)
    wc_before  = sum(len(p.split()) for p in orig_paras)
    models     = models or DEFAULT_HUMANIZER_MODELS

    # Check if this is a para folder
    is_para_folder = doc_path.parent.name in ("ai_paras", "human_paras")

    # Phase 1: Generation (with retries)
    _stage("Phase 1: Generation", logger)
    drafts = None
    for attempt in range(1, max_retries + 1):
        try:
            if attempt > 1:
                _maybe_log(f"🔄 Retrying Phase 1 (attempt {attempt}/{max_retries})", logger)
                time.sleep(min(30 * (attempt - 1), 120))  # exponential backoff: 30s, 60s, 120s
            
            if is_para_folder:
                # Para folders: only para mode with para prompt
                drafts = _generate_all_drafts(
                    models, iterations, orig_full, para_objs,
                    log=logger, 
                    include_para=False,  # Not regular para mode
                    include_doc=False,   # Not doc mode
                    is_para_folder=True, # Special handling
                )
            else:
                # Regular folders: respect include_doc_mode setting
                drafts = _generate_all_drafts(
                    models, iterations, orig_full, para_objs,
                    log=logger, 
                    include_para=True,
                    include_doc=include_doc_mode,
                    is_para_folder=False,
                )
            break  # Success, exit retry loop
            
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            _maybe_log(f"❌ Phase 1 error (attempt {attempt}): {exc}", logger)
            if attempt == max_retries:
                _maybe_log(f"❌ Phase 1 failed after {max_retries} attempts", logger)
                return {"document": doc_path.name, "runs": [], "paragraph_count": len(orig_paras), 
                        "error": f"Phase 1 failed: {exc}", "phase_failed": 1}

    # Phase 2: Detector scoring (with retries)
    _stage("Phase 2: Detector scoring", logger)
    texts_paras = [(orig_full, orig_paras)] + [
        (d["humanized_text"], d["humanized_paras_resolved"]) for d in drafts
    ]
    
    doc_scores_gz, doc_scores_sp, para_scores_gz, para_scores_sp = None, None, None, None
    for attempt in range(1, max_retries + 1):
        try:
            if attempt > 1:
                _maybe_log(f"🔄 Retrying Phase 2 (attempt {attempt}/{max_retries})", logger)
                time.sleep(min(30 * (attempt - 1), 120))
                
            doc_scores_gz, doc_scores_sp, para_scores_gz, para_scores_sp = \
                _score_all_texts_concurrently(texts_paras, logger)
            break
            
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            _maybe_log(f"❌ Phase 2 error (attempt {attempt}): {exc}", logger)
            if attempt == max_retries:
                _maybe_log(f"❌ Phase 2 failed after {max_retries} attempts - continuing with partial results", logger)
                # Continue with drafts but no detector scores
                return {"document": doc_path.name, "runs": drafts, "paragraph_count": len(orig_paras),
                        "warning": f"Phase 2 failed: {exc}", "phase_failed": 2}

    # Phase 3: Gemini quality checks (with retries)
    _stage("Phase 3: Gemini quality evaluation", logger)
    q_pairs = set()
    
    for d in drafts:
        # If it's para mode, para_pair_info is the source of truth for quality pairings.
        if d.get("mode") == "para" and d.get("para_pair_info"):
            for pair in d["para_pair_info"]:
                if pair:  # Ensure pair is not None
                    q_pairs.add((pair["original_paragraph"], pair["quality_evaluation_text"]))
        # For doc mode, or as a fallback if something went wrong with para_pair_info.
        # This part only works if paragraph counts match.
        elif len(orig_paras) == len(d["humanized_paras_resolved"]):
            for o, h in zip(orig_paras, d["humanized_paras_resolved"]):
                q_pairs.add((o, h))
        # If it's a doc-mode draft with a mismatch, quality checks for individual paragraphs
        # can't be paired reliably, so they would have been skipped anyway. This maintains that behavior.

    q_results = {}
    q_pairs_list = list(q_pairs)
    if not q_pairs_list:
        _maybe_log("– SKIP quality checks (no pairs)", logger)
    else:
        for attempt in range(1, max_retries + 1):
            try:
                if attempt > 1:
                    _maybe_log(f"🔄 Retrying Phase 3 (attempt {attempt}/{max_retries})", logger)
                    time.sleep(min(30 * (attempt - 1), 120))
                    
                q_results = _batch_quality_check(q_pairs_list, logger)
                break
                
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                _maybe_log(f"❌ Phase 3 error (attempt {attempt}): {exc}", logger)
                if attempt == max_retries:
                    _maybe_log(f"⚠️ Phase 3 failed after {max_retries} attempts - continuing without quality checks", logger)
                    q_results = {}  # Continue with empty quality results

    # Phase 4: Assembly
    _stage("Phase 4: Assembly", logger)
    try:
        scores_before = _assemble_scores_from_batch(
            orig_full, orig_paras, doc_scores_gz, doc_scores_sp,
            para_scores_gz, para_scores_sp
        )
    except Exception:
        scores_before = {}

    runs = []
    for spec in drafts:
        hum_text  = spec["humanized_text"]
        hum_paras = spec["humanized_paras_resolved"]
        
        # Enhanced mismatch detection: check both count and structure
        mismatch, mismatch_reason = _detect_paragraph_mismatch(para_objs, hum_paras)
        
        if mismatch:
            _maybe_log(f"⚠️ Paragraph mismatch detected: {mismatch_reason}", logger)

        # Calculate draft-level length deviation for all cases
        wc_after = sum(len(p.split()) for p in hum_paras)
        draft_length_deviation = ((wc_after - wc_before) / wc_before) * 100 if wc_before > 0 else 0

        # Try to create paragraph analysis even with document-level mismatches
        try:
            if not mismatch:
                # Normal case: no document-level mismatch
                scores_after = _assemble_scores_from_batch(
                    hum_text, hum_paras, doc_scores_gz, doc_scores_sp,
                    para_scores_gz, para_scores_sp
                )
                para_details, flag_counts = _assemble_per_para_stats(
                    orig_paras, hum_paras,
                    scores_before["ind_par"], scores_after["ind_par"],
                    q_results, para_objs,
                    spec.get("para_pair_info")  # Pass paragraph mismatch info
                )
            elif spec.get("mode") == "para" and spec.get("para_pair_info"):
                # Para mode with document-level mismatch: use para_pair_info for analysis
                base = {
                    "gptzero": doc_scores_gz.get(_hash(hum_text)),
                    "sapling": doc_scores_sp.get(_hash(hum_text))
                }
                scores_after = {
                    "group_doc": base, "ind_doc": base,
                    "group_par": {"gptzero": [], "sapling": []},
                    "ind_par":   {"gptzero": [], "sapling": []}
                }
                
                # Create paragraph details using para_pair_info
                para_details, flag_counts = _assemble_per_para_stats_from_pairs(
                    scores_before["ind_par"], scores_after["ind_par"],
                    q_results, para_objs,
                    spec.get("para_pair_info"),
                    para_scores_gz, para_scores_sp
                )
            else:
                # Other mismatch cases: no detailed analysis available
                base = {
                    "gptzero": doc_scores_gz.get(_hash(hum_text)),
                    "sapling": doc_scores_sp.get(_hash(hum_text))
                }
                scores_after = {
                    "group_doc": base, "ind_doc": base,
                    "group_par": {"gptzero": [], "sapling": []},
                    "ind_par":   {"gptzero": [], "sapling": []}
                }
                para_details, flag_counts = [], {}
        except Exception as exc:
            _maybe_log(f"❌ per-para assembly error: {exc}", logger)
            base = {
                "gptzero": doc_scores_gz.get(_hash(hum_text)),
                "sapling": doc_scores_sp.get(_hash(hum_text))
            }
            scores_after = {
                "group_doc": base, "ind_doc": base,
                "group_par": {"gptzero": [], "sapling": []},
                "ind_par":   {"gptzero": [], "sapling": []}
            }
            para_details, flag_counts = [], {}
        
        runs.append(_pack_run(
            spec["model"], spec["mode"], spec["iter"],
            scores_before, scores_after,
            wc_before, wc_after,
            flag_counts, para_details, mismatch, hum_text,
            len(orig_paras), len(hum_paras), draft_length_deviation,
            mismatch_reason if mismatch else None
        ))

    _stage("run_test COMPLETE", logger)
    return {
        "document": doc_path.name,
        "folder": doc_path.parent.name,
        "runs": runs,
        "paragraph_count": len(orig_paras)
    }

# ═══════════════ 11 · Packer ══════════════════════════════════════
def _pack_run(model: str, mode: str, it: int,
              scores_before: Dict, scores_after: Dict,
              wc_before: int, wc_after: int,
              flag_counts: Dict[str, int], para_details: List[Dict],
              para_mismatch: bool, humanized_text: str,
              para_count_before: int, para_count_after: int,
              draft_length_deviation: float,
              mismatch_reason: str = None):
    return {
        "model": model, "mode": mode, "iter": it,
        "scores_before": scores_before, "scores_after": scores_after,
        "wordcount_before": wc_before, "wordcount_after": wc_after,
        "flag_counts": flag_counts, "paragraph_details": para_details,
        "para_mismatch": para_mismatch, "humanized_text": humanized_text,
        "para_count_before": para_count_before, "para_count_after": para_count_after,
        "draft_length_deviation": draft_length_deviation,
        "mismatch_reason": mismatch_reason,
    }

# ═══════════════ 12 · Sequential loader ════════════════════════════
def load_ai_scores(doc_path: Path, log: Callable[[str], None] | None = None, max_retries: int = 3):
    """Load AI scores for a single document (used by browser) with retry logic."""
    para_objs = extract_paragraphs_with_type(doc_path)
    segs = [p["text"] for p in para_objs]
    full_text = "\n\n".join(segs)
    _maybe_log(f"Detector scores for {doc_path.name}", log)

    scores = None
    for attempt in range(1, max_retries + 1):
        try:
            if attempt > 1:
                _maybe_log(f"🔄 Retrying detector scoring (attempt {attempt}/{max_retries})", log)
                time.sleep(min(30 * (attempt - 1), 120))

            # For original documents, try cache first (skip_cache=False)
            scores = _detect_both(full_text, segs, skip_cache=False, log=log)
            break

        except Exception as exc:
            _maybe_log(f"❌ Detector error (attempt {attempt}): {exc}", log)
            if attempt == max_retries:
                raise Exception(f"Failed to get detector scores after {max_retries} attempts: {exc}")

    doc_scores_gz = {_hash(full_text): scores["g_doc"]}
    doc_scores_sp = {_hash(full_text): scores["s_doc"]}
    para_scores_gz = {_hash(p): s for p, s in zip(segs, scores["g_par"])}
    para_scores_sp = {_hash(p): s for p, s in zip(segs, scores["s_par"])}

    assembled = _assemble_scores_from_batch(
        full_text, segs, doc_scores_gz, doc_scores_sp, para_scores_gz, para_scores_sp
    )

    return {
        "document": doc_path.name,
        "segments": segs,
        "overall": assembled["group_doc"],
        "group_par": assembled["group_par"],
        "ind_par": assembled["ind_par"],
    }