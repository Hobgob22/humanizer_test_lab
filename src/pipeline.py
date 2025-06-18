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
    OPENAI_API_KEY, HUMANIZER_OPENAI_API_KEY,
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
    if not skip_cache:
        cached = sapling.get("sapling", text)
        if cached is not None:
            _maybe_log("Sapling: ✨ cache hit — scores retrieved", log)
            raw = cached
        else:
            _maybe_log("Sapling: 🔄 cache miss — computing scores", log)
            raw = sapling.detect_ai(text)
    else:
        _maybe_log("Sapling: 🔄 cache miss — computing scores", log)
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
        _stage("✓ Detector scoring complete (all cached)", log)
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

def _humanize_paragraphs(paragraphs: List[str], model: str, log=None) -> List[str]:
    """
    Paragraph-wise humanisation with a single executor.
    """
    _stage(f"Para humanization START • {model} • {len(paragraphs)} paragraphs", log)
    start_time = time.time()
    if not paragraphs:
        return []

    max_workers = min(HUMANIZER_MAX_WORKERS, len(paragraphs))
    out = [None] * len(paragraphs)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        fut2idx = {
            pool.submit(humanize, p, model, "para"): i
            for i, p in enumerate(paragraphs)
        }
        completed = 0
        try:
            for fut in as_completed(fut2idx):
                idx = fut2idx[fut]
                out[idx] = fut.result()
                completed += 1
                if completed % 5 == 0 or completed == len(paragraphs):
                    _maybe_log(f"Para progress: {completed}/{len(paragraphs)} • {model}", log)
        except KeyboardInterrupt:
            pool.shutdown(wait=False, cancel_futures=True)
            raise

    elapsed = time.time() - start_time
    _stage(f"Para humanization DONE • {model} • {elapsed:.1f}s total", log)
    return out

# ═══════════════ 7 · Paragraph helper ════════════════════════════════
def _merge_heading_content(para_objs, hum_content):
    out, idx = [], 0
    for p in para_objs:
        if p["type"] == "content":
            out.append(hum_content[idx]); idx += 1
        else:
            out.append(p["text"])
    return out

# ═══════════════ 8 · Draft generator ════════════════════════════════
def _generate_single_draft(model: str, iteration: int, orig_text: str, para_objs, log=None):
    _stage(f"Starting draft generation • model={model} • iter={iteration+1}", log)
    content_paras = [p["text"] for p in para_objs if p["type"] == "content"]
    _maybe_log(f"Found {len(content_paras)} content paragraphs", log)

    with _fast_pool(max_workers=2) as pool:
        fut_doc  = pool.submit(_humanize_doc, orig_text, model, log)
        fut_para = pool.submit(_humanize_paragraphs, content_paras, model, log)

        hum_doc = fut_doc.result()
        doc_paras = [p.strip() for p in hum_doc.splitlines() if p.strip()]
        _maybe_log(f"Doc-mode complete • {model} • {len(doc_paras)} paragraphs", log)

        hum_para_content = fut_para.result()
        hum_para_paras   = _merge_heading_content(para_objs, hum_para_content)
        _maybe_log(f"Para-mode complete • {model} • {len(hum_para_paras)} paragraphs", log)

    _stage(f"✓ Draft pair complete • model={model} • iter={iteration+1}", log)
    return [{
        "model": model, "mode": "doc", "iter": iteration,
        "humanized_text": hum_doc,
        "humanized_paras_resolved": doc_paras,
    }, {
        "model": model, "mode": "para", "iter": iteration,
        "humanized_paras": hum_para_paras,
        "humanized_paras_resolved": hum_para_paras,
        "humanized_text": "\n\n".join(hum_para_paras),
    }]

def _generate_all_drafts(models, iterations, orig_text, para_objs, log=None):
    out: List[Dict] = []
    total_tasks = len(models) * iterations
    max_workers = min(HUMANIZER_MAX_WORKERS, total_tasks)

    _stage(f"Generating {total_tasks} draft pairs", log)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        fut_to_info = {}
        for m in models:
            for i in range(iterations):
                fut = pool.submit(_generate_single_draft, m, i, orig_text, para_objs, log)
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
):
    flags_total = {k: 0 for k in _EXPECTED_FLAGS}
    details: List[Dict] = []

    for idx, (o, h) in enumerate(zip(orig, hum)):
        raw = quality_results.get((_hash(o), _hash(h)), {})
        p_flags = {k: bool(raw.get(k, False)) for k in _EXPECTED_FLAGS}
        for k, v in p_flags.items():
            if v: flags_total[k] += 1
        details.append({
            "paragraph": idx + 1,
            "wc_before": len(o.split()), "wc_after": len(h.split()),
            "ai_before": {d: ai_before.get(d, [None]*len(orig))[idx] for d in ("gptzero","sapling")},
            "ai_after":  {d: ai_after.get(d, [None]*len(orig))[idx] for d in ("gptzero","sapling")},
            "flags": p_flags,
        })
    return details, flags_total

# ═══════════════ 10 · Main runner ════════════════════════════════
def run_test(doc_path: Path, models: List[str]|None=None,
             logger: Callable[[str],None]|None=None,
             iterations: int = REHUMANIZE_N,
             max_retries: int = 3):
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


    # Phase 1: Generation (with retries)
    _stage("Phase 1: Generation", logger)
    drafts = None
    for attempt in range(1, max_retries + 1):
        try:
            if attempt > 1:
                _maybe_log(f"🔄 Retrying Phase 1 (attempt {attempt}/{max_retries})", logger)
                time.sleep(min(30 * (attempt - 1), 120))  # exponential backoff: 30s, 60s, 120s
            
            drafts = _generate_all_drafts(models, iterations, orig_full, para_objs, logger)
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
    q_pairs = {
        (o, h) for d in drafts
        if len(orig_paras) == len(d["humanized_paras_resolved"])
        for o, h in zip(orig_paras, d["humanized_paras_resolved"])
    }
    
    q_results = {}
    for attempt in range(1, max_retries + 1):
        try:
            if attempt > 1:
                _maybe_log(f"🔄 Retrying Phase 3 (attempt {attempt}/{max_retries})", logger)
                time.sleep(min(30 * (attempt - 1), 120))
                
            q_results = _batch_quality_check(list(q_pairs), logger)
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
        mismatch  = len(hum_paras) != len(orig_paras)

        if mismatch:
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
        else:
            try:
                scores_after = _assemble_scores_from_batch(
                    hum_text, hum_paras, doc_scores_gz, doc_scores_sp,
                    para_scores_gz, para_scores_sp
                )
                para_details, flag_counts = _assemble_per_para_stats(
                    orig_paras, hum_paras,
                    scores_before["ind_par"], scores_after["ind_par"],
                    q_results
                )
            except Exception as exc:
                _maybe_log(f"❌ per-para assembly error: {exc}", logger)
                scores_after, para_details, flag_counts = {}, [], {}

        runs.append(_pack_run(
            spec["model"], spec["mode"], spec["iter"],
            scores_before, scores_after,
            wc_before, sum(len(p.split()) for p in hum_paras),
            flag_counts, para_details, mismatch, hum_text,
            len(orig_paras), len(hum_paras)
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
              para_count_before: int, para_count_after: int):
    return {
        "model": model, "mode": mode, "iter": it,
        "scores_before": scores_before, "scores_after": scores_after,
        "wordcount_before": wc_before, "wordcount_after": wc_after,
        "flag_counts": flag_counts, "paragraph_details": para_details,
        "para_mismatch": para_mismatch, "humanized_text": humanized_text,
        "para_count_before": para_count_before, "para_count_after": para_count_after,
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

            # ----- fixed: pass log as keyword so signature matches -----
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