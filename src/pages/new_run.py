# src/pages/new_run.py
# v8.0 – Background job processing with status monitoring
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List

import streamlit as st

from src.pages.utils import natural_key
from src.results_db import load_run
from src.models import MODEL_REGISTRY
from src.job_manager import (
    start_benchmark_job, get_job, get_active_jobs, 
    cancel_job, JobStatus
)
from src.humanizers import humanizer as _humanizer
import src.prompts as _prompts


# ─────────────────── project root ────────────────────
ROOT = Path(__file__).resolve().parents[2]

# ═════════════════════════════════ helpers ═════════════════════════════
def _folder_doc_counts(paths: Dict[str, str]) -> Dict[str, int]:
    """Return {folder-label: number_of_docx}."""
    return {
        label: len(list((ROOT / path).glob("*.docx")))
        for label, path in paths.items()
    }


def _select_documents(
    folders: List[str],
    limits: Dict[str, int],
) -> Dict[str, int]:
    """
    Render sliders and return {folder: docs_to_include}.
    Order: ① global equal-count (if applicable) ➜ ② per-folder sliders.
    """
    counts: Dict[str, int] = {}

    # Store for possible later use
    st.session_state["selected_folders"] = folders

    # ── 1 · global equal-count (only when 2+ folders) ────────────────
    if len(folders) >= 2:
        min_available = min(limits[f] for f in folders)
        default_val = min(100, st.session_state.get("equal_count", min_available))

        equal_val = st.slider(
            "🔄 Equal docs for *all* selected folders",
            min_value=1,
            max_value=min_available,
            value=default_val,
            key="equal_count",
            help="Drag to assign the same document count to every folder",
        )

        # Detect change since last run *before* per-folder sliders exist
        prev = st.session_state.get("prev_equal_val")
        if prev is None or prev != equal_val:
            st.session_state["prev_equal_val"] = equal_val
            # Push into per-folder counts (safe – widgets not created yet)
            for lbl in folders:
                st.session_state[f"count_{lbl}"] = equal_val

    # ── 2 · per-folder sliders ───────────────────────────────────────
    st.subheader("📂 Documents per folder")

    for lbl in folders:
        max_docs = limits[lbl]
        key = f"count_{lbl}"
        # Initialise once with "all" if not yet present
        if key not in st.session_state:
            st.session_state[key] = max_docs

        counts[lbl] = st.slider(
            f"{lbl} – documents to include",
            min_value=1,
            max_value=max_docs,
            value=min(100, st.session_state[key], max_docs),
            key=key,
            help=f"{max_docs} .docx files available",
        )

    st.divider()
    return counts


def _gather_docs(selected: Dict[str, int], paths: Dict[str, str]) -> List[Path]:
    """Return actual Path list based on selected counts."""
    out: List[Path] = []
    for lbl, n in selected.items():
        folder = ROOT / paths[lbl]
        out.extend(sorted(folder.glob("*.docx"), key=natural_key)[: n])
    return out


def _show_active_jobs():
    """Display active jobs with status and controls."""
    jobs = get_active_jobs()
    if not jobs:
        return
    
    st.subheader("🔄 Active Jobs")
    
    for job in jobs:
        with st.container():
            col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
            
            with col1:
                st.write(f"**{job['run_name']}**")
                if job['current_doc']:
                    st.caption(f"Processing: {job['current_doc']}")
            
            with col2:
                progress = job['processed_docs'] / job['total_docs'] if job['total_docs'] > 0 else 0
                st.progress(progress)
                st.caption(f"{job['processed_docs']}/{job['total_docs']} docs")
            
            with col3:
                if job['status'] == JobStatus.RUNNING.value:
                    elapsed = time.time() - job['started_at'] if job['started_at'] else 0
                    st.caption(f"⏱️ {elapsed/60:.1f} min")
                else:
                    st.caption(f"Status: {job['status']}")
            
            with col4:
                if st.button("❌", key=f"cancel_{job['job_id']}", 
                           help="Cancel this job"):
                    if cancel_job(job['job_id']):
                        st.rerun()
    
    st.divider()


def _show_job_monitor(job_id: str):
    """Show detailed monitoring for a specific job."""
    job = get_job(job_id)
    if not job:
        st.error("Job not found!")
        return
    
    st.subheader(f"📊 Monitoring: {job['run_name']}")
    
    # Status overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status_emoji = {
            JobStatus.PENDING.value: "⏳",
            JobStatus.RUNNING.value: "🔄",
            JobStatus.COMPLETED.value: "✅",
            JobStatus.FAILED.value: "❌",
            JobStatus.CANCELLED.value: "🚫"
        }.get(job['status'], "❓")
        st.metric("Status", f"{status_emoji} {job['status'].title()}")
    
    with col2:
        progress = job['processed_docs'] / job['total_docs'] if job['total_docs'] > 0 else 0
        st.metric("Progress", f"{job['processed_docs']}/{job['total_docs']}")
        st.progress(progress)
    
    with col3:
        if job['started_at']:
            if job['completed_at']:
                duration = (job['completed_at'] - job['started_at']) / 60
                st.metric("Duration", f"{duration:.1f} min")
            else:
                elapsed = (time.time() - job['started_at']) / 60
                st.metric("Elapsed", f"{elapsed:.1f} min")
        else:
            st.metric("Duration", "Not started")
    
    with col4:
        if job['status'] in (JobStatus.PENDING.value, JobStatus.RUNNING.value):
            if st.button("Cancel Job", type="secondary"):
                if cancel_job(job_id):
                    st.success("Job cancelled")
                    time.sleep(1)
                    st.rerun()
    
    # Current document
    if job['current_doc'] and job['status'] == JobStatus.RUNNING.value:
        st.info(f"🔄 Currently processing: **{job['current_doc']}**")
    
    # Error display
    if job['error']:
        st.error(f"Error: {job['error']}")
    
    # Logs
    if job['logs']:
        import json
        logs = json.loads(job['logs'])
        if logs:
            with st.expander("📜 Job Logs", expanded=True):
                # Show last 20 logs
                for log_entry in logs[-20:]:
                    timestamp = time.strftime('%H:%M:%S', time.localtime(log_entry['timestamp']))
                    st.text(f"[{timestamp}] {log_entry['message']}")
    
    # Auto-refresh for active jobs
    if job['status'] in (JobStatus.PENDING.value, JobStatus.RUNNING.value):
        time.sleep(2)  # Wait 2 seconds before refresh
        st.rerun()
    
    # Show completion message
    if job['status'] == JobStatus.COMPLETED.value:
        st.success(f"✅ Benchmark completed successfully!")
        st.balloons()
        
        # Show button to go to analysis
        if st.button("📊 View Results", type="primary"):
            st.session_state.page = "Benchmark Analysis"
            st.rerun()


# ═════════════════════════════════ PAGE ════════════════════════════════
def page_new_run():
    st.header("⚡️ Launch new benchmark")

    # Check if we're monitoring a job
    if "monitoring_job" in st.session_state:
        _show_job_monitor(st.session_state.monitoring_job)
        
        if st.button("← Back to New Run"):
            del st.session_state.monitoring_job
            st.rerun()
        
        return
    
    # Show active jobs
    _show_active_jobs()

    with st.expander("ℹ️ About Benchmarking", expanded=False):
        st.markdown(
            """
            **What this does**

            * Tests multiple humanizer models on your documents  
            * Measures AI-detection scores **before & after** humanization  
            * Evaluates content-quality preservation  
            * Runs several iterations for robustness
            
            **Background Processing**
            
            * Jobs run in the background - you can navigate away safely
            * Progress is saved to database and persists across reloads
            * Multiple jobs can run concurrently
            * Jobs can be cancelled at any time
            """
        )

    # ── 1 · run name ────────────────────────────────────────────────
    run_name = st.text_input(
        "Unique run name",
        placeholder="Enter a descriptive name for this benchmark run",
    )

    # ── 2 · folder selection ───────────────────────────────────────
    FOLDERS = {
        "AI texts": "data/ai_texts",
        "Human texts": "data/human_texts",
        "AI paragraphs": "data/ai_paras",
        "Human paragraphs": "data/human_paras",
    }
    folder_labels = st.multiselect(
        "Folders to include",
        list(FOLDERS),
        help="Pick one or more folders",
    )

    if not folder_labels:
        st.info("Pick at least one folder to continue")
        return

    limits = _folder_doc_counts({f: FOLDERS[f] for f in folder_labels})
    doc_counts = _select_documents(folder_labels, limits)

    # ── 3 · model selection ─────────────────────────────────────────
    all_models = list(MODEL_REGISTRY)
    model_labels = st.multiselect(
        "Humanizer models",
        all_models,
        default=all_models[:3],
        help="Select which models you wish to test",
    )

    # ── 3b · prompt variant (fine-tunes only) ───────────────────────
    prompt_overrides: Dict[str, str] = {}

    for m in model_labels:
        meta = MODEL_REGISTRY[m]
        if meta.get("prompt_id") == "finetuned":
            st.markdown(f"**Prompt variant for `{m}`**")
            key = f"variant_{m}"
            variant = st.radio(
                "Choose variant",
                options=["v1", "v2"],
                index=0 if st.session_state.get(key, "v2") == "v1" else 1,
                key=key,
                horizontal=True,
            )
            prompt_overrides[m] = variant

            # Optional prompt preview (read-only)
            with st.expander("Show system prompt", expanded=False):
                preview = (
                    _prompts.FINETUNED_DOC_SYSTEM_PROMPT1
                    if variant == "v1"
                    else _prompts.FINETUNED_DOC_SYSTEM_PROMPT2
                )
                st.code(preview.strip())

    # ── 3c · Document mode option (NEW) ─────────────────────────────
    # Only show this option if regular text folders are selected
    has_regular_folders = any(f in ["AI texts", "Human texts"] for f in folder_labels)
    include_doc_mode = True  # Default value
    
    if has_regular_folders:
        st.markdown("### 📄 Humanization Mode Options")
        include_doc_mode = st.checkbox(
            "Include document-level humanization",
            value=True,
            help="When enabled, documents will be humanized at both document-level and paragraph-level. "
                 "When disabled, only paragraph-level humanization will be performed (paragraphs are "
                 "humanized individually then combined).",
            key="include_doc_mode"
        )
        
        if not include_doc_mode:
            st.info("ℹ️ Only paragraph-level humanization will be performed for AI texts and Human texts folders.")

    # ── 4 · iteration count ─────────────────────────────────────────
    iterations = st.slider(
        "Iterations per document",
        1,
        10,
        value=1,
        help="How many drafts each model should generate for every document",
    )

    # ── 5 · workload preview ───────────────────────────────────────
    if model_labels:
        docs = _gather_docs(doc_counts, FOLDERS)

        # Count docs by type
        para_folder_docs = sum(1 for d in docs if d.parent.name.endswith("_paras"))
        regular_docs = len(docs) - para_folder_docs
        
        # Calculate total drafts based on settings
        total_drafts = 0
        
        # Para folder docs always get 1 mode (para mode)
        total_drafts += para_folder_docs * len(model_labels) * iterations
        
        # Regular docs can have 1 or 2 modes depending on include_doc_mode
        if regular_docs > 0:
            modes_per_regular = 2 if include_doc_mode else 1
            total_drafts += regular_docs * len(model_labels) * iterations * modes_per_regular

        # Display workload preview
        if para_folder_docs and regular_docs:
            modes_desc = "2 modes" if include_doc_mode else "1 mode"
            st.info(
                f"📊 **Workload preview:** "
                f"{regular_docs} docs × {modes_desc} + "
                f"{para_folder_docs} para docs × 1 mode × "
                f"{len(model_labels)} models × {iterations} iterations "
                f"= **{total_drafts} drafts**"
            )
        elif regular_docs:
            modes = 2 if include_doc_mode else 1
            st.info(
                f"📊 **Workload preview:** {regular_docs} docs × "
                f"{len(model_labels)} models × {iterations} iterations × {modes} mode"
                f"{'' if modes==1 else 's'} = **{total_drafts} drafts**"
            )
        else:
            st.info(
                f"📊 **Workload preview:** {para_folder_docs} para docs × "
                f"{len(model_labels)} models × {iterations} iterations × 1 mode "
                f"= **{total_drafts} drafts**"
            )


    # ── 6 · RUN button ─────────────────────────────────────────────
    if st.button(
        "🚀 Start Job",
        type="primary",
        disabled=not (run_name.strip() and folder_labels and model_labels),
        help="Start the benchmark as a background job"
    ):
        if load_run(run_name):
            st.error("Run name already exists")
            st.stop()

        docs = _gather_docs(doc_counts, FOLDERS)
        if not docs:
            st.error("No .docx files found for the current settings")
            st.stop()

        # Start background job
        with st.spinner("Starting background job..."):
            _humanizer.set_prompt_overrides(prompt_overrides)
            job_id = start_benchmark_job(
                run_name=run_name,
                docs=docs,
                folders=folder_labels,
                models=model_labels,
                iterations=iterations,
                doc_counts=doc_counts,
                include_doc_mode=include_doc_mode  # Pass the new parameter
            )
            
            st.success(f"✅ Job started! ID: {job_id}")
            st.info("The job is running in the background. You can navigate to other pages or close this tab.")
            
            # Set monitoring flag
            st.session_state.monitoring_job = job_id
            time.sleep(1)
            st.rerun()