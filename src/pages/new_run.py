# src/pages/new_run.py
# v8.0 – Background job processing with status monitoring
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List

import streamlit as st

# ─────────────────── project root ────────────────────
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pages.utils import natural_key
from src.models import MODEL_REGISTRY
from src.api_client import get_client, cached_list_jobs
from src.job_manager import JobStatus  # Keep enum for status checking
from src.humanizers import humanizer as _humanizer
import src.prompts as _prompts

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
    try:
        jobs = cached_list_jobs(status="active", limit=20)
    except Exception as e:
        st.warning(f"Could not load jobs: {e}")
        return
    
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
                    try:
                        get_client().cancel_job(job['job_id'])
                        st.success("Job cancelled")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to cancel job: {e}")
    
    st.divider()


def _show_job_monitor(job_id: str):
    """Show detailed monitoring for a specific job."""
    try:
        job = get_client().get_job(job_id)
    except Exception as e:
        st.error(f"Could not load job: {e}")
        return
    
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
                try:
                    get_client().cancel_job(job_id)
                    st.success("Job cancelled")
                    # Remove sleep - rerun immediately
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to cancel job: {e}")
    
    # Current document
    if job['current_doc'] and job['status'] == JobStatus.RUNNING.value:
        st.info(f"🔄 Currently processing: **{job['current_doc']}**")
    
    # Error display
    if job['error']:
        st.error(f"Error: {job['error']}")
    
    # Logs
    try:
        logs_data = get_client().get_job_logs(job_id, limit=20)
        logs = logs_data.get('logs', [])
        if logs:
            with st.expander("📜 Job Logs", expanded=True):
                # Show last 20 logs
                for log_entry in logs[-20:]:
                    timestamp = time.strftime('%H:%M:%S', time.localtime(log_entry['timestamp']))
                    st.text(f"[{timestamp}] {log_entry['message']}")
    except Exception as e:
        # Fallback to stored logs if API fails
        if job.get('logs'):
            import json
            try:
                logs = json.loads(job['logs'])
                if logs:
                    with st.expander("📜 Job Logs", expanded=True):
                        for log_entry in logs[-20:]:
                            timestamp = time.strftime('%H:%M:%S', time.localtime(log_entry['timestamp']))
                            st.text(f"[{timestamp}] {log_entry['message']}")
            except:
                pass
    
    # Auto-refresh for active jobs - use Streamlit's built-in refresh
    if job['status'] in (JobStatus.PENDING.value, JobStatus.RUNNING.value):
        # Use placeholder to avoid full page rerun
        if "last_refresh" not in st.session_state:
            st.session_state.last_refresh = time.time()
        
        # Refresh every 3 seconds
        if time.time() - st.session_state.last_refresh > 3:
            st.session_state.last_refresh = time.time()
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
        
        if st.button("← Back to New Run", key="back_to_new_run"):
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

    # ── 3d · Detector selection options (NEW) ──────────────────────────
    st.markdown("### 🔍 AI Detector Selection")
    col1, col2 = st.columns(2)
    
    with col1:
        use_gptzero = st.checkbox(
            "🟠 GPTZero",
            value=True,
            help="Include GPTZero AI detection scoring",
            key="use_gptzero"
        )
    
    with col2:
        use_sapling = st.checkbox(
            "🟢 Sapling",
            value=True,
            help="Include Sapling AI detection scoring",
            key="use_sapling"
        )
    
    if not use_gptzero and not use_sapling:
        st.error("⚠️ At least one detector must be selected!")
        st.stop()
    
    # Show selected detectors info
    selected_detectors = []
    if use_gptzero:
        selected_detectors.append("GPTZero")
    if use_sapling:
        selected_detectors.append("Sapling")
    
    st.info(f"📊 Selected detectors: {', '.join(selected_detectors)}")
    
    if not use_gptzero:
        st.warning("⚠️ GPTZero results will be empty in the analysis")
    if not use_sapling:
        st.warning("⚠️ Sapling results will be empty in the analysis")

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
        print("=" * 80, flush=True)
        print("[START] [STREAMLIT] START JOB BUTTON CLICKED!", flush=True)
        print("=" * 80, flush=True)
        
        # Check if run name exists via API
        try:
            print(f"[STREAMLIT] Checking for existing run: {run_name}", flush=True)
            existing_run = get_client().get_run_summary(run_name)
            if existing_run:
                print(f"[STREAMLIT] Run name already exists: {run_name}", flush=True)
                st.error("Run name already exists")
                st.stop()
            print("[STREAMLIT] No existing run found - proceeding", flush=True)
        except Exception as e:
            print(f"[STREAMLIT] Exception checking run (expected if new): {e}", flush=True)
            pass

        print(f"[STREAMLIT] Gathering documents for: {doc_counts}", flush=True)
        docs = _gather_docs(doc_counts, FOLDERS)
        if not docs:
            print("[STREAMLIT] ERROR: No .docx files found", flush=True)
            st.error("No .docx files found for the current settings")
            st.stop()

        print(f"[STREAMLIT] Found {len(docs)} documents", flush=True)
        
        # Prepare job data
        total_docs = len(docs)
        
        # Start background job via API
        with st.spinner("Starting background job..."):
            try:
                print("[STREAMLIT] Setting prompt overrides...", flush=True)
                _humanizer.set_prompt_overrides(prompt_overrides)
                
                job_data = {
                    "run_name": run_name,
                    "folders": folder_labels,
                    "models": model_labels,
                    "iterations": iterations,
                    "doc_counts": doc_counts,
                    "total_docs": total_docs,
                    "include_doc_mode": include_doc_mode,
                    "use_gptzero": use_gptzero,
                    "use_sapling": use_sapling
                }
                
                print(f"[STREAMLIT] Calling create_job API with data: {job_data}", flush=True)
                job_response = get_client().create_job(job_data)
                print(f"[STREAMLIT] API response received: {job_response}", flush=True)
                job_id = job_response['job_id']
                
                print(f"[STREAMLIT] [OK] Job started successfully! ID: {job_id}", flush=True)
                st.success(f"[OK] Job started! ID: {job_id}")
                st.info("The job is running in the background. You can navigate to other pages or close this tab.")
                
                # Set monitoring flag
                print(f"[STREAMLIT] Setting monitoring_job in session_state: {job_id}", flush=True)
                st.session_state.monitoring_job = job_id
                # Remove sleep - rerun immediately
                print("[STREAMLIT] Calling st.rerun()...", flush=True)
                st.rerun()
            except Exception as e:
                print(f"[STREAMLIT] [ERROR] Exception: {type(e).__name__}: {e}", flush=True)
                import traceback
                print(traceback.format_exc(), flush=True)
                st.error(f"Failed to start job: {e}")
                st.exception(e)

# ──────────────────────────── Standalone Page Setup ────────────────────
# When this file is executed directly by Streamlit's multi-page system,
# set up the page config and sidebar, then call the page function
# Check if we're being run as a standalone page (not imported)
if __name__ == "__main__":
    # Page config
    st.set_page_config(page_title="New Run - Humanizer Test-Bench", layout="wide", initial_sidebar_state="expanded")
    
    # Setup shared sidebar
    from src.pages._shared_layout import setup_sidebar
    setup_sidebar()
    
    # Call the page function
    page_new_run()