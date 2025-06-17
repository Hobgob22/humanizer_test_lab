# src/pages/new_run.py - New benchmark run functionality
from __future__ import annotations

import time
from pathlib import Path

import streamlit as st

from .utils import log, show_log
from results_db import load_run, save_run
from pipeline import run_test
from models import MODEL_REGISTRY


# ─────────────────── project root ────────────────────
ROOT = Path(__file__).resolve().parents[2]

# ═════════════════════════════ NEW RUN PAGE ═══════════════════════════
def page_new_run():
    st.header("⚡️ Launch new benchmark")
    
    with st.expander("ℹ️ About Benchmarking", expanded=False):
        st.markdown("""
        **What this does:**
        - Tests multiple humanizer models on your documents
        - Measures AI detection scores before and after humanization
        - Evaluates quality preservation (meaning, citations, etc.)
        - Runs multiple iterations to ensure consistent results
        
        **Key metrics:**
        - **AI Detection Scores**: Lower is better (0-1 scale)
        - **Zero-shot Success**: % of drafts scoring below 10% AI detection
        - **Quality Flags**: Checks if humanization preserves content integrity
        """)

    run_name = st.text_input("Unique run name", placeholder="Enter a descriptive name for this benchmark run")

    FOLDERS = {
        "AI texts":    "data/ai_texts",
        "Human texts": "data/human_texts",
        "Mixed texts": "data/mixed_texts",
    }
    folder_labels = st.multiselect("Folders", list(FOLDERS), help="Select which document folders to process")
    folder_paths  = [FOLDERS[f] for f in folder_labels]

    all_models   = list(MODEL_REGISTRY)
    model_labels = st.multiselect("Models", all_models, default=all_models[:3], 
                                   help="Select humanizer models to test")

    iterations = st.slider("Iterations per document", 1, 10, value=5,
                          help="Number of times to humanize each document with each model")

    # Show expected workload
    if folder_paths and model_labels:
        doc_count = sum(len(list((ROOT / fp).glob("*.docx"))) for fp in folder_paths)
        total_drafts = doc_count * len(model_labels) * iterations * 2  # x2 for doc + para modes
        st.info(f"📊 Expected workload: {doc_count} documents × {len(model_labels)} models × {iterations} iterations × 2 modes = **{total_drafts} drafts**")

    # Live-log placeholder
    log_box = st.empty()

    if st.button("🚀 Run benchmark", type="primary", disabled=not (run_name and folder_paths and model_labels)):
        if not run_name.strip():
            st.error("Please provide a run name"); st.stop()
        if load_run(run_name):
            st.error("Run name already exists"); st.stop()
        if not folder_paths or not model_labels:
            st.error("Pick at least one folder and one model"); st.stop()

        docs = [p for fp in folder_paths for p in (ROOT / fp).glob("*.docx")]
        if not docs:
            st.error("No .docx files in the selected folders"); st.stop()

        # initial metadata logs
        log(f"🚀 Benchmark '{run_name}': {len(docs)} docs, {len(model_labels)} models, {iterations} iters")
        log(f"🎯 Total drafts: {len(docs) * len(model_labels) * iterations * 2}")
        log(f"🗂  Folders: {', '.join(folder_labels)}")
        log(f"🤖 Models:   {', '.join(model_labels)}")
        show_log(log_box)

        # status spinner with live updates
        with st.status("Running benchmark …", expanded=True) as status:
            start_time = time.time()
            results = []

            for idx, doc_path in enumerate(docs, start=1):
                status.update(label=f"Processing document {idx}/{len(docs)}: {doc_path.name}")
                st.progress(idx / len(docs))
                log(f"\n📄 Starting document {idx}/{len(docs)}: {doc_path.name}")
                show_log(log_box)

                try:
                    result = run_test(doc_path, model_labels, log, iterations)
                    if result.get("runs"):
                        results.append(result)
                        log(f"✅ Completed {doc_path.name}: {len(result['runs'])} drafts")
                    else:
                        log(f"⚠️  Skipped {doc_path.name} (no paragraphs)")
                except Exception as e:
                    log(f"❌ ERROR in {doc_path.name}: {e}")

                show_log(log_box)

            # Save
            duration = (time.time() - start_time) / 60
            log(f"\n💾 Saving run '{run_name}' (took {duration:.1f} min)…")
            save_run(run_name, folder_labels, model_labels, {"docs": results, "iterations": iterations})
            show_log(log_box)
            status.update(label="Benchmark finished!", state="complete", expanded=False)

        # Final summary
        st.success(f"✅ Run '{run_name}' completed in {duration:.1f} minutes")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📄 Documents", len(results))
        with col2:
            st.metric("📝 Total drafts", sum(len(d['runs']) for d in results))
        with col3:
            st.metric("🤖 Models used", len(model_labels))
        with col4:
            avg_sec = (duration * 60) / len(results) if results else 0
            st.metric("⏱️ Avg time/doc", f"{avg_sec:.1f}s")