# src/pages/new_run.py
# v7.2 – final bug-free per-folder & equal-count sliders
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List

import streamlit as st

from src.pages.utils import log, show_log
from src.pipeline import run_test
from src.results_db import save_run, load_run
from src.models import MODEL_REGISTRY

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
        default_val = st.session_state.get("equal_count", min_available)

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
        # Initialise once with “all” if not yet present
        if key not in st.session_state:
            st.session_state[key] = max_docs

        counts[lbl] = st.slider(
            f"{lbl} – documents to include",
            min_value=1,
            max_value=max_docs,
            value=min(st.session_state[key], max_docs),
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
        out.extend(sorted(folder.glob("*.docx"))[: n])
    return out


# ═════════════════════════════════ PAGE ════════════════════════════════
def page_new_run():
    st.header("⚡️ Launch new benchmark")

    with st.expander("ℹ️ About Benchmarking", expanded=False):
        st.markdown(
            """
            **What this does**

            * Tests multiple humanizer models on your documents  
            * Measures AI-detection scores **before & after** humanization  
            * Evaluates content-quality preservation  
            * Runs several iterations for robustness
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
        "Mixed texts": "data/mixed_texts",
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

    # ── 4 · iteration count ─────────────────────────────────────────
    iterations = st.slider(
        "Iterations per document",
        1,
        10,
        value=5,
        help="How many drafts each model should generate for every document",
    )

    # ── 5 · workload preview ───────────────────────────────────────
    if model_labels:
        docs = _gather_docs(doc_counts, FOLDERS)
        total_drafts = len(docs) * len(model_labels) * iterations * 2
        st.info(
            f"📊 **Workload preview:** {len(docs)} docs × "
            f"{len(model_labels)} models × {iterations} iterations × 2 modes "
            f"= **{total_drafts} drafts**"
        )

    # ── 6 · live log placeholder ───────────────────────────────────
    log_box = st.empty()

    # ── 7 · RUN button ─────────────────────────────────────────────
    if st.button(
        "🚀 Run benchmark",
        type="primary",
        disabled=not (run_name.strip() and folder_labels and model_labels),
    ):
        if load_run(run_name):
            st.error("Run name already exists")
            st.stop()

        docs = _gather_docs(doc_counts, FOLDERS)
        if not docs:
            st.error("No .docx files found for the current settings")
            st.stop()

        # ── initial log ────────────────────────────────────────────
        log(f"🚀 Benchmark **{run_name}**")
        log(
            "📂 Folders: "
            + ", ".join(f"{f}({doc_counts[f]})" for f in folder_labels)
        )
        log(f"🤖 Models:  {', '.join(model_labels)}")
        log(f"🔁 Iterations: {iterations}")
        log(f"📝 Total drafts: {len(docs)*len(model_labels)*iterations*2}")
        show_log(log_box)

        # ── processing loop ────────────────────────────────────────
        with st.status("Running benchmark …", expanded=True) as status:
            t0 = time.time()
            results = []

            for idx, doc_path in enumerate(docs, 1):
                status.update(label=f"{idx}/{len(docs)} – {doc_path.name}")
                st.progress(idx / len(docs))
                log(f"\n📄 ({idx}/{len(docs)}) {doc_path.name}")
                show_log(log_box)

                try:
                    res = run_test(doc_path, model_labels, log, iterations)
                    if res.get("runs"):
                        results.append(res)
                        log(f"✅ Completed {doc_path.name} – {len(res['runs'])} drafts")
                    else:
                        log(f"⚠️  Skipped {doc_path.name} (no paragraphs)")
                except Exception as e:
                    log(f"❌ ERROR in {doc_path.name}: {e}")

                show_log(log_box)

            # ── save run ───────────────────────────────────────────
            duration = (time.time() - t0) / 60
            log(f"\n💾 Saving run **{run_name}** (took {duration:.1f} min)")
            save_run(
                run_name,
                folder_labels,
                model_labels,
                {"docs": results, "iterations": iterations, "doc_counts": doc_counts},
            )
            show_log(log_box)
            status.update("Benchmark finished!", state="complete", expanded=False)

        # ── summary ───────────────────────────────────────────────
        st.success(f"✅ Run **{run_name}** completed in {duration:.1f} minutes")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📄 Documents", len(results))
        with col2:
            st.metric("📝 Drafts", sum(len(d["runs"]) for d in results))
        with col3:
            st.metric("🤖 Models", len(model_labels))
        with col4:
            avg_sec = (duration * 60) / len(results) if results else 0
            st.metric("⏱️ Avg time/doc", f"{avg_sec:.1f}s")
