# src/pages/job_status.py
"""
Job status page for monitoring background benchmark jobs.
Shows active jobs, recent completed jobs, and detailed logs.
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

import streamlit as st

# ─────────────────── project root ────────────────────
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api_client import get_client, cached_list_jobs
from src.job_manager import JobStatus, cleanup_old_jobs  # Keep JobStatus enum and cleanup function

def _format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

def _format_timestamp(ts: float) -> str:
    """Format timestamp as readable date/time."""
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")

def _show_job_card(job: dict, show_details: bool = False):
    """Display a job card with status and controls."""
    job_id = job['job_id']
    status = job['status']
    
    # Status emoji and color
    status_config = {
        JobStatus.PENDING.value: ("⏳", "orange"),
        JobStatus.RUNNING.value: ("🔄", "blue"),
        JobStatus.COMPLETED.value: ("✅", "green"),
        JobStatus.FAILED.value: ("❌", "red"),
        JobStatus.CANCELLED.value: ("🚫", "gray")
    }
    emoji, color = status_config.get(status, ("❓", "gray"))
    
    with st.container():
        # Header row
        col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
        
        with col1:
            st.markdown(f"**{emoji} {job['run_name']}**")
            st.caption(f"ID: {job_id}")
        
        with col2:
            # Progress bar for active jobs with current document info
            if status in (JobStatus.PENDING.value, JobStatus.RUNNING.value):
                progress = job['processed_docs'] / job['total_docs'] if job['total_docs'] > 0 else 0
                st.progress(progress)
                
                # Show active documents in compact format
                try:
                    active_docs = json.loads(job.get('active_docs', '[]'))
                except json.JSONDecodeError:
                    active_docs = []
                
                # Fall back to current_doc if active_docs is empty
                if not active_docs and job.get('current_doc'):
                    active_docs = [job['current_doc']]
                
                if active_docs:
                    # Show first active document prominently
                    current_info = active_docs[0]
                    if ' | ' in current_info:
                        doc_name, stage = current_info.split(' | ', 1)
                        # Truncate long document names
                        display_doc = doc_name if len(doc_name) <= 20 else doc_name[:17] + "..."
                        st.caption(f"📄 {display_doc}")
                        
                        # Stage with emoji (updated for actual pipeline phases)
                        stage_emoji = {
                            "Starting": "🚀",
                            "Phase 1: Generation": "✍️",
                            "Phase 2: Detector scoring": "🔍", 
                            "Phase 3: Gemini quality evaluation": "⭐",
                            "Phase 4: Assembly": "🔧",
                            "Completed": "✅",
                            "Failed": "❌",
                            "Error": "💥"
                        }.get(stage, "⚙️")
                        
                        st.caption(f"{stage_emoji} {stage}")
                        
                        # Show count if multiple active documents
                        if len(active_docs) > 1:
                            st.caption(f"+ {len(active_docs) - 1} more active")
                    else:
                        st.caption(f"📄 {current_info}")
                        if len(active_docs) > 1:
                            st.caption(f"+ {len(active_docs) - 1} more active")
                else:
                    st.caption(f"{job['processed_docs']}/{job['total_docs']} docs")
            else:
                st.markdown(f":{color}[{status.title()}]")
        
        with col3:
            # Timing info
            if job['started_at']:
                if job['completed_at']:
                    duration = job['completed_at'] - job['started_at']
                    st.caption(f"⏱️ {_format_duration(duration)}")
                else:
                    elapsed = time.time() - job['started_at']
                    st.caption(f"⏱️ {_format_duration(elapsed)}")
            else:
                created = time.time() - job['created_at']
                st.caption(f"Created {_format_duration(created)} ago")
        
        with col4:
            # Action buttons
            if status in (JobStatus.PENDING.value, JobStatus.RUNNING.value):
                if st.button("❌", key=f"cancel_{job_id}", help="Cancel job"):
                    try:
                        get_client().cancel_job(job_id)
                        st.success("Job cancelled")
                        # Remove sleep - rerun immediately
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to cancel: {e}")
            
            if st.button("📋", key=f"details_{job_id}", help="Show details"):
                st.session_state[f"show_details_{job_id}"] = not st.session_state.get(f"show_details_{job_id}", False)
                # Use st.experimental_rerun for faster updates
                st.rerun()
        
        # Details section
        if show_details or st.session_state.get(f"show_details_{job_id}", False):
            st.divider()
            
            # Job metadata
            meta_col1, meta_col2 = st.columns(2)
            
            with meta_col1:
                st.write("**Configuration:**")
                st.write(f"- Folders: {', '.join(json.loads(job['folders']))}")
                st.write(f"- Models: {', '.join(json.loads(job['models']))}")
                st.write(f"- Iterations: {job['iterations']}")
            
            with meta_col2:
                st.write("**Timeline:**")
                st.write(f"- Created: {_format_timestamp(job['created_at'])}")
                if job['started_at']:
                    st.write(f"- Started: {_format_timestamp(job['started_at'])}")
                if job['completed_at']:
                    st.write(f"- Completed: {_format_timestamp(job['completed_at'])}")
            
            # Active documents processing status - shows ALL documents being processed
            if status == JobStatus.RUNNING.value:
                try:
                    active_docs = json.loads(job.get('active_docs', '[]'))
                except json.JSONDecodeError:
                    active_docs = []
                
                # Also check current_doc for backward compatibility
                if job.get('current_doc') and not active_docs:
                    active_docs = [job['current_doc']]
                
                if active_docs:
                    st.markdown(f"### 🔄 **Currently Processing ({len(active_docs)} documents)**")
                    
                    # Stage-specific emoji and styling (updated for actual pipeline phases)
                    stage_config = {
                        "Starting": ("🚀", "blue"),
                        "Phase 1: Generation": ("✍️", "orange"),
                        "Phase 2: Detector scoring": ("🔍", "blue"),
                        "Phase 3: Gemini quality evaluation": ("⭐", "purple"),
                        "Phase 4: Assembly": ("🔧", "green"),
                        "Completed": ("✅", "green"),
                        "Failed": ("❌", "red"),
                        "Skipped": ("⏭️", "gray"),
                        "Error": ("💥", "red")
                    }
                    
                    # Display each active document
                    for doc_info in active_docs[:5]:  # Show max 5 to avoid clutter
                        if ' | ' in doc_info:
                            doc_name, stage = doc_info.split(' | ', 1)
                            emoji, color = stage_config.get(stage, ("⚙️", "blue"))
                            
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                st.markdown(f"📄 **{doc_name}**")
                            with col2:
                                st.markdown(f"{emoji} :{color}[{stage}]")
                        else:
                            st.markdown(f"📄 **{doc_info}**")
                    
                    # Show count if more documents than displayed
                    if len(active_docs) > 5:
                        st.caption(f"... and {len(active_docs) - 5} more documents")
                
                elif job.get('current_doc'):
                    # Fallback for old format
                    current_info = job['current_doc']
                    if ' | ' in current_info:
                        doc_name, stage = current_info.split(' | ', 1)
                        st.markdown(f"### 🔄 **Currently Processing**")
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            st.markdown(f"**📄 Document:** `{doc_name}`")
                        with col2:
                            stage_config = {
                                "Starting": ("🚀", "blue"),
                                "Phase 1: Generation": ("✍️", "orange"),
                                "Phase 2: Detector scoring": ("🔍", "blue"),
                                "Phase 3: Gemini quality evaluation": ("⭐", "purple"),
                                "Phase 4: Assembly": ("🔧", "green"),
                                "Completed": ("✅", "green"),
                                "Failed": ("❌", "red"),
                                "Skipped": ("⏭️", "gray"),
                                "Error": ("💥", "red")
                            }
                            emoji, color = stage_config.get(stage, ("⚙️", "blue"))
                            st.markdown(f"**{emoji} Stage:** :{color}[{stage}]")
                    else:
                        st.info(f"Currently processing: **{current_info}**")
            
            # Error display
            if job['error']:
                st.error(f"**Error:** {job['error']}")
            
            # Logs
            try:
                logs_data = get_client().get_job_logs(job_id, limit=50)
                logs = logs_data.get('logs', [])
                if logs:
                    with st.expander("📜 Detailed Logs", expanded=False):
                        # Reverse logs to show most recent first
                        for log_entry in reversed(logs[-50:]):  # Last 50 entries
                            ts = _format_timestamp(log_entry['timestamp'])
                            msg = log_entry['message']
                            
                            # Color code based on content
                            if "❌" in msg or "ERROR" in msg.upper():
                                st.markdown(f":red[{ts}] {msg}")
                            elif "✅" in msg or "completed" in msg.lower():
                                st.markdown(f":green[{ts}] {msg}")
                            elif "⚠️" in msg or "skip" in msg.lower():
                                st.markdown(f":orange[{ts}] {msg}")
                            else:
                                st.text(f"{ts} {msg}")
            except Exception as e:
                # Fallback to stored logs
                if job.get('logs'):
                    try:
                        logs = json.loads(job['logs'])
                        if logs:
                            with st.expander("📜 Detailed Logs", expanded=False):
                                for log_entry in reversed(logs[-50:]):
                                    ts = _format_timestamp(log_entry['timestamp'])
                                    msg = log_entry['message']
                                    
                                    if "❌" in msg or "ERROR" in msg.upper():
                                        st.markdown(f":red[{ts}] {msg}")
                                    elif "✅" in msg or "completed" in msg.lower():
                                        st.markdown(f":green[{ts}] {msg}")
                                    elif "⚠️" in msg or "skip" in msg.lower():
                                        st.markdown(f":orange[{ts}] {msg}")
                                    else:
                                        st.text(f"{ts} {msg}")
                    except:
                        pass
        
        st.divider()


def page_job_status():
    """Main job status page."""
    st.header("🔄 Job Status Monitor")
    
    # Page controls - use session state to persist settings
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        auto_refresh = st.checkbox(
            "Auto-refresh", 
            value=st.session_state.get("auto_refresh", True),
            key="auto_refresh_checkbox",
            help="Automatically refresh page every 5 seconds"
        )
        st.session_state.auto_refresh = auto_refresh
    
    with col2:
        show_completed = st.checkbox(
            "Show completed jobs", 
            value=st.session_state.get("show_completed", True),
            key="show_completed_checkbox",
            help="Display completed, failed, and cancelled jobs"
        )
        st.session_state.show_completed = show_completed
    
    with col3:
        if st.button("🗑️ Cleanup", help="Remove jobs older than 7 days"):
            cleanup_old_jobs(7)
            st.success("Old jobs cleaned up")
            st.rerun()
    
    # Create containers for dynamic content to avoid full page reload
    active_jobs_container = st.container()
    completed_jobs_container = st.container()
    
    # Active jobs section
    with active_jobs_container:
        try:
            active_jobs = cached_list_jobs(status="active", limit=20)
        except Exception as e:
            st.warning(f"Could not load active jobs: {e}")
            active_jobs = []
        
        if active_jobs:
            st.subheader(f"🚀 Active Jobs ({len(active_jobs)})")
            
            for job in active_jobs:
                _show_job_card(job)
        else:
            st.info("No active jobs running")
    
    # Recent jobs section - filter out any jobs already shown in active
    with completed_jobs_container:
        if show_completed:
            try:
                recent_jobs = cached_list_jobs(limit=20)
                active_job_ids = {j['job_id'] for j in active_jobs}
                completed_jobs = [
                    j for j in recent_jobs 
                    if j['status'] in (JobStatus.COMPLETED.value, JobStatus.FAILED.value, JobStatus.CANCELLED.value)
                    and j['job_id'] not in active_job_ids  # Don't show jobs already in active section
                ]
            except Exception as e:
                st.warning(f"Could not load completed jobs: {e}")
                completed_jobs = []
            
            if completed_jobs:
                st.subheader(f"📋 Recent Jobs ({len(completed_jobs)})")
                
                for job in completed_jobs:
                    _show_job_card(job)
    
    # Summary statistics
    with st.expander("📊 Job Statistics", expanded=False):
        try:
            all_jobs = cached_list_jobs(limit=100)
        except Exception as e:
            st.warning(f"Could not load job statistics: {e}")
            all_jobs = []
        
        if all_jobs:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total = len(all_jobs)
                st.metric("Total Jobs", total)
            
            with col2:
                completed = sum(1 for j in all_jobs if j['status'] == JobStatus.COMPLETED.value)
                st.metric("Completed", completed)
            
            with col3:
                failed = sum(1 for j in all_jobs if j['status'] == JobStatus.FAILED.value)
                st.metric("Failed", failed)
            
            with col4:
                cancelled = sum(1 for j in all_jobs if j['status'] == JobStatus.CANCELLED.value)
                st.metric("Cancelled", cancelled)
            
            # Success rate
            if total > 0:
                success_rate = (completed / total) * 100
                st.progress(success_rate / 100)
                st.caption(f"Success rate: {success_rate:.1f}%")
    
    # Auto-refresh logic - use a longer interval to reduce reloads
    if auto_refresh and active_jobs:
        # Use session state to track refresh timing
        refresh_interval = 10  # Increased from 5 to 10 seconds to reduce reloads
        
        if "last_refresh_time" not in st.session_state:
            st.session_state.last_refresh_time = time.time()
        
        elapsed = time.time() - st.session_state.last_refresh_time
        remaining = max(0, refresh_interval - elapsed)
        
        # Create a placeholder for the refresh indicator
        refresh_placeholder = st.empty()
        if remaining > 0:
            refresh_placeholder.caption(f"⏱️ Auto-refreshing in {remaining:.0f}s...")
        else:
            refresh_placeholder.caption("🔄 Refreshing...")
            st.session_state.last_refresh_time = time.time()
            # Only rerun if we actually need to refresh
            st.rerun()


# ──────────────────────────── Standalone Page Setup ────────────────────
# When this file is executed directly by Streamlit's multi-page system,
# set up the page config and sidebar, then call the page function
# Check if we're being run as a standalone page (not imported)
if __name__ == "__main__":
    # Page config
    st.set_page_config(page_title="Job Status - Humanizer Test-Bench", layout="wide", initial_sidebar_state="expanded")
    
    # Setup shared sidebar
    from src.pages._shared_layout import setup_sidebar
    setup_sidebar()
    
    # Call the page function
    page_job_status()


