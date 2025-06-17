# src/pages/job_status.py
"""
Job status page for monitoring background benchmark jobs.
Shows active jobs, recent completed jobs, and detailed logs.
"""

import json
import time
from datetime import datetime

import streamlit as st

from src.job_manager import (
    get_active_jobs, get_recent_jobs, get_job, cancel_job,
    JobStatus, cleanup_old_jobs
)

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
            # Progress bar for active jobs
            if status in (JobStatus.PENDING.value, JobStatus.RUNNING.value):
                progress = job['processed_docs'] / job['total_docs'] if job['total_docs'] > 0 else 0
                st.progress(progress)
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
                    if cancel_job(job_id):
                        st.success("Job cancelled")
                        time.sleep(0.5)
                        st.rerun()
            
            if st.button("📋", key=f"details_{job_id}", help="Show details"):
                st.session_state[f"show_details_{job_id}"] = not st.session_state.get(f"show_details_{job_id}", False)
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
            
            # Current status
            if job['current_doc'] and status == JobStatus.RUNNING.value:
                st.info(f"Currently processing: **{job['current_doc']}**")
            
            # Error display
            if job['error']:
                st.error(f"**Error:** {job['error']}")
            
            # Logs
            if job['logs']:
                logs = json.loads(job['logs'])
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
        
        st.divider()

def page_job_status():
    """Main job status page."""
    st.header("🔄 Job Status Monitor")
    
    # Page controls
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        auto_refresh = st.checkbox("Auto-refresh", value=True, 
                                  help="Automatically refresh page every 5 seconds")
    
    with col2:
        show_completed = st.checkbox("Show completed jobs", value=True,
                                   help="Display completed, failed, and cancelled jobs")
    
    with col3:
        if st.button("🗑️ Cleanup", help="Remove jobs older than 7 days"):
            cleanup_old_jobs(7)
            st.success("Old jobs cleaned up")
            st.rerun()
    
    # Active jobs section
    active_jobs = get_active_jobs()
    
    if active_jobs:
        st.subheader(f"🚀 Active Jobs ({len(active_jobs)})")
        
        for job in active_jobs:
            _show_job_card(job)
    else:
        st.info("No active jobs running")
    
    # Recent jobs section
    if show_completed:
        recent_jobs = get_recent_jobs(limit=20)
        completed_jobs = [
            j for j in recent_jobs 
            if j['status'] in (JobStatus.COMPLETED.value, JobStatus.FAILED.value, JobStatus.CANCELLED.value)
        ]
        
        if completed_jobs:
            st.subheader(f"📋 Recent Jobs ({len(completed_jobs)})")
            
            for job in completed_jobs:
                _show_job_card(job)
    
    # Summary statistics
    with st.expander("📊 Job Statistics", expanded=False):
        all_jobs = get_recent_jobs(limit=100)
        
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
    
    # Auto-refresh logic
    if auto_refresh and active_jobs:
        time.sleep(5)  # Refresh every 5 seconds
        st.rerun()