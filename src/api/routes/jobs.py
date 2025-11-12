# src/api/routes/jobs.py
"""
Job management endpoints.
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse

from src.api.models import JobCreate, JobResponse
from src.api.dependencies import get_job_db_conn, verify_api_key
from src.job_manager import (
    create_job, get_job, get_active_jobs, get_recent_jobs,
    cancel_job, start_benchmark_job, get_job_logs as get_logs_from_db, JobStatus
)
from pathlib import Path
import json

router = APIRouter()

@router.get("/", response_model=List[JobResponse])
async def list_jobs(
    status: Optional[str] = None,
    limit: int = 20,
    api_key: str = Depends(verify_api_key)
):
    """List jobs, optionally filtered by status."""
    try:
        if status:
            # Filter by status
            if status == "active":
                jobs = get_active_jobs()
            else:
                # Get recent jobs and filter
                jobs = get_recent_jobs(limit=100)
                jobs = [j for j in jobs if j["status"] == status]
                jobs = jobs[:limit]
        else:
            jobs = get_recent_jobs(limit=limit)
        
        # Convert to response format
        result = []
        for job in jobs:
            job_dict = dict(job)
            # Convert include_doc_mode from int to bool if present
            if 'include_doc_mode' in job_dict:
                job_dict['include_doc_mode'] = bool(job_dict['include_doc_mode'])
            # Parse JSON strings for models and folders
            if 'models' in job_dict:
                if isinstance(job_dict['models'], str):
                    try:
                        job_dict['models'] = json.loads(job_dict['models'])
                    except:
                        job_dict['models'] = []
                elif not isinstance(job_dict['models'], list):
                    job_dict['models'] = []
            if 'folders' in job_dict:
                if isinstance(job_dict['folders'], str):
                    try:
                        job_dict['folders'] = json.loads(job_dict['folders'])
                    except:
                        job_dict['folders'] = []
                elif not isinstance(job_dict['folders'], list):
                    job_dict['folders'] = []
            result.append(JobResponse(**job_dict))
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{job_id}", response_model=JobResponse)
async def get_job_by_id(
    job_id: str,
    api_key: str = Depends(verify_api_key)
):
    """Get a specific job by ID."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_dict = dict(job)
    if 'include_doc_mode' in job_dict:
        job_dict['include_doc_mode'] = bool(job_dict['include_doc_mode'])
    # Parse JSON strings for models and folders
    if 'models' in job_dict and isinstance(job_dict['models'], str):
        try:
            job_dict['models'] = json.loads(job_dict['models'])
        except:
            job_dict['models'] = []
    if 'folders' in job_dict and isinstance(job_dict['folders'], str):
        try:
            job_dict['folders'] = json.loads(job_dict['folders'])
        except:
            job_dict['folders'] = []
    
    return JobResponse(**job_dict)

@router.post("/", response_model=JobResponse)
async def create_new_job(
    job_data: JobCreate,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    """Create a new benchmark job."""
    try:
        print("=" * 80, flush=True)
        print("📥 [API] CREATE_NEW_JOB endpoint called", flush=True)
        print("=" * 80, flush=True)
        print(f"[API] Request data: run_name={job_data.run_name}, folders={job_data.folders}, models={job_data.models}", flush=True)
        
        # Gather documents
        from src.pages.new_run import _gather_docs, _folder_doc_counts
        
        FOLDERS = {
            "AI texts": "data/ai_texts",
            "Human texts": "data/human_texts",
            "AI paragraphs": "data/ai_paras",
            "Human paragraphs": "data/human_paras",
        }
        
        print(f"[API] Mapping folder labels: {list(job_data.folders)}", flush=True)
        folder_paths = {f: FOLDERS.get(f, f) for f in job_data.folders}
        print(f"[API] Folder paths: {folder_paths}", flush=True)
        
        limits = _folder_doc_counts(folder_paths)
        print(f"[API] Folder limits: {limits}", flush=True)
        
        # Create doc_counts dict matching folder labels
        doc_counts = {}
        for folder_label in job_data.folders:
            doc_counts[folder_label] = job_data.doc_counts.get(folder_label, limits.get(folder_label, 0))
        
        print(f"[API] Final doc_counts: {doc_counts}", flush=True)
        
        docs = _gather_docs(doc_counts, folder_paths)
        print(f"[API] Gathered {len(docs)} documents", flush=True)
        
        if not docs:
            print("[API] ERROR: No documents found", flush=True)
            raise HTTPException(status_code=400, detail="No documents found for selected folders")
        
        # Start background job (this creates the job AND starts the thread)
        print("[API] Calling start_benchmark_job...", flush=True)
        job_id = start_benchmark_job(
            run_name=job_data.run_name,
            docs=docs,
            folders=job_data.folders,
            models=job_data.models,
            iterations=job_data.iterations,
            doc_counts=doc_counts,
            include_doc_mode=job_data.include_doc_mode,
            use_gptzero=job_data.use_gptzero,
            use_sapling=job_data.use_sapling
        )
        print(f"[API] [OK] start_benchmark_job returned job_id: {job_id}", flush=True)
        
        # Return job details
        job = get_job(job_id)
        if not job:
            print(f"[API] ERROR: Failed to retrieve job {job_id} after creation", flush=True)
            raise HTTPException(status_code=500, detail="Failed to create job")
        
        job_dict = dict(job)
        if 'include_doc_mode' in job_dict:
            job_dict['include_doc_mode'] = bool(job_dict['include_doc_mode'])
        # Parse JSON strings for models and folders
        if 'models' in job_dict and isinstance(job_dict['models'], str):
            try:
                job_dict['models'] = json.loads(job_dict['models'])
            except:
                job_dict['models'] = []
        if 'folders' in job_dict and isinstance(job_dict['folders'], str):
            try:
                job_dict['folders'] = json.loads(job_dict['folders'])
            except:
                job_dict['folders'] = []
        
        print(f"[API] Returning JobResponse for job_id: {job_id}", flush=True)
        return JobResponse(**job_dict)
        
    except HTTPException as e:
        print(f"[API] HTTPException: {e.detail}", flush=True)
        raise
    except Exception as e:
        print(f"[API] [ERROR] Exception: {type(e).__name__}: {e}", flush=True)
        import traceback
        print(traceback.format_exc(), flush=True)
        raise HTTPException(status_code=500, detail=f"Failed to create job: {str(e)}")

@router.delete("/{job_id}")
async def cancel_job_endpoint(
    job_id: str,
    api_key: str = Depends(verify_api_key)
):
    """Cancel a job."""
    success = cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=404, detail="Job not found or cannot be cancelled")
    
    return {"message": "Job cancelled successfully", "job_id": job_id}

@router.post("/{job_id}/cancel")
async def cancel_job_post(
    job_id: str,
    api_key: str = Depends(verify_api_key)
):
    """Cancel a job (POST endpoint for frontend compatibility)."""
    success = cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=404, detail="Job not found or cannot be cancelled")
    
    return {"message": "Job cancelled successfully", "job_id": job_id}

@router.get("/{job_id}/logs")
async def get_job_logs(
    job_id: str,
    limit: int = 50,
    api_key: str = Depends(verify_api_key)
):
    """Get logs for a specific job from separate logs table (fast O(log n) query)."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Use optimized separate logs table query
    logs = get_logs_from_db(job_id, limit=limit)
    return {
        "job_id": job_id,
        "logs": logs,
        "total": len(logs)
    }

