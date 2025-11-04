# src/api/routes/statistics.py
"""
Statistics computation endpoints.
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
import uuid
import time
import threading

from src.api.models import StatisticsRequest, StatisticsResponse, StatisticsTaskResponse
from src.api.dependencies import verify_api_key

router = APIRouter()

# In-memory task storage (in production, use Redis or database)
_statistics_tasks: Dict[str, Dict[str, Any]] = {}
_task_lock = threading.Lock()

@router.post("/aggregate", response_model=StatisticsResponse)
async def compute_statistics(
    request: StatisticsRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    """Start statistics computation asynchronously."""
    try:
        task_id = str(uuid.uuid4())
        
        # Initialize task status
        with _task_lock:
            _statistics_tasks[task_id] = {
                "status": "pending",
                "progress": 0.0,
                "result": None,
                "error": None,
                "created_at": time.time()
            }
        
        # Start background computation
        background_tasks.add_task(
            _compute_statistics_task,
            task_id,
            request.run_names,
            request.merge
        )
        
        return StatisticsResponse(
            task_id=task_id,
            status="pending",
            message="Statistics computation started"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/aggregate/{task_id}", response_model=StatisticsTaskResponse)
async def get_statistics_task(
    task_id: str,
    api_key: str = Depends(verify_api_key)
):
    """Get statistics computation task status and results."""
    with _task_lock:
        task = _statistics_tasks.get(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return StatisticsTaskResponse(
        task_id=task_id,
        status=task["status"],
        progress=task.get("progress"),
        result=task.get("result"),
        error=task.get("error")
    )

def _compute_statistics_task(task_id: str, run_names: List[str], merge: bool):
    """Background task to compute statistics."""
    try:
        with _task_lock:
            _statistics_tasks[task_id]["status"] = "running"
            _statistics_tasks[task_id]["progress"] = 0.1
        
        # Load run data
        from src.results_db import load_run
        from src.pages.benchmark_analysis import _aggregate_statistics_by_model_mode_folder, _merge_runs_data
        
        if merge:
            # Merge runs first
            merged_docs, merge_metadata = _merge_runs_data(run_names)
            docs = merged_docs
        else:
            # Load first run only
            run_data = load_run(run_names[0])
            if not run_data:
                raise ValueError(f"Run not found: {run_names[0]}")
            docs = run_data.get("docs", [])
        
        with _task_lock:
            _statistics_tasks[task_id]["progress"] = 0.5
        
        # Compute statistics
        stats = _aggregate_statistics_by_model_mode_folder(docs)
        
        with _task_lock:
            _statistics_tasks[task_id]["status"] = "completed"
            _statistics_tasks[task_id]["progress"] = 1.0
            _statistics_tasks[task_id]["result"] = stats
            
            # Clean up old tasks (keep last 100)
            if len(_statistics_tasks) > 100:
                oldest_tasks = sorted(
                    _statistics_tasks.items(),
                    key=lambda x: x[1].get("created_at", 0)
                )
                for old_id, _ in oldest_tasks[:-100]:
                    _statistics_tasks.pop(old_id, None)
                    
    except Exception as e:
        with _task_lock:
            _statistics_tasks[task_id]["status"] = "failed"
            _statistics_tasks[task_id]["error"] = str(e)
