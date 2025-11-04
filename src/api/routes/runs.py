# src/api/routes/runs.py
"""
Run management endpoints.
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import JSONResponse

from src.api.models import RunSummary, RunResponse, RunListResponse
from src.api.dependencies import verify_api_key
from src.results_db import list_runs, load_run, delete_run

router = APIRouter()

@router.get("/", response_model=RunListResponse)
async def list_runs_endpoint(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    api_key: str = Depends(verify_api_key)
):
    """List all runs with pagination."""
    try:
        # Use optimized pagination from database
        paginated_runs, total = list_runs(limit=limit, offset=offset)

        # Convert to response format
        summaries = []
        for run in paginated_runs:
            summaries.append(RunSummary(
                name=run["name"],
                timestamp=run["ts"],
                folders=run["folders"],
                models=run["models"]
            ))

        return RunListResponse(
            runs=summaries,
            total=total,
            offset=offset,
            limit=limit
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{run_name}", response_model=RunResponse)
async def get_run_endpoint(
    run_name: str,
    api_key: str = Depends(verify_api_key)
):
    """Get a specific run by name."""
    try:
        run_data = load_run(run_name)
        if not run_data:
            raise HTTPException(status_code=404, detail="Run not found")
        
        # Convert to response format
        return RunResponse(
            name=run_name,
            timestamp=run_data.get("timestamp", 0),
            folders=run_data.get("folders", []),
            models=run_data.get("models", []),
            docs=run_data.get("docs", []),
            iterations=run_data.get("iterations"),
            doc_counts=run_data.get("doc_counts"),
            include_doc_mode=run_data.get("include_doc_mode"),
            use_gptzero=run_data.get("use_gptzero"),
            use_sapling=run_data.get("use_sapling")
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{run_name}")
async def delete_run_endpoint(
    run_name: str,
    api_key: str = Depends(verify_api_key)
):
    """Delete a run."""
    try:
        delete_run(run_name)
        return {"message": "Run deleted successfully", "run_name": run_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/merge")
async def merge_runs_endpoint(
    run_names: List[str],
    api_key: str = Depends(verify_api_key)
):
    """Merge multiple runs into one."""
    try:
        from src.pages.benchmark_analysis import _merge_runs_data
        
        merged_docs, merge_metadata = _merge_runs_data(run_names)
        
        return {
            "merged_docs": merged_docs,
            "metadata": merge_metadata
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{run_name}/summary")
async def get_run_summary_endpoint(
    run_name: str,
    api_key: str = Depends(verify_api_key)
):
    """Get run summary without full document data."""
    try:
        run_data = load_run(run_name)
        if not run_data:
            raise HTTPException(status_code=404, detail="Run not found")
        
        docs = run_data.get("docs", [])
        
        return {
            "name": run_name,
            "timestamp": run_data.get("timestamp", 0),
            "folders": run_data.get("folders", []),
            "models": run_data.get("models", []),
            "total_documents": len(docs),
            "iterations": run_data.get("iterations"),
            "doc_counts": run_data.get("doc_counts"),
            "include_doc_mode": run_data.get("include_doc_mode"),
            "use_gptzero": run_data.get("use_gptzero"),
            "use_sapling": run_data.get("use_sapling")
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

