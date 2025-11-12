# src/api/routes/documents.py
"""
Document analysis endpoints.
"""

from typing import Optional, List
from fastapi import APIRouter, HTTPException, Depends, Query
from pathlib import Path

from src.api.models import DocumentAnalysisRequest
from src.api.dependencies import verify_api_key
from src.pipeline import load_ai_scores

router = APIRouter()

@router.get("/{run_name}")
async def get_run_documents(
    run_name: str,
    api_key: str = Depends(verify_api_key)
):
    """Get list of documents in a specific run."""
    try:
        from src.results_db import load_run

        run_data = load_run(run_name)
        if not run_data:
            raise HTTPException(status_code=404, detail=f"Run '{run_name}' not found")

        docs = run_data.get("docs", [])
        doc_names = [doc.get("document", "unknown") for doc in docs]

        return {
            "documents": doc_names,
            "total": len(doc_names)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{run_name}/{doc_name}")
async def get_document_details(
    run_name: str,
    doc_name: str,
    api_key: str = Depends(verify_api_key)
):
    """Get detailed analysis for a specific document in a run."""
    try:
        from src.results_db import load_run

        run_data = load_run(run_name)
        if not run_data:
            raise HTTPException(status_code=404, detail=f"Run '{run_name}' not found")

        docs = run_data.get("docs", [])

        # Find the matching document
        matching_doc = None
        for doc in docs:
            if doc.get("document") == doc_name:
                matching_doc = doc
                break

        if not matching_doc:
            raise HTTPException(status_code=404, detail=f"Document '{doc_name}' not found in run")

        # Convert runs array to models dict for easier frontend consumption
        models = {}
        for run in matching_doc.get("runs", []):
            model_name = run.get("model", "unknown")
            mode = run.get("mode", "unknown")

            if model_name not in models:
                models[model_name] = {"iterations": []}

            # Extract relevant data
            iteration_data = {
                "iter": run.get("iter", 0),
                "mode": mode,
                "para_ai_score": None,
                "doc_ai_score": None,
                "para_quality_score": None,
                "doc_quality_score": None,
                "para_rewritten": None,
                "doc_rewritten": None,
            }

            # Get AI scores
            if "scores_after" in run and "group_doc" in run["scores_after"]:
                iteration_data["doc_ai_score"] = run["scores_after"]["group_doc"].get("gptzero")

            if "scores_after" in run and "group_par" in run["scores_after"]:
                para_scores = run["scores_after"]["group_par"].get("gptzero", [])
                if para_scores and isinstance(para_scores, list) and len(para_scores) > 0:
                    iteration_data["para_ai_score"] = sum(para_scores) / len(para_scores)

            # Get quality scores from flag_counts
            flag_counts = run.get("flag_counts", {})
            iteration_data["para_quality_score"] = flag_counts.get("grammar_score")
            iteration_data["doc_quality_score"] = flag_counts.get("grammar_score")

            models[model_name]["iterations"].append(iteration_data)

        return {
            "doc_name": matching_doc.get("document"),
            "folder": matching_doc.get("folder"),
            "paragraph_count": matching_doc.get("paragraph_count"),
            "models": models
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analyze")
async def analyze_document(
    doc_path: str,
    compare_run: Optional[str] = None,
    api_key: str = Depends(verify_api_key)
):
    """Analyze a document and optionally compare with a benchmark run."""
    try:
        doc_path_obj = Path(doc_path)
        if not doc_path_obj.exists():
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Load AI detection scores
        scores = load_ai_scores(doc_path_obj)
        
        result = {
            "document": doc_path,
            "scores": scores
        }
        
        # If compare_run specified, add comparison data
        if compare_run:
            from src.results_db import load_run
            run_data = load_run(compare_run)
            if run_data:
                # Find matching document in run
                docs = run_data.get("docs", [])
                matching_doc = None
                for doc in docs:
                    if doc.get("document") == doc_path_obj.name:
                        matching_doc = doc
                        break
                
                if matching_doc:
                    result["comparison"] = {
                        "run_name": compare_run,
                        "document_data": matching_doc
                    }
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/list")
async def list_documents(
    folders: str = Query(..., description="Comma-separated folder names"),
    api_key: str = Depends(verify_api_key)
):
    """List documents in specified folders."""
    try:
        from src.pages.utils import natural_key
        
        folder_list = [f.strip() for f in folders.split(",")]
        ROOT = Path(__file__).resolve().parents[3]
        
        docs = []
        for folder in folder_list:
            folder_path = ROOT / f"data/{folder}"
            if folder_path.exists():
                folder_docs = sorted(
                    folder_path.glob("*.docx"),
                    key=natural_key
                )
                docs.extend([{
                    "name": d.name,
                    "path": str(d),
                    "folder": folder
                } for d in folder_docs])
        
        return {
            "documents": docs,
            "total": len(docs)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

