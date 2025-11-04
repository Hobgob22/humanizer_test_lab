# src/api/routes/documents.py
"""
Document analysis endpoints.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, Query
from pathlib import Path

from src.api.models import DocumentAnalysisRequest
from src.api.dependencies import verify_api_key
from src.pipeline import load_ai_scores

router = APIRouter()

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

