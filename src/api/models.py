# src/api/models.py
"""
Pydantic models for API request/response schemas.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from datetime import datetime

# Job Models
class JobCreate(BaseModel):
    run_name: str
    folders: List[str]
    models: List[str]
    iterations: int
    doc_counts: Dict[str, int] = {}
    include_doc_mode: bool = True
    use_gptzero: bool = True
    use_sapling: bool = True

class JobResponse(BaseModel):
    job_id: str
    run_name: str
    status: str
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    total_docs: int
    processed_docs: int
    current_doc: Optional[str] = None
    active_docs: Optional[str] = None
    folders: List[str]  # Parsed from JSON string
    models: List[str]  # Parsed from JSON string
    iterations: int
    doc_counts: Optional[str] = None  # JSON string
    include_doc_mode: Optional[bool] = True
    use_gptzero: Optional[bool] = True
    use_sapling: Optional[bool] = True
    error: Optional[str] = None
    results: Optional[str] = None  # JSON string
    logs: Optional[str] = None  # JSON string
    
    class Config:
        from_attributes = True

# Run Models
class RunSummary(BaseModel):
    name: str
    timestamp: float
    folders: str  # JSON string
    models: str  # JSON string

class RunResponse(BaseModel):
    name: str
    timestamp: float
    folders: List[str]
    models: List[str]
    docs: List[Dict[str, Any]]
    iterations: Optional[int] = None
    doc_counts: Optional[Dict[str, int]] = None
    include_doc_mode: Optional[bool] = None
    use_gptzero: Optional[bool] = None
    use_sapling: Optional[bool] = None

class RunListResponse(BaseModel):
    runs: List[RunSummary]
    total: int
    offset: int
    limit: int

# Statistics Models
class StatisticsRequest(BaseModel):
    run_names: List[str]
    merge: bool = False

class StatisticsResponse(BaseModel):
    task_id: str
    status: str
    message: str

class StatisticsTaskResponse(BaseModel):
    task_id: str
    status: str
    progress: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# Document Models
class DocumentAnalysisRequest(BaseModel):
    doc_path: str
    compare_run: Optional[str] = None

# WebSocket Models
class JobUpdateMessage(BaseModel):
    job_id: str
    status: str
    processed_docs: int
    total_docs: int
    current_doc: Optional[str] = None
    active_docs: Optional[List[str]] = None
    log_entry: Optional[str] = None

