# src/api_client.py
"""
API client for Streamlit frontend to communicate with FastAPI backend.
"""

import httpx
import json
from typing import List, Optional, Dict, Any
import streamlit as st
from src.config import API_BASE_URL, WS_URL, APP_AUTH_KEY

class APIClient:
    """Client for making requests to the FastAPI backend with connection pooling."""

    def __init__(self, base_url: str = None):
        self.base_url = base_url or API_BASE_URL
        self.headers = {
            "X-API-Key": APP_AUTH_KEY or "dev",
            "Content-Type": "application/json"
        }
        # Persistent client with connection pooling
        self._client = httpx.Client(
            timeout=10.0,  # Reduced from 30s for faster failures
            limits=httpx.Limits(
                max_keepalive_connections=20,
                max_connections=50,
                keepalive_expiry=30.0
            ),
            http2=True  # Enable HTTP/2 for better performance
        )

    def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make an HTTP request to the API using persistent connection."""
        url = f"{self.base_url}{endpoint}"

        try:
            # Reuse persistent client - no connection overhead
            response = self._client.request(
                method,
                url,
                headers=self.headers,
                **kwargs
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            error_detail = "Unknown error"
            try:
                error_detail = e.response.json().get("detail", str(e))
            except:
                error_detail = str(e)
            raise Exception(f"API error: {error_detail}")
        except Exception as e:
            raise Exception(f"API request failed: {str(e)}")

    def close(self):
        """Close the persistent HTTP client."""
        if self._client:
            self._client.close()

    def __del__(self):
        """Cleanup on deletion."""
        self.close()
    
    # Job endpoints
    def list_jobs(self, status: Optional[str] = None, limit: int = 20) -> List[Dict]:
        """List jobs."""
        params = {"limit": limit}
        if status:
            params["status"] = status
        return self._request("GET", "/api/jobs/", params=params)
    
    def get_job(self, job_id: str) -> Dict:
        """Get a specific job."""
        return self._request("GET", f"/api/jobs/{job_id}")
    
    def create_job(self, job_data: Dict) -> Dict:
        """Create a new job."""
        print(f"[API_CLIENT] create_job called with data: {job_data}", flush=True)
        result = self._request("POST", "/api/jobs/", json=job_data)
        print(f"[API_CLIENT] create_job response: {result}", flush=True)
        return result
    
    def cancel_job(self, job_id: str) -> Dict:
        """Cancel a job."""
        return self._request("DELETE", f"/api/jobs/{job_id}")
    
    def get_job_logs(self, job_id: str, limit: int = 50) -> Dict:
        """Get job logs."""
        return self._request("GET", f"/api/jobs/{job_id}/logs", params={"limit": limit})
    
    # Run endpoints
    def list_runs(self, limit: int = 20, offset: int = 0) -> Dict:
        """List runs with pagination."""
        return self._request("GET", "/api/runs/", params={"limit": limit, "offset": offset})
    
    def get_run(self, run_name: str) -> Dict:
        """Get a specific run."""
        return self._request("GET", f"/api/runs/{run_name}")
    
    def get_run_summary(self, run_name: str) -> Dict:
        """Get run summary without full document data."""
        return self._request("GET", f"/api/runs/{run_name}/summary")
    
    def delete_run(self, run_name: str) -> Dict:
        """Delete a run."""
        return self._request("DELETE", f"/api/runs/{run_name}")
    
    def merge_runs(self, run_names: List[str]) -> Dict:
        """Merge multiple runs."""
        return self._request("POST", "/api/runs/merge", json=run_names)
    
    # Document endpoints
    def analyze_document(self, doc_path: str, compare_run: Optional[str] = None) -> Dict:
        """Analyze a document."""
        params = {"doc_path": doc_path}
        if compare_run:
            params["compare_run"] = compare_run
        return self._request("GET", "/api/documents/analyze", params=params)
    
    def list_documents(self, folders: List[str]) -> Dict:
        """List documents in folders."""
        folders_str = ",".join(folders)
        return self._request("GET", "/api/documents/list", params={"folders": folders_str})
    
    # Statistics endpoints
    def compute_statistics(self, run_names: List[str], merge: bool = False) -> Dict:
        """Start statistics computation."""
        return self._request(
            "POST",
            "/api/statistics/aggregate",
            json={"run_names": run_names, "merge": merge}
        )
    
    def get_statistics_task(self, task_id: str) -> Dict:
        """Get statistics computation task status."""
        return self._request("GET", f"/api/statistics/aggregate/{task_id}")

# Global instance
_client = None

def get_client() -> APIClient:
    """Get or create the global API client instance."""
    global _client
    if _client is None:
        _client = APIClient()
    return _client

# Cached versions for Streamlit with optimized TTLs and spinners
@st.cache_data(ttl=5, show_spinner=False)  # Short TTL for active data, no spinner (fast with connection pooling)
def cached_list_jobs(status: Optional[str] = None, limit: int = 20):
    """Cached version of list_jobs with optimized TTL."""
    return get_client().list_jobs(status=status, limit=limit)

@st.cache_data(ttl=300, show_spinner=False)  # Keep longer TTL for runs, no spinner on cache hit
def cached_get_run(run_name: str):
    """Cached version of get_run."""
    return get_client().get_run(run_name)

@st.cache_data(ttl=60, show_spinner=False)  # Moderate TTL for run lists, no spinner
def cached_list_runs(limit: int = 20, offset: int = 0):
    """Cached version of list_runs."""
    return get_client().list_runs(limit=limit, offset=offset)

