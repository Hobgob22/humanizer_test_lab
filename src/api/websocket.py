# src/api/websocket.py
"""
WebSocket support for real-time job updates.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, Set
import json
import asyncio
from src.api.models import JobUpdateMessage
from src.job_manager import get_job, get_active_jobs

router = APIRouter()

# Store active WebSocket connections
_active_connections: Dict[str, Set[WebSocket]] = {}
# Store general WebSocket connections (for all jobs)
_general_connections: Set[WebSocket] = set()

async def broadcast_job_update(job_id: str, update: dict):
    """Broadcast job update to all connected clients."""
    # Broadcast to job-specific connections
    if job_id in _active_connections:
        disconnected = set()
        for connection in _active_connections[job_id]:
            try:
                await connection.send_json(update)
            except:
                disconnected.add(connection)
        
        # Remove disconnected clients
        _active_connections[job_id] -= disconnected
        if not _active_connections[job_id]:
            del _active_connections[job_id]
    
    # Broadcast to general connections (all jobs)
    disconnected = set()
    for connection in _general_connections:
        try:
            await connection.send_json({
                "type": "job_update",
                "job_id": job_id,
                "data": update
            })
        except:
            disconnected.add(connection)
    
    # Remove disconnected clients
    _general_connections -= disconnected

@router.websocket("/ws")
async def websocket_general_updates(websocket: WebSocket):
    """General WebSocket endpoint for all job updates."""
    await websocket.accept()
    _general_connections.add(websocket)
    
    try:
        # Send initial state of all active jobs
        active_jobs = get_active_jobs()
        for job in active_jobs:
            try:
                # Parse JSON strings
                models = json.loads(job.get('models', '[]')) if isinstance(job.get('models'), str) else job.get('models', [])
                folders = json.loads(job.get('folders', '[]')) if isinstance(job.get('folders'), str) else job.get('folders', [])
                
                initial_update = {
                    "type": "job_update",
                    "job_id": job["job_id"],
                    "data": {
                        "job_id": job["job_id"],
                        "status": job["status"],
                        "processed_docs": job["processed_docs"],
                        "total_docs": job["total_docs"],
                        "current_doc": job.get("current_doc"),
                        "active_docs": job.get("active_docs", "[]"),
                        "run_name": job.get("run_name"),
                        "models": models,
                        "folders": folders,
                    }
                }
                await websocket.send_json(initial_update)
            except Exception as e:
                # Skip jobs that fail to parse
                pass
        
        # Keep connection alive and send periodic updates
        while True:
            await asyncio.sleep(2)  # Update every 2 seconds
            
            active_jobs = get_active_jobs()
            for job in active_jobs:
                try:
                    active_docs = json.loads(job.get("active_docs", "[]")) if isinstance(job.get("active_docs"), str) else job.get("active_docs", [])
                    models = json.loads(job.get('models', '[]')) if isinstance(job.get('models'), str) else job.get('models', [])
                    folders = json.loads(job.get('folders', '[]')) if isinstance(job.get('folders'), str) else job.get('folders', [])
                    
                    update = {
                        "type": "job_update",
                        "job_id": job["job_id"],
                        "data": {
                            "job_id": job["job_id"],
                            "status": job["status"],
                            "processed_docs": job["processed_docs"],
                            "total_docs": job["total_docs"],
                            "current_doc": job.get("current_doc"),
                            "active_docs": active_docs,
                            "run_name": job.get("run_name"),
                            "models": models,
                            "folders": folders,
                        }
                    }
                    await websocket.send_json(update)
                except Exception as e:
                    # Skip jobs that fail to parse
                    pass
                    
    except WebSocketDisconnect:
        pass
    finally:
        _general_connections.discard(websocket)

@router.websocket("/ws/jobs/{job_id}")
async def websocket_job_updates(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for real-time job updates."""
    await websocket.accept()
    
    # Add connection to active set
    if job_id not in _active_connections:
        _active_connections[job_id] = set()
    _active_connections[job_id].add(websocket)
    
    try:
        # Send initial job state
        job = get_job(job_id)
        if job:
            initial_update = {
                "job_id": job_id,
                "status": job["status"],
                "processed_docs": job["processed_docs"],
                "total_docs": job["total_docs"],
                "current_doc": job.get("current_doc"),
                "active_docs": job.get("active_docs", "[]"),
                "type": "initial"
            }
            await websocket.send_json(initial_update)
        
        # Keep connection alive and send periodic updates
        while True:
            await asyncio.sleep(2)  # Update every 2 seconds
            
            job = get_job(job_id)
            if not job:
                await websocket.send_json({
                    "type": "error",
                    "message": "Job not found"
                })
                break
            
            # Only send updates for active jobs
            if job["status"] in ("pending", "running"):
                try:
                    active_docs = json.loads(job.get("active_docs", "[]"))
                except:
                    active_docs = []
                
                update = {
                    "job_id": job_id,
                    "status": job["status"],
                    "processed_docs": job["processed_docs"],
                    "total_docs": job["total_docs"],
                    "current_doc": job.get("current_doc"),
                    "active_docs": active_docs,
                    "type": "update"
                }
                await websocket.send_json(update)
            else:
                # Job completed/failed/cancelled, send final update and close
                final_update = {
                    "job_id": job_id,
                    "status": job["status"],
                    "processed_docs": job["processed_docs"],
                    "total_docs": job["total_docs"],
                    "type": "final"
                }
                await websocket.send_json(final_update)
                break
                
    except WebSocketDisconnect:
        pass
    finally:
        # Remove connection
        if job_id in _active_connections:
            _active_connections[job_id].discard(websocket)
            if not _active_connections[job_id]:
                del _active_connections[job_id]

