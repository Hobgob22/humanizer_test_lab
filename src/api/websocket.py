# src/api/websocket.py
"""
WebSocket support for real-time job updates.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, Set
import json
import asyncio
from src.api.models import JobUpdateMessage
from src.job_manager import get_job

router = APIRouter()

# Store active WebSocket connections
_active_connections: Dict[str, Set[WebSocket]] = {}

async def broadcast_job_update(job_id: str, update: dict):
    """Broadcast job update to all connected clients."""
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

