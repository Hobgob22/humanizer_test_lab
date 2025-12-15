# src/api/main.py
"""
FastAPI application entry point.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from pathlib import Path
import os

from src.api.routes import jobs, runs, documents, statistics, writing_profile, pricing
from src.api.websocket import router as websocket_router
from src.job_manager import init_db as init_job_db
from src.results_db import _conn as get_runs_db
from src.models import MODEL_REGISTRY

# Initialize databases on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup."""
    print("=" * 80, flush=True)
    print("[START] API Startup: Initializing databases...", flush=True)
    print("=" * 80, flush=True)
    
    # Initialize job database
    print("Initializing job database...", flush=True)
    init_job_db()
    print("[OK] Job database initialized", flush=True)
    
    # Initialize runs database
    try:
        print("Initializing runs database...", flush=True)
        with get_runs_db() as conn:
            pass  # Just ensure connection works
        print("[OK] Runs database initialized", flush=True)
    except Exception as e:
        print(f"[WARNING] Could not initialize runs database: {e}", flush=True)
    
    print("=" * 80, flush=True)
    print("[OK] API Startup complete", flush=True)
    print("=" * 80, flush=True)
    
    yield
    
    # Cleanup on shutdown
    pass

# Create FastAPI app
app = FastAPI(
    title="Humanizer Test-Bench API",
    description="REST API backend for Humanizer Test-Bench",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware - allow React frontend to access API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(jobs.router, prefix="/api/jobs", tags=["jobs"])
app.include_router(runs.router, prefix="/api/runs", tags=["runs"])
app.include_router(documents.router, prefix="/api/documents", tags=["documents"])
app.include_router(statistics.router, prefix="/api/statistics", tags=["statistics"])
app.include_router(websocket_router, prefix="/api", tags=["websocket"])
app.include_router(writing_profile.router, prefix="/api/writing-profile", tags=["writing-profile"])
app.include_router(pricing.router, prefix="/api/pricing", tags=["pricing"])

@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.get("/api/models")
async def get_models():
    """Return the model registry for UI mapping."""
    return MODEL_REGISTRY

# Mount static files for React frontend (if built)
STATIC_DIR = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if STATIC_DIR.exists():
    # Serve static assets (JS, CSS, images)
    app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

    # Catch-all route for React SPA - must be last!
    @app.get("/{full_path:path}")
    async def serve_react_app(full_path: str):
        """Serve React app for all non-API routes."""
        # If it's an API route, let FastAPI handle it
        if full_path.startswith("api/"):
            return {"error": "Not found"}

        # Check if file exists in static directory
        file_path = STATIC_DIR / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)

        # Otherwise serve index.html for React Router
        return FileResponse(STATIC_DIR / "index.html")
else:
    @app.get("/")
    async def root():
        """Root endpoint when React app is not built."""
        return {
            "message": "Humanizer Test-Bench API",
            "version": "1.0.0",
            "docs": "/docs",
            "note": "React frontend not built. Run 'npm run build' in the frontend directory."
        }

if __name__ == "__main__":
    import uvicorn
    import os
    from pathlib import Path
    
    # Load environment variables
    from dotenv import load_dotenv
    ROOT = Path(__file__).resolve().parents[2]
    load_dotenv(ROOT / ".env")
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )

