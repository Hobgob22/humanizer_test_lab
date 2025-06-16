# src/pages/__init__.py
"""
Pages package for the Humanizer Test-Bench UI
"""

from .new_run import page_new_run
from .benchmark_analysis import page_runs
from .document_browser import page_browser

__all__ = [
    'page_new_run',
    'page_runs', 
    'page_browser'
]