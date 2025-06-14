# src/cli.py
"""
Command-line batch runner with immediate Ctrl+C handling.

– Documents are processed sequentially (one at a time)
– Within each document, all iterations run concurrently
– Progress is shown document by document
"""

import json
import argparse
import signal
import sys
from pathlib import Path

from .config import REHUMANIZE_N, OPENAI_API_KEY
from .pipeline import run_test
from .models import MODEL_REGISTRY

# Hard-fail early so the user doesn't stare at a blank screen.
if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is empty – create a .env file or export the variable."
    )

def _log(msg: str):
    print(msg, flush=True)

def _handle_sigint(signum, frame):
    print("\nInterrupted by user. Exiting.", flush=True)
    sys.exit(1)

# Install our Ctrl+C handler
signal.signal(signal.SIGINT, _handle_sigint)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--folder",
        default="data/ai_texts",
        help="Folder with .docx files",
    )
    ap.add_argument(
        "--models",
        default="",
        help="Comma-separated display-names (empty = all)",
    )
    ap.add_argument(
        "--out",
        default="results/out.json",
        help="Output JSON file",
    )
    ap.add_argument(
        "--iters",
        type=int,
        default=None,
        help=f"Iterations per model (default {REHUMANIZE_N})",
    )
    args = ap.parse_args()

    iterations = args.iters or REHUMANIZE_N
    models = (
        [m.strip() for m in args.models.split(",") if m.strip()]
        or list(MODEL_REGISTRY)
    )

    docs = list(Path(args.folder).glob("*.docx"))
    if not docs:
        print("No .docx in folder.")
        return

    print(f"Processing {len(docs)} documents sequentially")
    print(f"Models: {', '.join(models)} ({len(models)} total)")
    print(f"Iterations per model: {iterations}")
    print(f"Total drafts per document: {len(models) * iterations * 2} (doc + para modes)")
    print("-" * 60)
    
    results = []
    
    try:
        # Process documents sequentially
        for idx, doc_path in enumerate(docs, 1):
            print(f"\n[Document {idx}/{len(docs)}] Starting: {doc_path.name}")
            print(f"  This document will process {len(models) * iterations} draft pairs concurrently")
            
            result = run_test(doc_path, models, _log, iterations)
            results.append(result)
            
            # Show summary for this document
            if result.get("runs"):
                draft_count = len(result["runs"])
                para_count = result.get("paragraph_count", 0)
                print(f"\n✅ Completed: {doc_path.name}")
                print(f"   - Generated {draft_count} drafts")
                print(f"   - Document has {para_count} paragraphs")
            else:
                print(f"\n⚠️  Skipped: {doc_path.name} (no paragraphs)")

    except KeyboardInterrupt:
        # Our signal handler should catch this first, but just in case:
        print("\nInterrupted by user. Exiting.", flush=True)
        sys.exit(1)

    # Save results
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print("\n" + "=" * 60)
    print(f"✅ All done! Processed {len(results)} documents")
    print(f"📁 Results saved to: {args.out}")

if __name__ == "__main__":
    main()
