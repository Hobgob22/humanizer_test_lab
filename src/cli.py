# src/cli.py
"""
Command-line batch runner with immediate Ctrl+C handling.

– Documents are processed sequentially (one at a time)
– Within each document, all iterations run concurrently
– Progress is shown document by document
"""
import os
import json
import argparse
import signal
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    ap.add_argument(
        "--max-parallel-docs",
        type=int,
        default=int(os.getenv("MAX_PARALLEL_DOCS", 4)),
        help="Maximum documents processed in parallel",
    )
    ap.add_argument(
        "--no-doc-mode",
        action="store_true",
        help="Disable document-level humanization for AI/Human texts folders (para mode only)",
    )

    args = ap.parse_args()

    iterations = args.iters or REHUMANIZE_N
    include_doc_mode = not args.no_doc_mode  # Convert to positive boolean
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
    print(f"Document-level mode: {'Enabled' if include_doc_mode else 'Disabled (para mode only)'}")
    doc_only   = sum(1 for d in docs if d.parent.name.endswith("_paras"))
    both_modes = len(docs) - doc_only

    if both_modes and doc_only:
        # When doc mode is disabled, regular folders only get para mode
        per_regular = len(models) * iterations * (2 if include_doc_mode else 1)
        per_para    = len(models) * iterations * 1
        print(f"Drafts per document: {per_regular} (regular folders), "
            f"{per_para} (*_paras folders)")
    else:
        # When doc mode is disabled, all documents get only 1 mode
        per_doc  = len(models) * iterations * (2 if (both_modes and include_doc_mode) else 1)
        mode_lbl = "doc + para modes" if (both_modes and include_doc_mode) else "para mode only"
        print(f"Total drafts per document: {per_doc} ({mode_lbl})")
        print("-" * 60)
    
    results = []

    def _worker(p: Path):
        return p, run_test(p, models, _log, iterations, include_doc_mode=include_doc_mode)

    try:
        with ThreadPoolExecutor(max_workers=args.max_parallel_docs) as pool:
            fut2doc = {pool.submit(_worker, p): p for p in docs}

            for idx, fut in enumerate(as_completed(fut2doc), 1):
                p, res = fut.result()
                print(f"\n[{idx}/{len(docs)}] Finished: {p.name}")
                results.append(res)

    except KeyboardInterrupt:
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