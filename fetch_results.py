"""
Fetch experiment results (evaluation summaries & predictions) from Hugging Face Hub.

Usage:
    # Fetch all results from the configured TARGET_REPO
    python fetch_results.py

    # Fetch results for a specific dataset
    python fetch_results.py --dataset nemotron-pii-ready

    # Fetch only evaluation summaries (lightweight)
    python fetch_results.py --summaries-only

    # Fetch from a custom repo
    python fetch_results.py --repo <your-org>/my-pii-models

    # Fetch trained model weights too
    python fetch_results.py --include-weights

    # List what's available without downloading
    python fetch_results.py --list
"""

import argparse
import os
import json
import sys
from huggingface_hub import HfApi, hf_hub_download, list_repo_tree
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN", None)
TARGET_REPO = os.getenv("TARGET_REPO", "")
OUTPUT_DIR = "./outputs"


def get_api():
    return HfApi(token=HF_TOKEN)


def list_repo_contents(api, repo_id):
    """List all files in the HF repo, grouped by dataset."""
    print(f"\n{'='*60}")
    print(f"  Repository: {repo_id}")
    print(f"{'='*60}")

    try:
        files = list(api.list_repo_tree(repo_id, repo_type="model", recursive=True))
    except Exception as e:
        print(f"Error accessing repo '{repo_id}': {e}")
        return

    # Group files by top-level dataset folder
    datasets = {}
    for f in files:
        if hasattr(f, 'rfilename'):
            path = f.rfilename
        elif hasattr(f, 'path'):
            path = f.path
        else:
            continue

        parts = path.split("/")
        if len(parts) < 2:
            continue
        ds = parts[0]
        datasets.setdefault(ds, []).append(path)

    if not datasets:
        print("  (empty repo)")
        return

    for ds in sorted(datasets):
        files_list = datasets[ds]
        eval_files = [f for f in files_list if "/evaluations/" in f]
        model_files = [f for f in files_list if "/evaluations/" not in f]
        summaries = [f for f in eval_files if f.endswith("eval_summary.json")]

        print(f"\n  Dataset: {ds}")
        print(f"    Evaluation files : {len(eval_files)}")
        print(f"    Model files      : {len(model_files)}")

        if summaries:
            print(f"    Eval summaries   :")
            for s in sorted(summaries):
                model_name = s.split("/evaluations/")[-1].split("/")[0]
                print(f"      - {model_name}")


def fetch_evaluations(api, repo_id, dataset_filter=None, summaries_only=False, output_dir=OUTPUT_DIR):
    """Download evaluation results from the HF repo."""
    print(f"\nFetching evaluations from: {repo_id}")

    try:
        all_files = list(api.list_repo_tree(repo_id, repo_type="model", recursive=True))
    except Exception as e:
        print(f"Error accessing repo: {e}")
        return

    eval_files = []
    for f in all_files:
        path = f.rfilename if hasattr(f, 'rfilename') else getattr(f, 'path', None)
        if not path or "/evaluations/" not in path:
            continue
        if dataset_filter and not path.startswith(dataset_filter):
            continue
        if summaries_only and not path.endswith("eval_summary.json"):
            continue
        eval_files.append(path)

    if not eval_files:
        print("  No evaluation files found.")
        return

    print(f"  Found {len(eval_files)} evaluation file(s) to download.")

    for filepath in sorted(eval_files):
        local_path = os.path.join(output_dir, filepath)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        try:
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=filepath,
                repo_type="model",
                token=HF_TOKEN,
                local_dir=output_dir,
            )
            print(f"  OK  {filepath}")
        except Exception as e:
            print(f"  FAIL {filepath}: {e}")

    # Print summary of downloaded eval_summary files
    summaries = [f for f in eval_files if f.endswith("eval_summary.json")]
    if summaries:
        print(f"\n{'='*60}")
        print("  Evaluation Summary")
        print(f"{'='*60}")
        for s in sorted(summaries):
            local_path = os.path.join(output_dir, s)
            if os.path.exists(local_path):
                with open(local_path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                model = data.get("model_name", "?")
                ds = data.get("dataset", "?")
                f1 = data.get("avg_f1", 0)
                prec = data.get("avg_precision", 0)
                rec = data.get("avg_recall", 0)
                print(f"  {model:30s} | {ds:30s} | P={prec:.4f} R={rec:.4f} F1={f1:.4f}")


def fetch_model_weights(api, repo_id, dataset_filter=None, output_dir=OUTPUT_DIR):
    """Download trained model weights from HF repo."""
    print(f"\nFetching model weights from: {repo_id}")

    try:
        all_files = list(api.list_repo_tree(repo_id, repo_type="model", recursive=True))
    except Exception as e:
        print(f"Error accessing repo: {e}")
        return

    model_files = []
    for f in all_files:
        path = f.rfilename if hasattr(f, 'rfilename') else getattr(f, 'path', None)
        if not path or "/evaluations/" in path:
            continue
        if dataset_filter and not path.startswith(dataset_filter):
            continue
        model_files.append(path)

    if not model_files:
        print("  No model weight files found.")
        return

    print(f"  Found {len(model_files)} model file(s) to download.")

    for filepath in sorted(model_files):
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=filepath,
                repo_type="model",
                token=HF_TOKEN,
                local_dir=output_dir,
            )
            print(f"  OK  {filepath}")
        except Exception as e:
            print(f"  FAIL {filepath}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Fetch PII experiment results from Hugging Face Hub")
    parser.add_argument("--repo", type=str, default=TARGET_REPO,
                        help=f"HuggingFace repo ID (default: TARGET_REPO from .env)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Filter by dataset name (e.g. 'nemotron-pii-ready')")
    parser.add_argument("--summaries-only", action="store_true",
                        help="Only download eval_summary.json files (skip predictions)")
    parser.add_argument("--include-weights", action="store_true",
                        help="Also download trained model weights")
    parser.add_argument("--list", action="store_true",
                        help="List available results without downloading")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help="Local output directory (default: ./outputs)")
    args = parser.parse_args()

    if not args.repo:
        print("Error: No repo specified. Set TARGET_REPO in .env or use --repo")
        sys.exit(1)

    output_dir = args.output_dir

    api = get_api()

    if args.list:
        list_repo_contents(api, args.repo)
        return

    fetch_evaluations(api, args.repo, args.dataset, args.summaries_only, output_dir)

    if args.include_weights:
        fetch_model_weights(api, args.repo, args.dataset, output_dir)

    print(f"\nDone. Results saved to: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()
