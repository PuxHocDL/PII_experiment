import argparse
import sys
import torch
import random
import gc
import os
import re
import json
import shutil
from collections import defaultdict
from huggingface_hub import HfFileSystem, snapshot_download

import config
from datetime import datetime
from config import HF_TOKEN, TARGET_REPO, DEVICE
from utils.logger import setup_logger, set_seed, init_hf_api
from data.data_manager import PIIDataManager
from models.token_based import TokenBasedModule
from models.span_based import SpanBasedModule
from models.gliner_based import GlinerPurePyTorch
from validate.evaluator import BoundaryTolerantEvaluator

logger = setup_logger("MainPipeline")

def clean_memory(model_obj):
    del model_obj
    torch.cuda.empty_cache()
    gc.collect()

def augment_training_data(train_data):
    from data.augmenter import targeted_augmentation
    logger.info("Applying Targeted Data Augmentation (TA) on training data...")
    augmented = targeted_augmentation(train_data, seed=config.SEED)
    logger.info("Targeted Augmentation completed.")
    return augmented

def resolve_model_path(path):
    """
    Smartly resolves local paths or Hugging Face Hub paths.
    Automatically finds 'best_model' or the latest 'checkpoint-XXX'
    and downloads it efficiently if it's on the Hub.
    """
    if not path:
        return path

    # 1. LOCAL PATH HANDLING
    if os.path.exists(path):
        best_model_dir = os.path.join(path, "best_model")
        if os.path.exists(best_model_dir):
            logger.info(f"🔍 Found 'best_model' locally at: {best_model_dir}")
            return best_model_dir
        
        checkpoints = [d for d in os.listdir(path) if d.startswith("checkpoint-") and os.path.isdir(os.path.join(path, d))]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
            latest_ckpt = os.path.join(path, checkpoints[-1])
            logger.info(f"🔍 Found latest local checkpoint: {latest_ckpt}")
            return latest_ckpt
        return path

    # 2. HUGGING FACE HUB PATH HANDLING
    parts = path.split('/')
    if len(parts) >= 2:
        repo_id = f"{parts[0]}/{parts[1]}"
        subfolder = "/".join(parts[2:]) if len(parts) > 2 else ""
        
        try:
            fs = HfFileSystem()
            resolved_subfolder = subfolder
            
            # Scan Hub for best_model or checkpoints
            best_model_path = f"{path}/best_model" if path else "best_model"
            if fs.exists(best_model_path):
                resolved_subfolder = f"{subfolder}/best_model" if subfolder else "best_model"
                logger.info(f" Found 'best_model' on Hub: {repo_id}/{resolved_subfolder}")
            else:
                contents = fs.ls(path, detail=False)
                checkpoints = [c for c in contents if "checkpoint-" in c.split('/')[-1]]
                if checkpoints:
                    checkpoints.sort(key=lambda x: int(re.search(r'checkpoint-(\d+)', x).group(1)))
                    latest_ckpt = checkpoints[-1]
                    resolved_subfolder = latest_ckpt.replace(f"{repo_id}/", "").lstrip("/")
                    logger.info(f" Found latest checkpoint on Hub: {repo_id}/{resolved_subfolder}")

            logger.info(f"⬇ Downloading specific weights from Hub to local cache...")
            allow_patterns = f"{resolved_subfolder}/**" if resolved_subfolder else None
            
            cached_dir = snapshot_download(
                repo_id=repo_id, 
                allow_patterns=allow_patterns,
                ignore_patterns=["*.msgpack", "*.h5"] # Ignore heavy unused formats
            )
            
            final_path = os.path.join(cached_dir, resolved_subfolder) if resolved_subfolder else cached_dir
            return final_path
            
        except Exception as e:
            logger.warning(f" Could not scan HF Hub deeply (repo might be private). Proceeding with raw path. Error: {e}")
            return path

    return path

def evaluate_model(model_module, test_data, evaluator, dataset_name, model_name, api=None):
    debug = config.DEBUG_MODE
    seeds = [42] if debug else [42, 123, 456, 789, 999]
    max_samples = 5 if debug else 1000
    
    logger.info(f"--- Evaluating Model: {model_name} on {dataset_name} ---")
    if debug:
        logger.info(f"    DEBUG MODE: 1 seed, {max_samples} samples")
    else:
        logger.info(f"    Mode: {len(seeds)} Random Seeds, Max {max_samples} samples per seed")

    all_metrics = {"w_pre": [], "w_rec": [], "w_f1": []}
    
    safe_model_name = model_name.replace(' ', '_').replace('+', 'plus').replace('/', '_').lower()
    output_dir = f"./outputs/{dataset_name.split('/')[-1]}/evaluations/{safe_model_name}"
    os.makedirs(output_dir, exist_ok=True)

    for seed in seeds:
        logger.info(f"  -> Running Evaluation for Seed: {seed}...")
        
        random.seed(seed)
        if len(test_data) > max_samples:
            sampled_test_data = random.sample(test_data, max_samples)
        else:
            sampled_test_data = test_data

        tp, fp, fn, support = defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)
        export_results = []

        for record in sampled_test_data:
            text = record['source_text']
            
            gt_ents = []
            for mask in record['privacy_mask']:
                gt_ents.append({
                    "start": mask.get("start"),
                    "end": mask.get("end"),
                    "tag": mask.get("label"), 
                    "value": mask.get("value", text[mask.get("start", 0):mask.get("end", 0)])
                })

            try:
                pred_ents = model_module.predict(text)
                
                export_results.append({
                    "source_text": text,
                    "ground_truth": gt_ents,
                    "predictions": pred_ents
                })
                
                doc_tp, doc_fp, doc_fn, doc_sup = evaluator.calculate_counts(gt_ents, pred_ents)

                for k, v in doc_tp.items(): tp[k] += v
                for k, v in doc_fp.items(): fp[k] += v
                for k, v in doc_fn.items(): fn[k] += v
                for k, v in doc_sup.items(): support[k] += v
                
            except AttributeError:
                logger.error(f"Evaluation skipped: {model_name} is missing a 'predict(text)' method.")
                return

        metrics = evaluator.compute_metrics(tp, fp, fn, support)
        all_metrics["w_pre"].append(metrics.get('w_pre', 0.0))
        all_metrics["w_rec"].append(metrics.get('w_rec', 0.0))
        all_metrics["w_f1"].append(metrics.get('w_f1', 0.0))
        
        output_file = f"{output_dir}/seed_{seed}_predictions.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_results, f, ensure_ascii=False, indent=4)

    avg_pre = sum(all_metrics["w_pre"]) / len(seeds)
    avg_rec = sum(all_metrics["w_rec"]) / len(seeds)
    avg_f1 = sum(all_metrics["w_f1"]) / len(seeds)

    # Save evaluation summary
    eval_summary = {
        "model_name": model_name,
        "dataset": dataset_name,
        "seeds": seeds,
        "max_samples_per_seed": max_samples,
        "avg_precision": round(avg_pre, 4),
        "avg_recall": round(avg_rec, 4),
        "avg_f1": round(avg_f1, 4),
        "per_seed": {str(s): {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}
                     for s, p, r, f in zip(seeds, all_metrics["w_pre"], all_metrics["w_rec"], all_metrics["w_f1"])},
        "timestamp": datetime.now().isoformat(),
    }
    with open(f"{output_dir}/eval_summary.json", 'w', encoding='utf-8') as f:
        json.dump(eval_summary, f, indent=2, ensure_ascii=False)

    logger.info(f" FINAL AVERAGED RESULTS FOR {model_name} (Across {len(seeds)} Seeds):")
    logger.info(f"   Avg Precision: {avg_pre:.4f}")
    logger.info(f"   Avg Recall:    {avg_rec:.4f}")
    logger.info(f"   Avg F1-Score:  {avg_f1:.4f}")

    # Push evaluations to Hugging Face
    if api and not config.DEBUG_MODE:
        hub_path = f"{dataset_name.split('/')[-1]}/evaluations/{safe_model_name}"
        logger.info(f"Pushing evaluation results to HF: {hub_path}")
        api.upload_folder(
            folder_path=output_dir,
            repo_id=TARGET_REPO,
            path_in_repo=hub_path,
            repo_type="model"
        )
        # Cleanup local evaluation files
        shutil.rmtree(output_dir, ignore_errors=True)
        logger.info(f"Local evaluation files cleaned up: {output_dir}")


def interactive_config():
    """Interactive training wizard (npm init style)."""
    print("\n" + "=" * 60)
    print("   GenPII Framework \u2014 Training Configuration Wizard")
    print("=" * 60)

    # --- Dataset ---
    DATASETS = [
        "PuxAI/financial-pii-ready",
        "PuxAI/nemotron-pii-ready",
        "PuxAI/PII_Merged_Mapped_Dataset_Augmented",
        "ai4privacy/open-pii-masking-500k-ai4privacy",
        "PuxAI/gretel-pii-ready",
    ]
    print("\n  Available datasets:")
    for i, ds in enumerate(DATASETS, 1):
        print(f"    {i}. {ds}")
    print(f"    {len(DATASETS) + 1}. Custom (enter HuggingFace dataset ID)")
    choice = input(f"\n? Select dataset [1-{len(DATASETS) + 1}] (default 1): ").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(DATASETS):
        dataset = DATASETS[int(choice) - 1]
    elif choice == str(len(DATASETS) + 1):
        dataset = input("  Enter HuggingFace dataset ID: ").strip()
    else:
        dataset = DATASETS[0]

    # --- Model architecture ---
    MODELS = [
        ("bert",     "Token-based BERT"),
        ("bert_crf", "Token-based BERT + CRF"),
        ("span",     "Span-based DeBERTa"),
        ("gliner",   "GLiNER"),
        ("all",      "All architectures"),
    ]
    print("\n  Available architectures:")
    for i, (key, desc) in enumerate(MODELS, 1):
        print(f"    {i}. {key:10s} \u2014 {desc}")
    choice = input("\n? Select architecture [1-5] (default 1): ").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(MODELS):
        model_key = MODELS[int(choice) - 1][0]
    else:
        model_key = "bert"

    # --- Base pretrained model ---
    DEFAULT_BASES = {
        "bert":     "google-bert/bert-base-multilingual-cased",
        "bert_crf": "google-bert/bert-base-multilingual-cased",
        "span":     "microsoft/deberta-v3-base",
        "gliner":   "urchade/gliner_multi_pii-v1",
    }
    base_model = None
    if model_key != "all":
        default_base = DEFAULT_BASES.get(model_key, "")
        base_model = input(f"\n? Base pretrained model [{default_base}]: ").strip()
        if not base_model:
            base_model = default_base

    # --- Tagging scheme ---
    use_bioes = False
    if model_key in ("bert", "bert_crf", "all"):
        print("\n  Tagging scheme (token-based models):")
        print("    1. BIO   \u2014 Begin / Inside / Outside")
        print("    2. BIOES \u2014 Begin / Inside / Outside / End / Single")
        choice = input("\n? Select scheme [1-2] (default 1): ").strip()
        use_bioes = (choice == "2")

    # --- Other settings ---
    quick = input("\n? Quick-test mode (small subset)? [y/N]: ").strip().lower() == "y"
    augment = input("? Data augmentation? [y/N]: ").strip().lower() == "y"
    seed_in = input("? Random seed [42]: ").strip()
    seed = int(seed_in) if seed_in.isdigit() else 42

    # --- Summary ---
    scheme_str = "BIOES" if use_bioes else "BIO"
    print("\n" + "=" * 60)
    print("  Configuration Summary")
    print("-" * 60)
    print(f"  Dataset      : {dataset}")
    print(f"  Architecture : {model_key}")
    if base_model:
        print(f"  Base model   : {base_model}")
    if model_key in ("bert", "bert_crf", "all"):
        print(f"  Tag scheme   : {scheme_str}")
    print(f"  Quick test   : {'Yes' if quick else 'No'}")
    print(f"  Augmentation : {'Yes' if augment else 'No'}")
    print(f"  Seed         : {seed}")
    print("=" * 60)

    confirm = input("\n? Proceed? [Y/n]: ").strip().lower()
    if confirm == "n":
        print("Cancelled.")
        exit(0)

    return argparse.Namespace(
        datasets=[dataset],
        models=[model_key],
        quick_test=quick,
        augment=augment,
        seed=seed,
        eval_only=False,
        model_name_or_path=None,
        use_bioes=use_bioes,
        base_model=base_model,
        interactive=True,
    )


def load_training_meta(model_path):
    """Load training_meta.json from a model directory (or its parent)."""
    for candidate in [model_path, os.path.dirname(model_path)]:
        meta_file = os.path.join(candidate, "training_meta.json")
        if os.path.exists(meta_file):
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            logger.info(f"Loaded training metadata from {meta_file}")
            return meta
    return None


def parse_args():
    parser = argparse.ArgumentParser(description="GenPII Framework Multi-Domain & Multi-Model Training")
    
    parser.add_argument("--datasets", nargs='+', default=["PuxAI/PII_Merged_Mapped_Dataset_Augmented"], help="List of HuggingFace dataset repositories to process.")
    parser.add_argument("--models", nargs='+', choices=["bert", "bert_crf", "span", "gliner", "all"], default=["bert"], help="Select the model architectures to process.")
    parser.add_argument("--quick_test", action="store_true", help="Enable Dry-run mode.")
    parser.add_argument("--augment", action="store_true", help="Enable Targeted Data Augmentation (TA).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--eval_only", action="store_true", help="Skip training and only evaluate a pre-trained model.")
    parser.add_argument("--model_name_or_path", type=str, default=None, help="Local directory path OR Hugging Face Hub model ID (Supports subfolders).")
    parser.add_argument("--use_bioes", action="store_true", help="Use BIOES tagging scheme instead of BIO for token-based models.")
    parser.add_argument("--base_model", type=str, default=None, help="Override the default base pretrained model.")
    parser.add_argument("--interactive", "-i", action="store_true", help="Launch interactive training wizard.")
    parser.add_argument("--debug", action="store_true", help="Debug mode: tiny data, 1 epoch, skip HF push, keep local files.")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Always enter interactive mode unless explicit flags are given
    has_explicit_flags = args.eval_only or args.debug or args.quick_test or args.model_name_or_path
    if not has_explicit_flags:
        args = interactive_config()
    config.RUN_QUICK_TEST = args.quick_test
    config.DEBUG_MODE = getattr(args, 'debug', False)
    config.SEED = args.seed

    if config.DEBUG_MODE:
        config.RUN_QUICK_TEST = True
        logger.info("DEBUG MODE ENABLED: tiny data, 1 epoch, no HF push, local files kept")

    if args.eval_only and not args.model_name_or_path:
        logger.error("You must provide a --model_name_or_path when using --eval_only.")
        return

    logger.info("="*80)
    if args.eval_only:
        logger.info("STARTING EVALUATION-ONLY MODE")
    else:
        logger.info("STARTING MULTI-DOMAIN & MULTI-MODEL TRAINING CAMPAIGN")
    logger.info("="*80)

    set_seed(config.SEED)
    if config.DEBUG_MODE:
        api = None
        logger.info("Skipping HF Hub init (debug mode)")
    else:
        api = init_hf_api(HF_TOKEN, TARGET_REPO)
    evaluator = BoundaryTolerantEvaluator()

    models_to_process = args.models
    if "all" in models_to_process:
        models_to_process = ["bert", "bert_crf", "span", "gliner"]

    # --- SMART PATH RESOLVER IN ACTION ---
    resolved_model_path = None
    if args.eval_only:
        logger.info(f"Analyzing provided model path: {args.model_name_or_path}")
        resolved_model_path = resolve_model_path(args.model_name_or_path)

    for dataset_repo in args.datasets:
        safe_name = dataset_repo.split('/')[-1]
        logger.info("\n" + "*"*60)
        logger.info(f"STARTING WORKFLOW FOR DATASET: {safe_name}")
        logger.info("*"*60)

        dm = PIIDataManager(dataset_repo)
        train_data, test_data = dm.load_data()
        unique_labels = dm.get_unique_labels()
        logger.info(f"Recognized {len(unique_labels)} labels in {safe_name}")

        if args.augment and not args.eval_only:
            train_data = augment_training_data(train_data)
            safe_name = f"{safe_name}-augmented"

        # EVALUATION ONLY MODE
        if args.eval_only:
            path_parts = args.model_name_or_path.strip('/').split('/')
            unique_tag = path_parts[-2] if len(path_parts) >= 2 else "loaded"

            # Auto-detect training config from saved metadata
            meta = load_training_meta(resolved_model_path)
            eval_bioes = meta.get("tagging_scheme") == "bioes" if meta else args.use_bioes
            if meta:
                logger.info(f"Auto-detected: scheme={meta.get('tagging_scheme')}, crf={meta.get('use_crf')}")

            if "bert" in models_to_process:
                logger.info(f"--- Loading Pre-trained Base BERT from {resolved_model_path} ---")
                m1 = TokenBasedModule(resolved_model_path, unique_labels, safe_name, use_crf=False, use_bioes=eval_bioes)
                m1.model.to(DEVICE)
                evaluate_model(m1, test_data, evaluator, safe_name, f"Base BERT ({unique_tag})", api)
                clean_memory(m1)

            if "bert_crf" in models_to_process:
                logger.info(f"--- Loading Pre-trained BERT+CRF from {resolved_model_path} ---")
                m2 = TokenBasedModule(resolved_model_path, unique_labels, safe_name, use_crf=True, use_bioes=eval_bioes)
                m2.model.to(DEVICE)
                evaluate_model(m2, test_data, evaluator, safe_name, f"BERT+CRF ({unique_tag})", api)
                clean_memory(m2)
                
            if "span" in models_to_process:
                logger.info(f"--- Loading Pre-trained Span-Based DeBERTa from {resolved_model_path} ---")
                m3 = SpanBasedModule(resolved_model_path, unique_labels, safe_name)
                m3.model.to(DEVICE)
                evaluate_model(m3, test_data, evaluator, safe_name, f"Span DeBERTa ({unique_tag})", api)
                clean_memory(m3)
                
            if "gliner" in models_to_process:
                logger.info(f"--- Loading Pre-trained GLiNER from {resolved_model_path} ---")
                m4 = GlinerPurePyTorch(resolved_model_path, unique_labels, safe_name)
                evaluate_model(m4, test_data, evaluator, safe_name, f"GLiNER ({unique_tag})", api)
                clean_memory(m4)
                
            logger.info(f"EVALUATION COMPLETED FOR {safe_name}\n")
            continue

        # STANDARD TRAINING MODE
        if "bert" in models_to_process:
            base = getattr(args, 'base_model', None) or "google-bert/bert-base-multilingual-cased"
            logger.info(f"--- Training Model: Base BERT ({base}) on {safe_name} ---")
            m1 = TokenBasedModule(base, unique_labels, safe_name, use_crf=False, use_bioes=args.use_bioes)
            m1.model.to(DEVICE)
            m1.train(train_data, test_data, api)
            evaluate_model(m1, test_data, evaluator, safe_name, "Base BERT", api)
            clean_memory(m1)

        if "bert_crf" in models_to_process:
            base = getattr(args, 'base_model', None) or "google-bert/bert-base-multilingual-cased"
            logger.info(f"--- Training Model: BERT + CRF ({base}) on {safe_name} ---")
            m2 = TokenBasedModule(base, unique_labels, safe_name, use_crf=True, use_bioes=args.use_bioes)
            m2.model.to(DEVICE)
            m2.train(train_data, test_data, api)
            evaluate_model(m2, test_data, evaluator, safe_name, "BERT + CRF", api)
            clean_memory(m2)

        if "span" in models_to_process:
            base = getattr(args, 'base_model', None) or "microsoft/deberta-v3-base"
            logger.info(f"--- Training Model: Span-Based DeBERTa ({base}) on {safe_name} ---")
            m3 = SpanBasedModule(base, unique_labels, safe_name)
            m3.model.to(DEVICE)
            m3.train(train_data, test_data, api)
            evaluate_model(m3, test_data, evaluator, safe_name, "Span-Based DeBERTa", api)
            clean_memory(m3)

        if "gliner" in models_to_process:
            base = getattr(args, 'base_model', None) or "urchade/gliner_multi_pii-v1"
            logger.info(f"--- Training Model: GLiNER ({base}) on {safe_name} ---")
            m4 = GlinerPurePyTorch(base, unique_labels, safe_name)
            m4.train(train_data, api)
            evaluate_model(m4, test_data, evaluator, safe_name, "GLiNER", api)
            clean_memory(m4)

        logger.info(f"ALL REQUESTED MODELS COMPLETED, EVALUATED, AND PUSHED FOR {safe_name}\n")

    logger.info("CAMPAIGN COMPLETED. ALL DATASETS AND MODELS PROCESSED.")

if __name__ == "__main__":
    main()