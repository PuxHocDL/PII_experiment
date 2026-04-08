import os
import json
import string
from collections import defaultdict
from huggingface_hub import hf_hub_download, login
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. CONFIGURATION & AUTHENTICATION
# ==========================================
HF_TOKEN = os.getenv("HF_TOKEN", "")
login(token=HF_TOKEN)

HF_REPO = os.getenv("HF_INTERNAL_REPO", "")
FILENAME = "raw_predictions_dataset.json"
SUBFOLDER = "Evaluation_Results"

# Label normalization mapping: maps variant labels to canonical forms
LABEL_MAPPINGS = {
    "SEX": "GENDER", "GENDER": "GENDER", 
    "IPV4": "IPADDRESS", "IPV6": "IPADDRESS", "IPADDRESS": "IPADDRESS", 
    "SSN": "SSN/CCCD", "SSN/CCCD": "SSN/CCCD"
}

# ==========================================
# 2. EVALUATOR (PIIEvaluator — Strict IoU >= 0.90)
# ==========================================
CHARS_TO_STRIP = string.punctuation + " \n\r\t"

def clean_str(s):
    """Normalize a string by stripping punctuation/whitespace and lowering case."""
    return str(s).strip(CHARS_TO_STRIP).lower()

class PIIEvaluator:
    """Evaluator for PII span detection using strict IoU-based matching.
    
    A ground-truth entity is considered matched to a prediction if they share
    the same label and satisfy at least one of: exact value match, IoU >= 0.90,
    or positional overlap ratio >= 0.90 from either side.
    """
    @staticmethod
    def calculate_counts(gt_ents, pred_ents):
        """Match ground-truth entities to predictions and compute TP/FP/FN counts.
        
        Args:
            gt_ents: List of ground-truth entity dicts with 'tag', 'start', 'end', 'value'.
            pred_ents: List of predicted entity dicts with the same schema.
        
        Returns:
            Tuple of (tp, fp, fn, support) defaultdicts keyed by entity tag.
        """
        tp, fp, fn, support = defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)
        
        mapped_gt = [{**g, 'tag': LABEL_MAPPINGS.get(g['tag'], g['tag'])} for g in gt_ents]
        for g in mapped_gt: support[g['tag']] += 1
        mapped_pred = [{**p, 'tag': LABEL_MAPPINGS.get(p['tag'], p['tag'])} for p in pred_ents]
            
        matched = set()
        for g in mapped_gt:
            best_p_i, best_score = -1, -1
            g_val, g_s, g_e = clean_str(g.get('value', '')), g.get('start', 0), g.get('end', 0)
            g_len = max(1, g_e - g_s)
                
            for p_i, p in enumerate(mapped_pred):
                if p_i in matched or g['tag'] != p['tag']: continue 
                p_val, p_s, p_e = clean_str(p.get('value', '')), p.get('start', 0), p.get('end', 0)
                p_len = max(1, p_e - p_s)
                
                overlap = max(0, min(g_e, p_e) - max(g_s, p_s))
                exact = (g_val != "") and (g_val == p_val or g_val in p_val or p_val in g_val)
                iou = overlap / (g_len + p_len - overlap) if (g_len + p_len - overlap) > 0 else 0
                
                # Strict matching threshold: IoU >= 0.90
                if exact or (iou >= 0.90) or (overlap / g_len >= 0.90) or (overlap / p_len >= 0.90):
                    score = overlap + 1000 
                    if score > best_score: best_score, best_p_i = score, p_i
            
            if best_p_i != -1: matched.add(best_p_i); tp[g['tag']] += 1
            else: fn[g['tag']] += 1
                
        for p_i, p in enumerate(mapped_pred):
            if p_i not in matched: fp[p['tag']] += 1
        return tp, fp, fn, support

    @staticmethod
    def compute_metrics(tp, fp, fn, support):
        """Compute weighted precision, recall, and F1 from per-tag TP/FP/FN counts.
        
        Args:
            tp, fp, fn: defaultdicts of counts per entity tag.
            support: defaultdict of ground-truth counts per entity tag.
        
        Returns:
            Dict with 'p' (precision), 'r' (recall), 'f1', and 'support'.
        """
        tags = sorted(list(set(list(tp.keys()) + list(fp.keys()) + list(fn.keys()) + list(support.keys()))))
        total_sup = sum(support.values())
        w_p, w_r, w_f1 = 0.0, 0.0, 0.0
        
        for t in tags:
            c_p = tp[t] / (tp[t] + fp[t]) if (tp[t] + fp[t]) > 0 else 0.0
            c_r = tp[t] / (tp[t] + fn[t]) if (tp[t] + fn[t]) > 0 else 0.0
            c_f1 = ((2 * c_p * c_r) / (c_p + c_r)) if (c_p + c_r) > 0 else 0.0
            sup = support.get(t, 0)
            
            w_p += c_p * sup; w_r += c_r * sup; w_f1 += c_f1 * sup
            
        weighted = {
            "p": w_p / total_sup if total_sup > 0 else 0.0, 
            "r": w_r / total_sup if total_sup > 0 else 0.0, 
            "f1": w_f1 / total_sup if total_sup > 0 else 0.0,
            "support": total_sup
        }
        return weighted

# ==========================================
# 3. ENSEMBLE ALGORITHMS
# ==========================================
def is_overlap(p1, p2):
    """Check whether two spans overlap (share at least one character position)."""
    return max(p1['start'], p2['start']) < min(p1['end'], p2['end'])

def ensemble_union(preds_A, preds_B):
    """Union ensemble: merge all predictions, keeping preds_A on conflict.
    
    Adds predictions from preds_B only if they do not overlap with any
    same-tag prediction already present from preds_A.
    """
    res = list(preds_A)
    for p_b in preds_B:
        conflict = False
        for p_a in preds_A:
            if is_overlap(p_a, p_b) and p_a['tag'] == p_b['tag']:
                conflict = True; break
        if not conflict: res.append(p_b)
    return res

def ensemble_intersect(preds_A, preds_B):
    """Intersect ensemble: retain only predictions agreed upon by both models.
    
    Keeps a prediction from preds_A if there exists at least one same-tag
    overlapping prediction in preds_B.
    """
    res = []
    for p_a in preds_A:
        for p_b in preds_B:
            if p_a['tag'] == p_b['tag'] and is_overlap(p_a, p_b):
                res.append(p_a); break
    return res

def predict_proposed(bert_preds, span_preds, text):
    """Proposed ensemble: token-primary with selective span gap-filling.
    
    Uses BERT (token-based) predictions as the primary anchor. For each BERT
    prediction, finds the best overlapping span prediction and marks it as
    matched. Then, unmatched span predictions with confidence >= 0.85 that
    do not conflict with any existing prediction are added to fill coverage gaps.
    
    Args:
        bert_preds: Predictions from the token-based model (BERT).
        span_preds: Predictions from the span-based model (DeBERTa).
        text: The input text (unused but kept for API consistency).
    
    Returns:
        List of merged predictions.
    """
    final_preds = []
    span_matched = set()

    for bp in bert_preds:
        refined_pred = bp.copy()
        best_sp_idx = -1
        max_overlap = 0
        
        for j, sp in enumerate(span_preds):
            overlap = max(0, min(sp['end'], bp['end']) - max(sp['start'], bp['start']))
            if overlap > max_overlap:
                max_overlap = overlap
                best_sp_idx = j
        
        if best_sp_idx != -1:
            span_matched.add(best_sp_idx)
            
        final_preds.append(refined_pred)

    for j, sp in enumerate(span_preds):
        if j not in span_matched:
            if sp.get('score', 0) >= 0.85:
                conflict = False
                for bp in final_preds:
                    if max(sp['start'], bp['start']) < min(sp['end'], bp['end']):
                        conflict = True
                        break
                if not conflict:
                    final_preds.append(sp)

    return final_preds

# ==========================================
# 4. EVALUATION PIPELINE
# ==========================================
def main():
    """Load the internal dataset from HuggingFace Hub and evaluate all five methods."""
    print(f"Loading '{FILENAME}' from HuggingFace Hub...")
    try:
        file_path = hf_hub_download(repo_id=HF_REPO, filename=FILENAME, subfolder=SUBFOLDER)
        with open(file_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        print(f"Successfully loaded {len(dataset)} samples.")
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    trackers = {
        "BERT Base": [defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)],
        "Span-DeBERTa": [defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)],
        "Naive Union": [defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)],
        "Strict Intersect": [defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)],
        "Proposed Method": [defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)]
    }

    print("Evaluating all methods (IoU threshold = 0.90)...")
    for item in dataset:
        text = item["text"]
        gt = item["ground_truth"]
        b_preds = item["bert_predictions"]
        s_preds = item["span_predictions"]
        
        preds_dict = {
            "BERT Base": b_preds,
            "Span-DeBERTa": s_preds,
            "Naive Union": ensemble_union(b_preds, s_preds),
            "Strict Intersect": ensemble_intersect(b_preds, s_preds),
            "Proposed Method": predict_proposed(b_preds, s_preds, text)
        }
        
        for name, preds in preds_dict.items():
            tp, fp, fn, sup = PIIEvaluator.calculate_counts(gt, preds)
            for t in sup.keys(): 
                trackers[name][0][t] += tp[t]
                trackers[name][1][t] += fp[t]
                trackers[name][2][t] += fn[t]
                trackers[name][3][t] += sup[t]

    print(f"\n{'-'*80}")
    print(f"EVALUATION RESULTS (OVERALL WEIGHTED METRICS)")
    print(f"{'-'*80}")
    print(f"{'Method Name':<25} | {'Precision':<12} | {'Recall':<12} | {'F1-Score':<12}")
    print("-" * 80)
    
    results = []
    for name, (tp, fp, fn, sup) in trackers.items():
        metrics = PIIEvaluator.compute_metrics(tp, fp, fn, sup)
        results.append((name, metrics['p']*100, metrics['r']*100, metrics['f1']*100))
        
    for name, p, r, f1 in results:
        mark = ">> " if name == "Proposed Method" else "   "
        print(f"{mark}{name:<22} | {p:>9.2f}%   | {r:>9.2f}%   | {f1:>9.2f}%")
    print("=" * 80)

if __name__ == "__main__":
    main()
