from collections import defaultdict
from typing import List, Dict, Tuple
from utils.logger import setup_logger

logger = setup_logger("Evaluator")

def clean_str(s: str) -> str:
    """Normalize a string by stripping whitespace and lowering case for exact match comparison."""
    return str(s).strip().lower() if s else ""

def convert_token_to_char_spans(spans: List[Dict], text: str) -> List[Dict]:
    """Convert token-index spans to character-offset spans using whitespace tokenization.
    
    If spans already contain 'value' fields, they are assumed to be character-level
    and returned unchanged. Otherwise, maps token indices to character positions.
    
    Args:
        spans: List of span dicts with 'start'/'end' as token indices and 'tag'.
        text: The original text used to compute character offsets.
    
    Returns:
        List of span dicts with character-level 'start', 'end', and 'value' fields.
    """
    if not spans: return []
    if spans[0].get("value", "") != "": return spans 
    
    words = text.split()
    word_offsets = []
    char_idx = 0
    for w in words:
        start_idx = text.find(w, char_idx)
        if start_idx == -1: start_idx = char_idx
        end_idx = start_idx + len(w)
        word_offsets.append((start_idx, end_idx))
        char_idx = end_idx
        
    char_spans = []
    for s in spans:
        tok_s = s["start"]
        tok_e = s["end"] - 1
        if tok_s < len(word_offsets) and tok_e < len(word_offsets):
            char_s = word_offsets[tok_s][0]
            char_e = word_offsets[tok_e][1]
            char_spans.append({
                "tag": s["tag"], 
                "start": char_s, 
                "end": char_e, 
                "value": text[char_s:char_e]
            })
    return char_spans

class BoundaryTolerantEvaluator:
    """Span-level PII evaluator with boundary-tolerant matching.
    
    Matches ground-truth entities to predictions using a combination of:
    - Exact string value matching (case-insensitive, substring)
    - IoU >= 0.9 on character positions
    - Overlap ratio >= 0.9 from either ground-truth or prediction side
    
    This tolerates minor boundary differences while still requiring
    substantial overlap for a match.
    """
    @staticmethod
    def calculate_counts(gt_ents: List[Dict], pred_ents: List[Dict]) -> Tuple[Dict, Dict, Dict, Dict]:
        """Compute per-tag TP, FP, FN, and support counts via greedy matching.
        
        Args:
            gt_ents: Ground-truth entity list with 'tag', 'start', 'end', 'value'.
            pred_ents: Predicted entity list with the same schema.
        
        Returns:
            Tuple of (tp, fp, fn, support) as defaultdict(int) keyed by tag.
        """
        tp, fp, fn, support = defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)
        norm_gt = [{**g, 'tag': g['tag'].strip().upper() if g.get('tag') else ''} for g in gt_ents]
        norm_pred = [{**p, 'tag': p['tag'].strip().upper() if p.get('tag') else ''} for p in pred_ents]
        for g in norm_gt: support[g['tag']] += 1
            
        matched = set()
        for g in norm_gt:
            best_p_i, best_score = -1, -1
            g_val = clean_str(g.get('value', ''))
            g_s, g_e = g.get('start', 0), g.get('end', 0)
            g_len = max(1, g_e - g_s)
                
            for p_i, p in enumerate(norm_pred):
                if p_i in matched or g['tag'] != p['tag']: continue 
                
                p_val = clean_str(p.get('value', ''))
                p_s, p_e = p.get('start', 0), p.get('end', 0)
                p_len = max(1, p_e - p_s)
                
                overlap = max(0, min(g_e, p_e) - max(g_s, p_s))
                exact = (g_val != "") and (g_val == p_val or g_val in p_val or p_val in g_val)
                iou = overlap / (g_len + p_len - overlap) if (g_len + p_len - overlap) > 0 else 0
                overlap_gt = overlap / g_len if g_len > 0 else 0
                overlap_pred = overlap / p_len if p_len > 0 else 0
                
                if exact or (iou >= 0.9) or (overlap_gt >= 0.9) or (overlap_pred >= 0.9):
                    score = overlap + 1000 
                    if score > best_score: 
                        best_score, best_p_i = score, p_i
            
            if best_p_i != -1:
                matched.add(best_p_i)
                tp[g['tag']] += 1
            else: 
                fn[g['tag']] += 1
                
        for p_i, p in enumerate(norm_pred):
            if p_i not in matched: 
                fp[p['tag']] += 1
                
        return tp, fp, fn, support

    @staticmethod
    def compute_metrics(tp: Dict, fp: Dict, fn: Dict, support: Dict) -> Dict[str, float]:
        """Compute support-weighted precision, recall, and F1 across all tags.
        
        Args:
            tp, fp, fn: Per-tag count dicts.
            support: Per-tag ground-truth count dict.
        
        Returns:
            Dict with keys 'w_pre', 'w_rec', 'w_f1' (weighted metrics).
        """
        tags = sorted(list(set(list(tp.keys()) + list(fp.keys()) + list(fn.keys()) + list(support.keys()))))
        total_sup = sum(support.values())
        w_p, w_r, w_f1 = 0.0, 0.0, 0.0
        
        for t in tags:
            c_p = tp[t] / (tp[t] + fp[t]) if (tp[t] + fp[t]) > 0 else 0.0
            c_r = tp[t] / (tp[t] + fn[t]) if (tp[t] + fn[t]) > 0 else 0.0
            c_f1 = (2 * c_p * c_r) / (c_p + c_r) if (c_p + c_r) > 0 else 0.0
            sup = support.get(t, 0)
            
            w_p += c_p * sup
            w_r += c_r * sup
            w_f1 += c_f1 * sup
        
        return {
            "w_pre": w_p / total_sup if total_sup > 0 else 0.0,
            "w_rec": w_r / total_sup if total_sup > 0 else 0.0,
            "w_f1": w_f1 / total_sup if total_sup > 0 else 0.0
        }