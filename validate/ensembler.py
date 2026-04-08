from typing import List, Dict
from utils.logger import setup_logger

logger = setup_logger("Ensembler")

def is_overlap(p1: Dict, p2: Dict) -> bool: 
    """Check whether two spans overlap (share at least one character position)."""
    return max(p1['start'], p2['start']) < min(p1['end'], p2['end'])

class BoundaryAwareEnsembler:
    """Ensemble strategies for combining token-based and span-based PII predictions."""
    @staticmethod
    def ensemble_union(preds_A: List[Dict], preds_B: List[Dict]) -> List[Dict]:
        """Union strategy: merge all predictions, prioritizing preds_A on conflict.
        
        Predictions from preds_B are added only if they do not overlap with any
        same-tag prediction from preds_A. Tends to maximize recall at the cost
        of higher false positives.
        """
        res = list(preds_A)
        for p_b in preds_B:
            conflict = False
            for p_a in preds_A:
                if is_overlap(p_a, p_b) and p_a['tag'] == p_b['tag']:
                    conflict = True
                    break
            if not conflict: 
                res.append(p_b)
        return res

    @staticmethod
    def ensemble_intersect(preds_A: List[Dict], preds_B: List[Dict]) -> List[Dict]:
        """Intersect strategy: retain only predictions agreed upon by both models.
        
        Keeps a prediction from preds_A only if at least one same-tag overlapping
        prediction exists in preds_B. Tends to maximize precision at the cost
        of higher false negatives.
        """
        res = []
        for p_a in preds_A:
            for p_b in preds_B:
                if p_a['tag'] == p_b['tag'] and is_overlap(p_a, p_b):
                    res.append(p_a)
                    break
        return res

    @staticmethod
    def ensemble_proposed(token_preds: List[Dict], span_preds: List[Dict]) -> List[Dict]:
        """Proposed strategy: Boundary-Aware Consensus ensemble.
        
        The span-based model (DeBERTa) proposes candidate spans with precise
        boundaries. The token-based model (BERT) validates them semantically.
        A span is kept if the token model agrees (same-tag overlap), or if the
        span's confidence score exceeds 0.75. Unmatched token predictions are
        added to fill remaining gaps.
        """
        final_preds = []

        for sp in span_preds:
            has_overlap_and_same_tag = False
            for tp in token_preds:
                if tp['tag'] == sp['tag'] and is_overlap(sp, tp):
                    has_overlap_and_same_tag = True
                    break
            
            if has_overlap_and_same_tag:
                final_preds.append(sp)
            elif sp.get('score', 0) > 0.75:
                final_preds.append(sp)

        for tp in token_preds:
            overlaps_with_final = False
            for fp in final_preds:
                if is_overlap(tp, fp):
                    overlaps_with_final = True
                    break
            
            if not overlaps_with_final:
                final_preds.append(tp)
                
        return final_preds

    @staticmethod
    def ensemble_token_primary_gapfill(token_preds: List[Dict], span_preds: List[Dict], gap_threshold: float = 0.98) -> List[Dict]:
        """Token-Primary with Selective Span Gap-Fill ensemble.
        
        Uses the token-based model (BERT) as the primary predictor to maintain
        high precision. The span-based model (DeBERTa) supplements with new spans
        (no overlap with existing predictions) only when their confidence score
        meets or exceeds gap_threshold, improving recall without sacrificing precision.
        
        Args:
            token_preds: Primary predictions from the token-based model.
            span_preds: Secondary predictions from the span-based model.
            gap_threshold: Minimum confidence score for span gap-fill (default: 0.98).
        
        Returns:
            Merged prediction list.
        """
        result = list(token_preds)
        for sp in span_preds:
            has_overlap = any(is_overlap(sp, bp) for bp in result)
            if not has_overlap and sp.get('score', 0) >= gap_threshold:
                result.append(sp)
        return result