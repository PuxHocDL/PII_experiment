from typing import List, Dict
from utils.logger import setup_logger

logger = setup_logger("Ensembler")

def is_overlap(p1: Dict, p2: Dict) -> bool: 
    """Kiểm tra xem hai khoảng (span) có giao nhau hay không."""
    return max(p1['start'], p2['start']) < min(p1['end'], p2['end'])

class BoundaryAwareEnsembler:
    @staticmethod
    def ensemble_union(preds_A: List[Dict], preds_B: List[Dict]) -> List[Dict]:
        """Chiến lược Union: Gộp tất cả, ưu tiên preds_A nếu có xung đột (dễ gây nhiễu - False Positive cao)."""
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
        """Chiến lược Intersect: Chỉ lấy những span được cả hai model đồng thuận (dễ bỏ sót - False Negative cao)."""
        res = []
        for p_a in preds_A:
            for p_b in preds_B:
                if p_a['tag'] == p_b['tag'] and is_overlap(p_a, p_b):
                    res.append(p_a)
                    break
        return res

    @staticmethod
    def ensemble_proposed(token_preds: List[Dict], span_preds: List[Dict]) -> List[Dict]:
        """
        Chiến lược đề xuất (Boundary-Aware Consensus):
        - Span-based model (DeBERTa) đề xuất ranh giới.
        - Token-based model (BERT) xác thực ngữ nghĩa.
        - Giữ lại span nếu token đồng ý, hoặc nếu span có độ tin cậy cực cao.
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