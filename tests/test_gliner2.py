"""
Test Gliner2PurePyTorch - kiểm tra các chức năng:
1. __init__ (load model)
2. prepare_dataset
3. predict
4. cleanup
(train bỏ qua vì cần GPU + thời gian dài)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Patch config để tránh cần .env
import config
config.RUN_QUICK_TEST = True
config.DEBUG_MODE = True
config.TARGET_REPO = "test/repo"

from models.gliner2_based import Gliner2PurePyTorch, _find_latest_checkpoint

# ─────────────────────────────────────────────
# 1. Test _find_latest_checkpoint
# ─────────────────────────────────────────────
print("=" * 50)
print("TEST 1: _find_latest_checkpoint")
result = _find_latest_checkpoint("./nonexistent_path_xyz")
assert result is None, "Should return None for missing dir"
print("  [PASS] Returns None for missing directory")

import tempfile, pathlib
with tempfile.TemporaryDirectory() as tmp:
    # No checkpoints → None
    assert _find_latest_checkpoint(tmp) is None
    print("  [PASS] Returns None for empty directory")

    # Create fake 'best' checkpoint
    best = pathlib.Path(tmp) / "best"
    best.mkdir()
    (best / "config.json").write_text("{}")
    assert _find_latest_checkpoint(tmp) == str(best)
    print(f"  [PASS] Found 'best' checkpoint: {best}")

# epoch checkpoints in a separate temp dir (no 'best')
with tempfile.TemporaryDirectory() as tmp2:
    for n in [10, 20, 5]:
        (pathlib.Path(tmp2) / f"checkpoint-epoch-{n}").mkdir()
    latest = _find_latest_checkpoint(tmp2)
    assert "checkpoint-epoch-20" in latest
    print(f"  [PASS] Found latest epoch checkpoint: {latest}")

# ─────────────────────────────────────────────
# 2. Test __init__ (load model)
# ─────────────────────────────────────────────
print()
print("=" * 50)
print("TEST 2: __init__ - Load model")

LABELS = ["PERSON", "EMAIL", "PHONE_NUM", "DATE", "LOCATION"]
MODEL_NAME = "fastino/gliner2-base-v1"

print(f"  Loading {MODEL_NAME} ...")
model = Gliner2PurePyTorch(MODEL_NAME, LABELS, dataset_name="test_ds")
assert model.model is not None
assert "PERSON" in model.label_map
assert "EMAIL" in model.label_map
assert model.label_map["PERSON"] == "person"
assert model.label_map["EMAIL"] == "email"
print(f"  [PASS] Model loaded, label_map: {model.label_map}")
print(f"  [PASS] inverse_label_map: {model.inverse_label_map}")

# ─────────────────────────────────────────────
# 3. Test prepare_dataset
# ─────────────────────────────────────────────
print()
print("=" * 50)
print("TEST 3: prepare_dataset")

records = [
    {
        "source_text": "Hello, my name is John Doe and my email is john@example.com.",
        "privacy_mask": [
            {"label": "PERSON", "start": 18, "end": 26, "value": "John Doe"},
            {"label": "EMAIL",  "start": 42, "end": 59, "value": "john@example.com"},
        ],
    },
    {
        "source_text": "Call me at +1-800-555-0199.",
        "privacy_mask": [
            {"label": "PHONE_NUM", "start": 11, "end": 25, "value": "+1-800-555-0199"},
        ],
    },
    {
        # Record with no entities → should be skipped
        "source_text": "No PII here.",
        "privacy_mask": [],
    },
]

examples = model.prepare_dataset(records)
assert len(examples) == 2, f"Expected 2 examples (1 skipped), got {len(examples)}"
print(f"  [PASS] {len(examples)} InputExamples created (1 empty record skipped)")

first = examples[0]
assert hasattr(first, "text")
assert "John Doe" in str(first.entities.get("person", []))
print(f"  [PASS] entities in first example: {first.entities}")

# ─────────────────────────────────────────────
# 4. Test predict
# ─────────────────────────────────────────────
print()
print("=" * 50)
print("TEST 4: predict")

text = "My name is Alice Smith and I live in New York. Contact me at alice@example.com."
print(f"  Input: {text}")

preds = model.predict(text)
print(f"  Predictions ({len(preds)}):")
for p in preds:
    print(f"    tag={p['tag']:<12} value={p['value']!r:30} start={p['start']} end={p['end']}")

assert isinstance(preds, list)
for p in preds:
    assert "start" in p and "end" in p and "tag" in p and "value" in p
    assert p["tag"] in LABELS, f"Unexpected tag: {p['tag']}"
    assert p["value"] == text[p["start"]:p["end"]], "value/span mismatch"
print("  [PASS] All predictions have correct schema and consistent span/value")

# ─────────────────────────────────────────────
# 5. Test cleanup
# ─────────────────────────────────────────────
print()
print("=" * 50)
print("TEST 5: cleanup")
model.cleanup()
print("  [PASS] cleanup() ran without error")

print()
print("=" * 50)
print("ALL TESTS PASSED")
