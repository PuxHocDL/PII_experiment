# GLiNER2 Model — `models/gliner2_based.py`

Module này cung cấp wrapper để **fine-tune và inference** với [GLiNER2](https://github.com/fastino/gliner2) cho bài toán nhận diện PII (Personally Identifiable Information).

---

## Kiến trúc

GLiNER2 là mô hình NER dạng **generalist span-extraction** dựa trên DeBERTa-v3-base. Khác với NER truyền thống (gắn nhãn cố định), GLiNER2 nhận **danh sách nhãn động** ở mỗi lần inference — phù hợp để phát hiện nhiều loại PII khác nhau mà không cần train lại.

```
Input text + [label_1, label_2, ...]
        ↓
   DeBERTa Encoder
        ↓
   Span Representation
        ↓
   Entity Extraction
        ↓
{label → [(text, start, end), ...]}
```

---

## Cấu trúc file

```
models/
└── gliner2_based.py
    ├── _find_latest_checkpoint()   # tìm checkpoint để resume
    └── Gliner2PurePyTorch          # class chính
        ├── __init__()
        ├── prepare_dataset()
        ├── train()
        ├── predict()
        └── cleanup()
tests/
└── test_gliner2.py                 # test tự động
```

---

## Class `Gliner2PurePyTorch`

### Khởi tạo

```python
from models.gliner2_based import Gliner2PurePyTorch

LABELS = ["PERSON", "EMAIL", "PHONE_NUM", "DATE", "LOCATION", "IP_ADDRESS"]

model = Gliner2PurePyTorch(
    model_name_or_path="fastino/gliner2-base-v1",  # HuggingFace Hub ID hoặc đường dẫn local
    unique_labels=LABELS,
    dataset_name="my_dataset",
)
```

**Tham số:**

| Tham số | Kiểu | Mô tả |
|---------|------|-------|
| `model_name_or_path` | `str` | HF Hub model ID hoặc đường dẫn local |
| `unique_labels` | `List[str]` | Danh sách nhãn PII (chữ HOA) |
| `dataset_name` | `str` | Tên dataset, dùng để đặt tên thư mục output |

Sau khi khởi tạo, model tự động tạo:
- `label_map`: `{"PERSON": "person", "PHONE_NUM": "phone_num", ...}` — chuyển nhãn về snake_case cho GLiNER2
- `inverse_label_map`: ngược lại, để khôi phục nhãn gốc sau inference

---

### `prepare_dataset(records)`

Chuyển danh sách records thành `InputExample` để train.

```python
records = [
    {
        "source_text": "Alice works at OpenAI, her email is alice@openai.com.",
        "privacy_mask": [
            {"label": "PERSON",       "start": 0,  "end": 5},
            {"label": "ORGANIZATION", "start": 15, "end": 21},
            {"label": "EMAIL",        "start": 36, "end": 52},
        ],
    },
    # ...
]

examples = model.prepare_dataset(records)
# → List[InputExample], records không có entity sẽ bị bỏ qua
```

**Lưu ý:**
- Records có `privacy_mask` rỗng bị bỏ qua tự động.
- Mỗi entity text trùng lặp trong cùng một label chỉ được giữ một lần.

---

### `train(train_data, api)`

Fine-tune model và upload lên HuggingFace Hub.

```python
from huggingface_hub import HfApi

api = HfApi()  # hoặc None để chỉ lưu local
model.train(train_data=records, api=api)
```

**Hành vi:**
- Nếu `config.RUN_QUICK_TEST = True` → chỉ train **1 epoch** (dùng để debug nhanh).
- Nếu `config.DEBUG_MODE = True` → không upload lên Hub, giữ file local.
- **Tự động resume** từ checkpoint mới nhất nếu đã có trong `./outputs/<dataset>/<run_name>/`.
- Output được lưu tại: `./outputs/<dataset_name>/gliner2-<model_short>/`

**Cấu trúc output sau train:**
```
outputs/
└── <dataset_name>/
    └── gliner2-<model_short>/
        ├── best/                  # checkpoint tốt nhất (nếu có)
        ├── checkpoint-epoch-N/    # checkpoint từng epoch
        ├── training_meta.json     # metadata
        └── ...
```

**`training_meta.json`:**
```json
{
  "architecture": "gliner2",
  "base_model": "fastino/gliner2-base-v1",
  "dataset": "my_dataset",
  "label_map": {"PERSON": "person", "EMAIL": "email"},
  "timestamp": "2026-04-21T10:00:00"
}
```

---

### `predict(text)`

Inference trên một đoạn văn bản.

```python
text = "My name is Alice Smith and I live in New York. Contact me at alice@example.com."

predictions = model.predict(text)
# [
#   {"start": 11, "end": 22, "tag": "PERSON",   "value": "Alice Smith"},
#   {"start": 37, "end": 45, "tag": "LOCATION", "value": "New York"},
#   {"start": 61, "end": 78, "tag": "EMAIL",    "value": "alice@example.com"},
# ]
```

**Định dạng output — mỗi entity:**

| Key | Kiểu | Mô tả |
|-----|------|-------|
| `start` | `int` | Vị trí ký tự bắt đầu trong `text` |
| `end` | `int` | Vị trí ký tự kết thúc (exclusive) |
| `tag` | `str` | Nhãn PII gốc (chữ HOA, ví dụ `"PERSON"`) |
| `value` | `str` | Chuỗi entity, đúng với `text[start:end]` |

---

### `cleanup()`

Giải phóng bộ nhớ GPU/RAM sau khi dùng xong.

```python
model.cleanup()
```

---

## Hàm tiện ích

### `_find_latest_checkpoint(output_dir)`

Tìm checkpoint phù hợp nhất trong thư mục output, theo thứ tự ưu tiên:
1. Thư mục `best/` (nếu có `config.json` bên trong)
2. Thư mục `checkpoint-epoch-N/` mới nhất (N lớn nhất)
3. `None` nếu không tìm thấy

```python
from models.gliner2_based import _find_latest_checkpoint

ckpt = _find_latest_checkpoint("./outputs/my_dataset/gliner2-base-v1")
# → "./outputs/my_dataset/gliner2-base-v1/best"
```

---

## Tích hợp với `main.py`

`Gliner2PurePyTorch` được tích hợp đầy đủ vào pipeline `main.py`:

```bash
# Train GLiNER2
python main.py --models gliner2 --datasets quynong/my-dataset

# Eval model đã train
python main.py --models gliner2 --eval_only --model_name_or_path ./outputs/my_dataset/gliner2-base-v1

# Train tất cả models (bao gồm GLiNER2)
python main.py --models all --datasets quynong/my-dataset
```

Hoặc dùng **interactive wizard**:
```bash
python main.py
# → chọn architecture: 5. gliner2
```

---

## Cấu hình (`config.py`)

| Biến | Mặc định | Mô tả |
|------|----------|-------|
| `RUN_QUICK_TEST` | `True` | Nếu True: 1 epoch, subset nhỏ |
| `DEBUG_MODE` | `False` | Nếu True: không upload HF Hub |
| `TARGET_REPO` | `""` | HuggingFace repo ID để upload model |
| `HF_TOKEN` | `""` | Token xác thực HuggingFace |

Cấu hình qua file `.env`:
```env
HF_TOKEN=hf_xxxxxxxxxxxx
TARGET_REPO=your_username/your_repo
```

---

## Chạy test

```bash
conda activate piidata
python tests/test_gliner2.py
```

**Tests bao gồm:**
1. `_find_latest_checkpoint` — tìm checkpoint đúng trong các trường hợp
2. `__init__` — load model, kiểm tra `label_map`
3. `prepare_dataset` — tạo `InputExample`, bỏ qua record rỗng
4. `predict` — inference và kiểm tra schema output
5. `cleanup` — giải phóng bộ nhớ

---

## Yêu cầu

```
gliner2>=1.2.6
torch
huggingface_hub
```

Cài đặt:
```bash
pip install gliner2 torch huggingface_hub
```
