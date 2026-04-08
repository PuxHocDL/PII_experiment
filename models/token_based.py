import json
import shutil
from datetime import datetime
import torch
import torch.nn as nn
from datasets import Dataset
from torchcrf import CRF
from transformers import (
    AutoConfig, AutoTokenizer, AutoModel, AutoModelForTokenClassification, 
    PreTrainedModel, TrainingArguments, Trainer, DataCollatorForTokenClassification
)
from utils.logger import setup_logger
from config import MAX_SEQ_LENGTH, RUN_QUICK_TEST, DEBUG_MODE, TARGET_REPO

logger = setup_logger("TokenBasedModel")

class TransformerCrfForTokenClassification(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = AutoModel.from_config(config)
        dropout_rate = getattr(config, 'hidden_dropout_prob', None) or getattr(config, 'classifier_dropout', None) or 0.1
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self._init_crf_transitions(config)

    def _init_crf_transitions(self, config):
        """Initialize CRF transition matrix with BIO/BIOES constraints."""
        id2label = config.id2label
        num_tags = self.num_labels
        IMPOSSIBLE = -100.0

        with torch.no_grad():
            for i in range(num_tags):
                for j in range(num_tags):
                    label_from = id2label.get(i, id2label.get(str(i), "O"))
                    label_to = id2label.get(j, id2label.get(str(j), "O"))
                    pfrom = label_from.split("-")[0] if "-" in label_from else label_from
                    pto = label_to.split("-")[0] if "-" in label_to else label_to
                    tfrom = label_from.split("-", 1)[1] if "-" in label_from else None
                    tto = label_to.split("-", 1)[1] if "-" in label_to else None

                    # O/E/S -> I is invalid
                    if pfrom in ("O", "E", "S") and pto == "I":
                        self.crf.transitions.data[j, i] = IMPOSSIBLE
                    # O/E/S -> E is invalid
                    elif pfrom in ("O", "E", "S") and pto == "E":
                        self.crf.transitions.data[j, i] = IMPOSSIBLE
                    # B-X -> I-Y or B-X -> E-Y where X != Y
                    elif pfrom == "B" and pto in ("I", "E") and tfrom != tto:
                        self.crf.transitions.data[j, i] = IMPOSSIBLE
                    # I-X -> I-Y or I-X -> E-Y where X != Y
                    elif pfrom == "I" and pto in ("I", "E") and tfrom != tto:
                        self.crf.transitions.data[j, i] = IMPOSSIBLE

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        kwargs.pop('num_items_in_batch', None)
        outputs = self.transformer(input_ids, attention_mask=attention_mask, **kwargs)
        sequence_output = self.dropout(outputs[0])
        emissions = self.classifier(sequence_output)

        # Build a consistent mask for both training and decoding.
        # torchcrf requires mask[:, 0] = True, so we keep [CLS] in the mask
        # but force its label/emission to O. We exclude [SEP] and [PAD].
        crf_mask = attention_mask.clone().bool() if attention_mask is not None else torch.ones_like(input_ids, dtype=torch.bool)
        # Mask out [SEP] (last non-pad token per sequence)
        for i in range(input_ids.size(0)):
            seq_len = attention_mask[i].sum().item() if attention_mask is not None else input_ids.size(1)
            if seq_len > 1:
                crf_mask[i, seq_len - 1] = False

        loss = None
        if labels is not None:
            crf_labels = labels.clone()
            crf_labels[labels == -100] = 0  # [CLS], [SEP], [PAD] → O
            log_likelihood = self.crf(emissions, tags=crf_labels, mask=crf_mask.byte(), reduction='mean')
            seq_lengths = crf_mask.sum(dim=1).float().clamp(min=1)
            loss = -log_likelihood / seq_lengths.mean()

        if not self.training:
            preds = self.crf.decode(emissions, mask=crf_mask.byte())
            device = emissions.device
            fake_logits = torch.full_like(emissions, -1e4, device=device)
            for i, pred_seq in enumerate(preds):
                for j, pred_idx in enumerate(pred_seq):
                    if j < fake_logits.size(1): fake_logits[i, j, pred_idx] = 1e4
            return {"loss": loss, "logits": fake_logits} if loss is not None else {"logits": fake_logits}
        return {"loss": loss, "logits": emissions}

class TokenBasedModule:
    def __init__(self, model_name, unique_labels, dataset_name, use_crf=False, use_bioes=False):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.use_crf = use_crf
        self.use_bioes = use_bioes # <--- Thêm biến này
        
        self.label_list = ["O"]
        for label in unique_labels: 
            if self.use_bioes:
                # Nếu dùng BIOES thì sinh ra 4 loại tag cho mỗi entity
                self.label_list.extend([f"B-{label}", f"I-{label}", f"E-{label}", f"S-{label}"])
            else:
                self.label_list.extend([f"B-{label}", f"I-{label}"])
                
        self.label2id = {l: i for i, l in enumerate(self.label_list)}
        self.id2label = {i: l for i, l in enumerate(self.label_list)}
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.config = AutoConfig.from_pretrained(model_name, num_labels=len(self.label_list), id2label=self.id2label, label2id=self.label2id)
        
        if self.use_crf:
            self.model = TransformerCrfForTokenClassification(self.config)
            pretrain_model = AutoModel.from_pretrained(model_name, config=self.config, ignore_mismatched_sizes=True)
            self.model.transformer.load_state_dict(pretrain_model.state_dict())
            del pretrain_model
        else:
            self.model = AutoModelForTokenClassification.from_pretrained(model_name, config=self.config, ignore_mismatched_sizes=True)

    def tokenize_and_align(self, example):
        tokenized_inputs = self.tokenizer(example['source_text'], max_length=MAX_SEQ_LENGTH, truncation=True, padding=False, return_offsets_mapping=True)
        offset_mapping = tokenized_inputs.pop("offset_mapping")
        labels = [self.label2id["O"]] * len(tokenized_inputs["input_ids"])
        
        # Đầu tiên, gán -100 cho toàn bộ special tokens
        for idx, (tok_start, tok_end) in enumerate(offset_mapping):
            if tok_start == tok_end:
                labels[idx] = -100

        for entity in example.get('privacy_mask', []):
            start_char, end_char = entity['start'], entity['end']
            label_type = entity['label'].strip().upper()
            
            # Thu thập các index token thuộc về entity này
            entity_tokens = []
            for idx, (tok_start, tok_end) in enumerate(offset_mapping):
                if labels[idx] == -100: 
                    continue
                if (tok_start < end_char) and (tok_end > start_char):
                    entity_tokens.append(idx)
                    
            if not entity_tokens:
                continue

            if self.use_bioes:
                if len(entity_tokens) == 1:
                    labels[entity_tokens[0]] = self.label2id.get(f"S-{label_type}", 0)
                else:
                    labels[entity_tokens[0]] = self.label2id.get(f"B-{label_type}", 0)
                    labels[entity_tokens[-1]] = self.label2id.get(f"E-{label_type}", 0)
                    for idx in entity_tokens[1:-1]:
                        labels[idx] = self.label2id.get(f"I-{label_type}", 0)
            else:
                # Hệ BIO cũ
                labels[entity_tokens[0]] = self.label2id.get(f"B-{label_type}", 0)
                for idx in entity_tokens[1:]:
                    labels[idx] = self.label2id.get(f"I-{label_type}", 0)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def train(self, train_data, test_data, api):
        model_short = self.model_name.split('/')[-1]
        scheme = "bioes" if self.use_bioes else "bio"
        crf_tag = "crf-" if self.use_crf else ""
        run_name = f"bert-{crf_tag}{scheme}-{model_short}"
        out_dir = f"./outputs/{self.dataset_name}/{run_name}"
        
        logger.info(f"Training: {run_name}")
        train_ds = Dataset.from_list(train_data).map(self.tokenize_and_align, remove_columns=["source_text", "privacy_mask", "language"])

        training_args = TrainingArguments(
            output_dir=out_dir, 
            learning_rate=5e-5, 
            per_device_train_batch_size=128,
            per_device_eval_batch_size=128, 
            num_train_epochs=3 if self.use_crf else (1 if RUN_QUICK_TEST else 3), 
            weight_decay=0.01,
            warmup_ratio=0.1,
            max_grad_norm=5.0 if self.use_crf else 1.0,
            eval_strategy="no", 
            fp16=False,
            bf16=True, 
            save_strategy="epoch", 
            save_total_limit=1,
            load_best_model_at_end=False, 
            report_to="none",
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
        )
        if self.use_crf:
            transformer_params = list(self.model.transformer.parameters())
            transformer_ids = set(id(p) for p in transformer_params)
            crf_params = list(self.model.crf.parameters())
            crf_ids = set(id(p) for p in crf_params)
            head_params = [p for p in self.model.parameters() if id(p) not in transformer_ids and id(p) not in crf_ids and p.requires_grad]
            optimizer_grouped_parameters = [
                {"params": transformer_params, "lr": 5e-5},
                {"params": head_params, "lr": 1e-3},
                {"params": crf_params, "lr": 8e-2, "weight_decay": 0.0},
            ]
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters, weight_decay=training_args.weight_decay)
            trainer = Trainer(model=self.model, args=training_args, train_dataset=train_ds, data_collator=DataCollatorForTokenClassification(tokenizer=self.tokenizer), optimizers=(optimizer, None))
        else:
            trainer = Trainer(model=self.model, args=training_args, train_dataset=train_ds, data_collator=DataCollatorForTokenClassification(tokenizer=self.tokenizer))
        trainer.train()
        
        trainer.save_model(f"{out_dir}/best_model")
        self.tokenizer.save_pretrained(f"{out_dir}/best_model")
        
        # Save training metadata for reproducibility and auto-detection
        training_meta = {
            "architecture": "token_based",
            "base_model": self.model_name,
            "tagging_scheme": scheme,
            "use_crf": self.use_crf,
            "dataset": self.dataset_name,
            "label_list": self.label_list,
            "max_seq_length": MAX_SEQ_LENGTH,
            "timestamp": datetime.now().isoformat(),
        }
        with open(f"{out_dir}/best_model/training_meta.json", "w", encoding="utf-8") as f:
            json.dump(training_meta, f, indent=2, ensure_ascii=False)
        
        if api and not DEBUG_MODE:
            logger.info(f"Uploading {run_name} to Hugging Face...")
            api.upload_folder(
                folder_path=f"{out_dir}/best_model", 
                repo_id=TARGET_REPO, 
                path_in_repo=f"{self.dataset_name}/{run_name}", 
                repo_type="model"
            )
            shutil.rmtree(out_dir, ignore_errors=True)
            logger.info(f"Local weights cleaned up: {out_dir}")
        else:
            logger.info(f"DEBUG: Weights kept locally at {out_dir}/best_model")

    def predict(self, text: str):
        """Runs inference on a single string of text and returns a list of predicted entities."""
        self.model.eval()
        
        # Tokenize with offset mapping to get exact character boundaries
        inputs = self.tokenizer(
            text, 
            return_offsets_mapping=True, 
            return_tensors="pt", 
            truncation=True, 
            max_length=self.config.max_position_embeddings
        )
        offset_mapping = inputs.pop("offset_mapping")[0].tolist()
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        logits = outputs["logits"][0]
        predictions = torch.argmax(logits, dim=-1).tolist()
        
        entities = []
        current_entity = None
        
        # Decode BIO/BIOES tags into contiguous spans
        for idx, (pred, offset) in enumerate(zip(predictions, offset_mapping)):
            if offset[0] == offset[1]: # Skip special tokens
                continue
                
            label = self.id2label[pred]
            if label == "O":
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
                continue

            prefix = label[:1]  # 'B', 'I', 'E' hoặc 'S'
            tag_type = label[2:]

            if prefix == "S":
                if current_entity: entities.append(current_entity)
                entities.append({"tag": tag_type, "start": offset[0], "end": offset[1]})
                current_entity = None
                
            elif prefix == "B":
                if current_entity: entities.append(current_entity)
                current_entity = {"tag": tag_type, "start": offset[0], "end": offset[1]}
                
            elif prefix == "I" and current_entity and current_entity["tag"] == tag_type:
                current_entity["end"] = offset[1]
                
            elif prefix == "E" and current_entity and current_entity["tag"] == tag_type:
                current_entity["end"] = offset[1]
                entities.append(current_entity)
                current_entity = None
                
            else:
                # Xử lý ngoại lệ (mô hình dự đoán lỗi I, E mà không có B đi trước)
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
                if prefix in ["I", "E"]:
                    current_entity = {"tag": tag_type, "start": offset[0], "end": offset[1]}
                    if prefix == "E": # Đóng luôn nếu vô tình bắt đầu bằng E
                        entities.append(current_entity)
                        current_entity = None

        if current_entity:
            entities.append(current_entity)
            
        # Extract the actual string values for the evaluator
        for ent in entities:
            ent["value"] = text[ent["start"]:ent["end"]]
            
        return entities