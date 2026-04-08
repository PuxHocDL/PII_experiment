import os
import json
import shutil
from datetime import datetime
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import (
    AutoConfig, AutoTokenizer, AutoModel, 
    PreTrainedModel, TrainingArguments, Trainer
)
from utils.logger import setup_logger
from config import MAX_SEQ_LENGTH, MAX_SPAN_WIDTH, RUN_QUICK_TEST, DEBUG_MODE, TARGET_REPO

logger = setup_logger("SpanBasedModel")

class DebertaAdvancedSpanClassifier(PreTrainedModel):
    def __init__(self, config):
        config.max_position_embeddings = MAX_SEQ_LENGTH 
        super().__init__(config)
        self.deberta = AutoModel.from_config(config)
        self.num_labels = config.num_labels
        self.max_span_width = MAX_SPAN_WIDTH
        
        self.width_embedding = nn.Embedding(MAX_SPAN_WIDTH, config.hidden_size // 2)
        total_feature_size = (config.hidden_size * 2) + (config.hidden_size // 2)
        self.classifier = nn.Sequential(
            nn.Linear(total_feature_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob if hasattr(config, 'hidden_dropout_prob') else 0.1),
            nn.Linear(config.hidden_size, config.num_labels)
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob if hasattr(config, 'hidden_dropout_prob') else 0.1)

    def forward(self, input_ids, attention_mask=None, span_labels=None, **kwargs):
        outputs = self.deberta(input_ids, attention_mask=attention_mask, **kwargs)
        sequence_output = self.dropout(outputs[0]) 
        B, L, H = sequence_output.shape
        device = input_ids.device
        
        start_indices = torch.arange(L, device=device).unsqueeze(1).expand(-1, self.max_span_width)
        end_indices = start_indices + torch.arange(self.max_span_width, device=device).unsqueeze(0)
        valid_mask = end_indices < L 
        
        start_indices, end_indices = start_indices[valid_mask], end_indices[valid_mask]     
        span_starts = start_indices.unsqueeze(0).expand(B, -1)
        span_ends = end_indices.unsqueeze(0).expand(B, -1)     
        
        start_features = torch.gather(sequence_output, 1, span_starts.unsqueeze(-1).expand(-1, -1, H))
        end_features = torch.gather(sequence_output, 1, span_ends.unsqueeze(-1).expand(-1, -1, H))
        span_widths_tensor = (span_ends - span_starts)
        width_features = self.width_embedding(span_widths_tensor)
        
        span_features = torch.cat([start_features, end_features, width_features], dim=-1)
        logits = self.classifier(span_features) 
        loss = None
        
        if span_labels is not None:
            valid_labels = span_labels[:, valid_mask] 
            start_att = torch.gather(attention_mask, 1, span_starts)
            end_att = torch.gather(attention_mask, 1, span_ends)
            span_att_mask = start_att & end_att 
            
            flat_logits = logits.view(-1, self.num_labels)
            flat_labels = valid_labels.reshape(-1)
            flat_mask = span_att_mask.reshape(-1).bool()
            
            if flat_mask.sum() > 0:
                # Negative sampling: keep ALL positive spans + K× random negative spans
                valid_logits = flat_logits[flat_mask]
                valid_labels_flat = flat_labels[flat_mask]
                
                pos_mask = valid_labels_flat > 0
                neg_mask = ~pos_mask
                n_pos = pos_mask.sum().item()
                n_neg = neg_mask.sum().item()
                
                if n_pos > 0 and n_neg > 0:
                    neg_sample_size = min(n_neg, max(n_pos * 5, 256))
                    neg_indices = torch.where(neg_mask)[0]
                    sampled_neg = neg_indices[torch.randperm(n_neg, device=device)[:neg_sample_size]]
                    pos_indices = torch.where(pos_mask)[0]
                    keep_indices = torch.cat([pos_indices, sampled_neg])
                    
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(valid_logits[keep_indices], valid_labels_flat[keep_indices])
                elif n_pos > 0:
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(valid_logits[pos_mask], valid_labels_flat[pos_mask])
                else:
                    loss = torch.tensor(0.0, device=device, requires_grad=True)
            else:
                loss = torch.tensor(0.0, device=device, requires_grad=True)
                
        if not self.training:
            return {
                "loss": loss, 
                "logits": logits, 
                "span_starts": span_starts, 
                "span_ends": span_ends
            }
            
        return {"loss": loss}

class SpanDataCollator:
    def __init__(self, tokenizer): 
        self.tokenizer = tokenizer
        
    def __call__(self, features):
        batch = self.tokenizer([],  is_split_into_words=False, return_tensors="pt", padding=True) if False else {}
        max_len = max(len(f['input_ids']) for f in features)
        input_ids = []
        attention_masks = []
        for f in features:
            pad_len = max_len - len(f['input_ids'])
            input_ids.append(f['input_ids'] + [self.tokenizer.pad_token_id] * pad_len)
            attention_masks.append(f['attention_mask'] + [0] * pad_len)
        batch = {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_masks, dtype=torch.long),
        }
        B, max_seq_len = len(features), batch['input_ids'].shape[1]
        span_labels = torch.zeros((B, max_seq_len, MAX_SPAN_WIDTH), dtype=torch.long)
        for i, f in enumerate(features):
            for span in f['valid_spans']:
                start_idx, width, label_id = span
                if start_idx < max_seq_len: 
                    span_labels[i, start_idx, width] = label_id
        batch['span_labels'] = span_labels
        return batch

class CustomSpanTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("span_labels")
        outputs = model(**inputs, span_labels=labels)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

import glob

class SpanBasedModule:
    def __init__(self, model_name, unique_labels, dataset_name):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.label_list = ["O"] + unique_labels
        self.label2id = {l: i for i, l in enumerate(self.label_list)}
        self.id2label = {i: l for i, l in enumerate(self.label_list)}
        
        self.config = AutoConfig.from_pretrained(model_name, num_labels=len(self.label_list), id2label=self.id2label, label2id=self.label2id)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            add_prefix_space=True, 
            extra_special_tokens={}
        )
        
        self.model = DebertaAdvancedSpanClassifier(self.config)
        
        if os.path.isdir(model_name):
            logger.info(f"Loading FULL trained weights (including Span Heads) from {model_name}")
            weight_path_safe = os.path.join(model_name, "model.safetensors")
            weight_path_bin = os.path.join(model_name, "pytorch_model.bin")
            
            if os.path.exists(weight_path_safe):
                from safetensors.torch import load_file
                self.model.load_state_dict(load_file(weight_path_safe), strict=False)
            elif os.path.exists(weight_path_bin):
                self.model.load_state_dict(torch.load(weight_path_bin, map_location="cpu"), strict=False)
            else:
                logger.warning("No weight files found in directory! Weights may be initialized randomly.")
        else:
            logger.info(f"Loading Base DeBERTa weights for training from {model_name}")
            pretrain_model = AutoModel.from_pretrained(model_name, config=self.config)
            self.model.deberta.load_state_dict(pretrain_model.state_dict(), strict=False) 
            del pretrain_model

    def tokenize_and_build_coords(self, example):
        tokenized_inputs = self.tokenizer(example['source_text'], max_length=MAX_SEQ_LENGTH, truncation=True, padding=False, return_offsets_mapping=True)
        offset_mapping = tokenized_inputs.pop("offset_mapping")
        valid_spans = []
        for entity in example.get('privacy_mask', []):
            start_char, end_char = entity['start'], entity['end']
            label_id = self.label2id.get(entity['label'].strip().upper(), 0)
            start_token_idx, end_token_idx = -1, -1
            
            for idx, (tok_start, tok_end) in enumerate(offset_mapping):
                if tok_start == tok_end: continue
                if start_token_idx == -1 and tok_start <= start_char < tok_end: start_token_idx = idx
                elif start_token_idx == -1 and tok_start >= start_char: start_token_idx = idx
                if tok_start < end_char <= tok_end: end_token_idx = idx
                elif tok_end >= end_char and end_token_idx == -1: end_token_idx = idx
                    
            if start_token_idx != -1 and end_token_idx == -1: end_token_idx = start_token_idx
            if start_token_idx != -1 and end_token_idx != -1 and start_token_idx <= end_token_idx:
                span_width = end_token_idx - start_token_idx
                if span_width < MAX_SPAN_WIDTH:
                    valid_spans.append([start_token_idx, span_width, label_id])
        tokenized_inputs["valid_spans"] = valid_spans
        return tokenized_inputs

    def train(self, train_data, test_data, api):
        model_short = self.model_name.split('/')[-1]
        run_name = f"span-{model_short}"
        out_dir = f"./outputs/{self.dataset_name}/{run_name}"
        
        logger.info(f"Training: {run_name}")
        train_ds = Dataset.from_list(train_data).map(self.tokenize_and_build_coords, remove_columns=["source_text", "privacy_mask", "language"])
        
        args = TrainingArguments(
            output_dir=out_dir, 
            learning_rate=4e-5, 
            per_device_train_batch_size=64, 
            gradient_accumulation_steps=6,
            num_train_epochs=3, 
            weight_decay=0.01, 
            warmup_ratio=0.1,
            eval_strategy="no", 
            fp16=False,
            bf16=True,
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            dataloader_prefetch_factor=2,
            remove_unused_columns=False, 
            report_to="none", 
            save_strategy="epoch", 
            save_total_limit=1,
        )
        trainer = CustomSpanTrainer(model=self.model, args=args, train_dataset=train_ds, data_collator=SpanDataCollator(self.tokenizer))
        trainer.train()
        
        os.makedirs(out_dir, exist_ok=True)
        torch.save(self.model.state_dict(), f"{out_dir}/pytorch_model.bin")
        self.tokenizer.save_pretrained(out_dir)
        self.config.save_pretrained(out_dir)
        
        # Save training metadata for reproducibility and auto-detection
        training_meta = {
            "architecture": "span_based",
            "base_model": self.model_name,
            "dataset": self.dataset_name,
            "label_list": self.label_list,
            "max_seq_length": MAX_SEQ_LENGTH,
            "max_span_width": MAX_SPAN_WIDTH,
            "timestamp": datetime.now().isoformat(),
        }
        with open(f"{out_dir}/training_meta.json", "w", encoding="utf-8") as f:
            json.dump(training_meta, f, indent=2, ensure_ascii=False)
        
        if api and not DEBUG_MODE:
            logger.info(f"Uploading {run_name} to Hugging Face...")
            api.upload_folder(
                folder_path=out_dir, 
                repo_id=TARGET_REPO, 
                path_in_repo=f"{self.dataset_name}/{run_name}", 
                repo_type="model"
            )
            shutil.rmtree(out_dir, ignore_errors=True)
            logger.info(f"Local weights cleaned up: {out_dir}")
        else:
            logger.info(f"DEBUG: Weights kept locally at {out_dir}")

    def predict(self, text: str, threshold: float = None):
        if threshold is None:
            threshold = min(0.5, 2.0 / max(len(self.label_list), 2))
        self.model.eval()
        inputs = self.tokenizer(
            text,
            return_offsets_mapping=True,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False
        )
        offset_mapping = inputs.pop("offset_mapping")[0].tolist()
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs["logits"][0]
        probs = torch.softmax(logits, dim=-1)
        max_probs, pred_ids = torch.max(probs, dim=-1)
        
        span_starts = outputs["span_starts"][0]
        span_ends = outputs["span_ends"][0]

        raw_entities = []
        for prob, pred_id, start_tok, end_tok in zip(max_probs, pred_ids, span_starts, span_ends):
            pred_id = pred_id.item()
            prob = prob.item()
            
            if pred_id == 0:  
                continue
            
            if prob < threshold: 
                continue

            label = self.id2label[pred_id]
            start_tok = start_tok.item()
            end_tok = end_tok.item()

            if start_tok >= len(offset_mapping) or end_tok >= len(offset_mapping):
                continue

            start_char = offset_mapping[start_tok][0]
            end_char = offset_mapping[end_tok][1]

            # Loại trừ span bị lỗi chiều (start >= end)
            if start_char >= end_char:
                continue

            value = text[start_char:end_char]
            raw_entities.append({
                "start": start_char,
                "end": end_char,
                "tag": label,
                "value": value,
                "score": prob
            })

        # NMS (Non-Maximum Suppression)
        raw_entities = sorted(raw_entities, key=lambda x: x["score"], reverse=True)
        
        final_entities = []
        for ent in raw_entities:
            overlap = False
            for kept_ent in final_entities:
                if max(ent["start"], kept_ent["start"]) < min(ent["end"], kept_ent["end"]):
                    overlap = True
                    break
            
            if not overlap:
                final_entities.append(ent)

        final_entities = sorted(final_entities, key=lambda x: x["start"])
        return final_entities