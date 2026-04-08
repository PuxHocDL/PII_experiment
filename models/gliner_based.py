import os
import json
import shutil
from datetime import datetime
import torch
import transformers
from torch.optim import AdamW
from torch.utils.data import DataLoader
from gliner import GLiNER
from gliner.data_processing.tokenizer import WordsSplitter 

from utils.logger import setup_logger
from config import RUN_QUICK_TEST, DEBUG_MODE, TARGET_REPO

logger = setup_logger("GLiNERModel")

class GlinerPurePyTorch:
    def __init__(self, model_name_or_path, unique_labels, dataset_name):
        self.dataset_name = dataset_name
        self.model_name = model_name_or_path 
        
        logger.info(f"Initializing GLiNER model from: {self.model_name}")
        self.model = GLiNER.from_pretrained(self.model_name).to('cuda' if torch.cuda.is_available() else 'cpu')
        self.splitter = WordsSplitter()
        self.label_map = {lbl: lbl.lower().replace('/', '_').replace('-', '_') for lbl in unique_labels}
        
        self.collator = self._steal_collator()
        
        
    def _steal_collator(self):
        logger.info("Extracting internal DataCollator from GLiNER framework...")
        original_init = transformers.Trainer.__init__
        stolen_collator = [None]
        
        def patched_init(self_trainer, *args, **kwargs):
            stolen_collator[0] = kwargs.get('data_collator')
            raise StopIteration("Got it!") 
            
        transformers.Trainer.__init__ = patched_init
        
        try:
            from transformers import TrainingArguments
            TrainingArguments.others_lr = 1e-5
            TrainingArguments.others_weight_decay = 0.01
            dummy_args = TrainingArguments(output_dir="./tmp", report_to="none")
            
            dummy_data = [{"tokenized_text": ["a"], "ner": []}]
            self.model.train_model(
                train_dataset=dummy_data, 
                eval_dataset=dummy_data, 
                training_args=dummy_args
            )
        except StopIteration:
            pass 
        except Exception as e:
            logger.warning(f"Exception during collator extraction: {e}")
        finally:
            transformers.Trainer.__init__ = original_init
            
        if stolen_collator[0] is not None:
            logger.info("Successfully extracted DataCollator!")
            return stolen_collator[0]
        else:
            logger.info("Attempting alternative extraction method...")
            import gliner.data_processing.collator as collator_module
            for name in dir(collator_module):
                if 'Collator' in name and name != 'BaseDataCollator':
                    return getattr(collator_module, name)(self.model.config)
            raise RuntimeError("Failed to extract DataCollator for GLiNER!")

    def prepare_dataset(self, records):
        data = []
        for rec in records:
            text = rec['source_text']
            token_info = list(self.splitter(text))
            tokens = [info[0] for info in token_info]
            ner = []
            
            for m in rec.get('privacy_mask', []):
                char_start, char_end = m['start'], m['end']
                snake = self.label_map.get(m['label'].upper(), m['label'].lower().replace('/', '_'))
                
                token_start, token_end = -1, -1
                for i, (_, t_start, t_end) in enumerate(token_info):
                    if t_start < char_end and t_end > char_start:
                        if token_start == -1: token_start = i
                        token_end = i
                        
                if token_start != -1 and token_end != -1:
                    ner.append([token_start, token_end, snake]) 
            
            if len(ner) > 0:
                data.append({"tokenized_text": tokens, "ner": ner})
                
        return data

    def train(self, train_data, api):
        logger.info("STARTING PURE PYTORCH TRAINING LOOP FOR GLiNER BASE")
        train_gliner = self.prepare_dataset(train_data)
        
        dataloader = DataLoader(train_gliner, batch_size=32, shuffle=True, collate_fn=self.collator)
        optimizer = AdamW(self.model.parameters(), lr=2e-5, weight_decay=0.01)
        
        self.model.train()
        epochs = 1 if RUN_QUICK_TEST else 2
        
        for epoch in range(epochs):
            for step, batch in enumerate(dataloader):
                optimizer.zero_grad() 
                
                batch = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                out = self.model(**batch)
                
                loss = out.loss
                loss.backward()
                optimizer.step()
                
                if step % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs} | Step {step:>4} | Loss: {loss.item():.4f}")
        
        logger.info("Uploading GLiNER model to Hugging Face...")
        model_short = self.model_name.split('/')[-1]
        run_name = f"gliner-{model_short}"
        out_dir = f"./outputs/{self.dataset_name}/{run_name}"
        os.makedirs(out_dir, exist_ok=True)
        
        self.model.save_pretrained(out_dir)
        
        # Save training metadata
        training_meta = {
            "architecture": "gliner",
            "base_model": self.model_name,
            "dataset": self.dataset_name,
            "label_map": self.label_map,
            "timestamp": datetime.now().isoformat(),
        }
        with open(f"{out_dir}/training_meta.json", "w", encoding="utf-8") as f:
            json.dump(training_meta, f, indent=2, ensure_ascii=False)
        
        if api and not DEBUG_MODE:
            api.upload_folder(
                folder_path=out_dir, 
                repo_id=TARGET_REPO, 
                path_in_repo=f"{self.dataset_name}/{run_name}", 
                repo_type="model"
            )
            shutil.rmtree(out_dir, ignore_errors=True)
            logger.info(f"Push {run_name} completed, local cleaned up.")
        else:
            logger.info(f"DEBUG: Weights kept locally at {out_dir}")

    def predict(self, text: str):
        self.model.eval()
        # Create an inverse map to map GLiNER's snake_case labels back to original format (e.g., PERSONNAME)
        inverse_label_map = {v: k for k, v in self.label_map.items()}
        target_labels = list(self.label_map.values())
        
        with torch.no_grad():
            preds = self.model.predict_entities(text, target_labels, flat_ner=True)
        
        entities = []
        for p in preds:
            original_tag = inverse_label_map.get(p["label"], p["label"].upper())
            entities.append({
                "start": p["start"],
                "end": p["end"],
                "tag": original_tag,
                "value": p["text"]
            })
            
        return entities