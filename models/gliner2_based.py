import os
import json
import shutil
from datetime import datetime
import torch
from gliner2 import GLiNER2
from gliner2.training.data import InputExample
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig

from utils.logger import setup_logger
from config import RUN_QUICK_TEST, DEBUG_MODE, TARGET_REPO

logger = setup_logger("GLiNER2Model")


class Gliner2PurePyTorch:
    def __init__(self, model_name_or_path, unique_labels, dataset_name):
        self.dataset_name = dataset_name
        self.model_name = model_name_or_path

        logger.info(f"Initializing GLiNER2 model from: {self.model_name}")
        self.model = GLiNER2.from_pretrained(self.model_name)
        self.label_map = {lbl: lbl.lower().replace('/', '_').replace('-', '_') for lbl in unique_labels}

    def prepare_dataset(self, records):
        examples = []
        for rec in records:
            text = rec['source_text']
            entities = {}

            for m in rec.get('privacy_mask', []):
                snake = self.label_map.get(m['label'].upper(), m['label'].lower().replace('/', '_'))
                entity_text = text[m['start']:m['end']]
                if entity_text:
                    entities.setdefault(snake, [])
                    if entity_text not in entities[snake]:
                        entities[snake].append(entity_text)

            if entities:
                examples.append(InputExample(text=text, entities=entities))

        return examples

    def train(self, train_data, api):
        logger.info("STARTING GLINER2 TRAINING LOOP")
        train_examples = self.prepare_dataset(train_data)
        logger.info(f"Prepared {len(train_examples)} training examples")

        model_short = self.model_name.split('/')[-1]
        run_name = f"gliner2-{model_short}"
        out_dir = f"./outputs/{self.dataset_name}/{run_name}"
        os.makedirs(out_dir, exist_ok=True)

        config = TrainingConfig(
            output_dir=out_dir,
            num_epochs=1 if RUN_QUICK_TEST else 3,
            batch_size=8,
            encoder_lr=1e-5,
            task_lr=5e-4,
        )

        trainer = GLiNER2Trainer(self.model, config)
        trainer.train(train_data=train_examples)

        # Save training metadata
        training_meta = {
            "architecture": "gliner2",
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
        # Create an inverse map to convert snake_case labels back to original format
        inverse_label_map = {v: k for k, v in self.label_map.items()}
        target_labels = list(self.label_map.values())

        result = self.model.extract_entities(text, target_labels, include_spans=True)

        entities = []
        for snake_label, spans in result.get('entities', {}).items():
            original_tag = inverse_label_map.get(snake_label, snake_label.upper())
            for span in spans:
                entities.append({
                    "start": span["start"],
                    "end": span["end"],
                    "tag": original_tag,
                    "value": span["text"]
                })

        return entities
