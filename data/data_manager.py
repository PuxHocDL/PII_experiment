import json
import ast
import random
from typing import List, Tuple, Dict
from datasets import load_dataset
from utils.logger import setup_logger
import config

logger = setup_logger("DataManager")

class PIIDataManager:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.train_data: List[Dict] = []
        self.test_data: List[Dict] = []
        self.required_columns = ['source_text', 'privacy_mask', 'language']

    def _parse_mask(self, mask_data):
        if isinstance(mask_data, str):
            try: return json.loads(mask_data)
            except:
                try: return ast.literal_eval(mask_data)
                except: return []
        return mask_data if isinstance(mask_data, list) else []

    def load_data(self) -> Tuple[List[Dict], List[Dict]]:
        logger.info(f"Loading and processing dataset: {self.dataset_name}")
        dataset = load_dataset(self.dataset_name)
        
        train_split_name = 'train' if 'train' in dataset else list(dataset.keys())[0]
        dataset_columns = dataset[train_split_name].column_names

        missing_cols = [col for col in self.required_columns if col not in dataset_columns]
        if missing_cols:
            error_msg = f"Dataset '{self.dataset_name}' missing required columns: {missing_cols}."
            logger.error(error_msg)
            raise ValueError(error_msg)

        records = []
        for e in dataset[train_split_name]:
            records.append({
                'source_text': str(e['source_text']),
                'privacy_mask': self._parse_mask(e['privacy_mask']),
                'language': str(e['language'])
            })

        random.shuffle(records)

        if config.RUN_QUICK_TEST:
            records = records[:config.QUICK_SAMPLE_SIZE]
        if config.DEBUG_MODE:
            records = records[:config.DEBUG_SAMPLE_SIZE]

        if 'test' in dataset:
            self.train_data = records
            self.test_data = [{
                'source_text': str(e['source_text']),
                'privacy_mask': self._parse_mask(e['privacy_mask']),
                'language': str(e['language'])
            } for e in dataset['test']]
        elif 'validation' in dataset:
            self.train_data = records
            self.test_data = [{
                'source_text': str(e['source_text']),
                'privacy_mask': self._parse_mask(e['privacy_mask']),
                'language': str(e['language'])
            } for e in dataset['validation']]
        else:
            split_idx = int(len(records) * 0.9)
            self.train_data = records[:split_idx]
            self.test_data = records[split_idx:]

        logger.info(f"Dataset: {self.dataset_name.split('/')[-1]} | Train: {len(self.train_data)} | Test: {len(self.test_data)}")
        return self.train_data, self.test_data

    def get_unique_labels(self) -> List[str]:
        labels = set()
        for r in self.train_data + self.test_data:
            for m in r.get('privacy_mask', []):
                lbl = str(m.get('label', '')).strip().upper()
                if lbl: labels.add(lbl)
        return sorted(list(labels))