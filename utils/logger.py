import logging
import random
import numpy as np
import torch
from huggingface_hub import login, HfApi

def setup_logger(name="GenPII"):
    """Create and configure a named logger with console output.
    
    Args:
        name: Logger name for identification in log messages.
    
    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

logger = setup_logger()

def set_seed(seed: int):
    """Set random seed across Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Seed set to {seed}")

def init_hf_api(token: str, repo_id: str):
    """Authenticate with HuggingFace Hub and ensure the target repo exists.
    
    Args:
        token: HuggingFace API token.
        repo_id: Target repository ID (e.g., 'org/repo-name').
    
    Returns:
        Authenticated HfApi instance.
    """
    login(token=token)
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    logger.info(f"HuggingFace Hub initialized. Target repo: {repo_id}")
    return api