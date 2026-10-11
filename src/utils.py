"""
Utility functions for Wood Grain Classification.
Seed management, configuration parsing, history logging, and environment setup.
"""

import os
import json
import random
import logging
from pathlib import Path
import numpy as np
import yaml
import torch

def set_seed(seed: int = 42) -> None:
    """Set random seed for Python, NumPy, and PyTorch for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

def setup_logger(name: str, log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """Configure structured logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger

def save_json(data: dict, file_path: str) -> None:
    """Save dictionary to formatted JSON."""
    p = Path(file_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_json(file_path: str) -> dict:
    """Load JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_device(requested: str = "auto") -> torch.device:
    """
    Get appropriate compute device with Maxwell / sm_50 capability safety check.
    """
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda" and torch.cuda.is_available():
        # Verify capability >= 6.0 for modern PyTorch CUDA support
        cap = torch.cuda.get_device_capability()
        if cap[0] >= 6:
            return torch.device("cuda")
        print(f"[Warning] Detected CUDA device with compute capability {cap[0]}.{cap[1]} (< 6.0). Falling back to CPU for stability.")
        return torch.device("cpu")
    if requested == "auto":
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            if cap[0] >= 6:
                return torch.device("cuda")
        return torch.device("cpu")
    return torch.device("cpu")
