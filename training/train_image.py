"""train_image.py: the Modal image used for training, dev scoring and the tests.

torch and transformers are pinned exactly (2.12.1 and 5.12.1), so every run uses the same model
code; the other packages have lower bounds only. The code in this folder is mounted at CODE_DIR.
"""
import pathlib

import modal

LOCAL_DIR = pathlib.Path(__file__).parent
CODE_DIR = "/root/code"  # where this folder's code lives inside the container

PINS = [
    "torch==2.12.1",
    "transformers==5.12.1",
    "accelerate>=1.1.0",
    "datasets>=2.18",
    "scikit-learn>=1.3",
    "scipy>=1.10",
    "sentencepiece>=0.1.99",
    "protobuf>=3.20",
    "huggingface_hub>=0.23",
    "numpy>=1.24",
    "pandas>=2.0",
    "pytest>=8",
]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(*PINS)
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
          "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("train_image")  # so app files can import this module remotely
    # code + committed scorecards only: never data files
    .add_local_dir(LOCAL_DIR, remote_path=CODE_DIR,
                   ignore=["__pycache__", "checkpoints", ".git", "*.pyc", "*.jsonl", "*.csv",
                           "label_sheets", "chart-app", "demo", "leaderboard",
                           "external_rerun", "harness", "release", "*.png"])
)
