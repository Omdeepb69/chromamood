import os
from pathlib import Path
import logging
import torch

try:
    PROJECT_ROOT = Path(__file__).parent.resolve()
except NameError:
    PROJECT_ROOT = Path.cwd()

class PathConfig:
    ROOT_DIR = PROJECT_ROOT
    DATA_DIR = ROOT_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    MODEL_DIR = ROOT_DIR / "models"
    LOG_DIR = ROOT_DIR / "logs"
    OUTPUT_DIR = ROOT_DIR / "output"
    TENSORBOARD_LOG_DIR = LOG_DIR / "tensorboard"

    for path in [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODEL_DIR,
        LOG_DIR,
        OUTPUT_DIR,
        TENSORBOARD_LOG_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)

class ModelConfig:
    MODEL_NAME = "ChromaMood_v1"
    EMBEDDING_DIM = 128
    HIDDEN_DIM = 256
    NUM_LAYERS = 2
    DROPOUT_RATE = 0.1
    IMAGE_SIZE = (224, 224)
    NUM_CLASSES = 10 # Example: Adjust based on your task

class TrainingConfig:
    SEED = 42
    EPOCHS = 50
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    OPTIMIZER = "AdamW"
    LOSS_FUNCTION = "CrossEntropyLoss"
    VALIDATION_SPLIT = 0.2
    EARLY_STOPPING_PATIENCE = 5
    SCHEDULER_STEP_SIZE = 10
    SCHEDULER_GAMMA = 0.1
    GRADIENT_CLIP_VAL = 1.0
    ACCUMULATE_GRAD_BATCHES = 1

class EnvConfig:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = max(1, os.cpu_count() // 2 if os.cpu_count() else 1)
    LOG_LEVEL = logging.INFO
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    WANDB_PROJECT = "ChromaMood"
    WANDB_ENTITY = os.getenv("WANDB_ENTITY") # Set via environment variable
    WANDB_API_KEY = os.getenv("WANDB_API_KEY") # Set via environment variable
    WANDB_MODE = os.getenv("WANDB_MODE", "online") # "online", "offline", "disabled"

paths = PathConfig()
model_params = ModelConfig()
train_params = TrainingConfig()
env_config = EnvConfig()

if __name__ == "__main__":
    print(f"Project Root: {paths.ROOT_DIR}")
    print(f"Data Directory: {paths.DATA_DIR}")
    print(f"Device: {env_config.DEVICE}")
    print(f"Model Name: {model_params.MODEL_NAME}")
    print(f"Epochs: {train_params.EPOCHS}")
    print(f"Batch Size: {train_params.BATCH_SIZE}")
    print(f"Learning Rate: {train_params.LEARNING_RATE}")
    print(f"Log Level: {logging.getLevelName(env_config.LOG_LEVEL)}")