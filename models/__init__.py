from pathlib import Path

MODELS_PATH = Path(__file__).resolve().parent

CACHE_DIR = MODELS_PATH.parent / ".hf_cache"
