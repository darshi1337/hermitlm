import os

from dotenv import load_dotenv

load_dotenv()

CHECKPOINT_PATH = os.getenv("HERMIT_CHECKPOINT", "checkpoints/best_model.pt")
TOKENIZER_PATH = os.getenv("HERMIT_TOKENIZER", "data/tokenizer.json")
DEVICE = os.getenv("HERMIT_DEVICE", "cpu")

DEFAULT_TEMPERATURE = float(os.getenv("HERMIT_TEMPERATURE", "0.6"))
DEFAULT_TOP_K = int(os.getenv("HERMIT_TOP_K", "20"))
DEFAULT_MAX_TOKENS = int(os.getenv("HERMIT_MAX_TOKENS", "80"))
