import os

from dotenv import load_dotenv

load_dotenv()

CHECKPOINT_PATH = os.getenv("HERMIT_CHECKPOINT", "checkpoints/best_model.pt")
TOKENIZER_PATH = os.getenv("HERMIT_TOKENIZER", "data/tokenizer.json")
DEVICE = os.getenv("HERMIT_DEVICE", "cpu")

DEFAULT_TEMPERATURE = float(os.getenv("HERMIT_TEMPERATURE", "0.6"))
DEFAULT_TOP_K = int(os.getenv("HERMIT_TOP_K", "20"))
DEFAULT_MAX_TOKENS = int(os.getenv("HERMIT_MAX_TOKENS", "80"))

PROBE_PATH = os.getenv("HERMIT_PROBE", "checkpoints/decision_probe.pt")
CONFUSED_THRESHOLD = float(os.getenv("HERMIT_CONFUSED_THRESHOLD", "0.9"))
TOOL_PROBE_PATH = os.getenv("HERMIT_TOOL_PROBE", "checkpoints/decision_tool_probe.pt")
TOOL_THRESHOLD = float(os.getenv("HERMIT_TOOL_THRESHOLD", "0.5"))
BEST_OF = int(os.getenv("HERMIT_BEST_OF", "3"))
CLARIFY = os.getenv("HERMIT_CLARIFY", "1") == "1"
CLARIFY_THRESHOLD = float(os.getenv("HERMIT_CLARIFY_THRESHOLD", "0.75"))
