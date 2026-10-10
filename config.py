from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"

SEED = 42
N_SIMS = 1000
HORIZON_DAYS = 60

# Policy knobs, not model outputs
GREEN_MAX = 0.10
AMBER_MAX = 0.40

CHASE_TOP_N = 10
CHASE_SHIFT_DAYS = 10
LATE_THRESHOLD_DAYS = 14

# Agent
MAX_AGENT_STEPS = 4
LANGS = ["en", "ur"]          # ur = Roman Urdu
TONES = ["friendly", "firm", "final"]