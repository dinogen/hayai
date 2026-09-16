"""
Central path and parameter constants for model_research.
Import this module everywhere — never hardcode paths.
"""
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────

ROOT         = Path(__file__).parent
CONFIG_DIR   = ROOT / "config"
DATA_DIR     = ROOT / "data" / "cache"
FEATURES_DIR = ROOT / "data" / "features"
MODELS_DIR   = ROOT / "data" / "models"
RESULTS_DIR  = ROOT / "experiments" / "results"

UNIVERSE_FILE  = CONFIG_DIR / "universe.yaml"
REGISTRY_FILE  = ROOT / "experiments" / "registry.yaml"

# ── Data ─────────────────────────────────────────────────────────────────────

DATA_START = "2015-01-01"   # earliest date to download
CACHE_MAX_AGE_DAYS = 1      # re-download if parquet is older than this

# ── Feature window ───────────────────────────────────────────────────────────

DEFAULT_WINDOW = 20         # trading days per sample

# ── Targets ──────────────────────────────────────────────────────────────────

TARGET_HORIZONS = [5, 10, 15]   # forward return horizons (trading days)
TARGET_CLIP     = 3.0           # clip normalized return to [-3, +3]

# ── Walk-forward split ───────────────────────────────────────────────────────

WF_TRAIN_YEARS  = 2     # years of training data per fold
WF_VAL_MONTHS   = 6     # months of validation data per fold
WF_TEST_MONTHS  = 6     # months of test data per fold
WF_EMBARGO_DAYS = 20    # gap between train/val and val/test to avoid leakage

# ── Training ─────────────────────────────────────────────────────────────────

DEFAULT_SEED       = 42
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS     = 100
DEFAULT_PATIENCE   = 10     # EarlyStopping patience
DEFAULT_LR         = 1e-4   # 1e-3 caused epoch=1 collapse (diagnostic EXP-002)
