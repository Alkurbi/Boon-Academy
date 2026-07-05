import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "outputs"))
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
MODEL = os.environ.get("MODEL", "claude-haiku-4-5")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "anthropic/claude-haiku-4.5")
RUN_DATE = os.environ.get("RUN_DATE", "2025-10-14")

PASS_SCORE = 60
SESSION_MAX_MIN = 90
PRACTICE_CAP = 60
QUIZ2_DATE = "2025-10-20"
ACTIONS_PER_DAY = 5
