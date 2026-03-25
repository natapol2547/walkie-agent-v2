"""Flask application entrypoint."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Redirect all model downloads to the project-local models/ directory.
# This must happen BEFORE any service or route module is imported, because
# route singletons are created at import time and trigger model downloads.
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent
_MODELS_DIR = _ROOT / "models"
_HF_CACHE = _MODELS_DIR / "huggingface"
_ULTRALYTICS_DIR = _MODELS_DIR / "ultralytics"

_HF_CACHE.mkdir(parents=True, exist_ok=True)
_ULTRALYTICS_DIR.mkdir(parents=True, exist_ok=True)

# HuggingFace Hub cache — covers Whisper (faster-whisper), CLIP, PaliGemma,
# and the YOLO Object365 model which is fetched via hf_hub_download.
os.environ.setdefault("HF_HUB_CACHE", str(_HF_CACHE))

# Ultralytics weights dir — covers YOLO Pose auto-download.
try:
    from ultralytics import settings as _ult_settings
    _ult_settings.update({"weights_dir": str(_ULTRALYTICS_DIR)})
except Exception:
    pass

# ---------------------------------------------------------------------------
# Deferred import: create_app registers blueprints which instantiate
# and load all model singletons.
# ---------------------------------------------------------------------------
from api import create_app  # noqa: E402

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
