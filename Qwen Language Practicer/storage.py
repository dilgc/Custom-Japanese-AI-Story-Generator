import json
import os
from pathlib import Path
from datetime import datetime
from config import PROFILE_PATH


def save_profile(profile: dict):
    profile["last_updated"] = datetime.now().isoformat()
    path = Path(PROFILE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")


def load_profile() -> dict | None:
    path = Path(PROFILE_PATH)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None
