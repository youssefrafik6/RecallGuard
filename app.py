"""Compatibility entry point for existing Streamlit deployments."""

import sys
from pathlib import Path
from runpy import run_path


if __name__ == "__main__":
    app_dir = Path(__file__).resolve().parent / "app"
    sys.path.insert(0, str(app_dir))
    try:
        run_path(str(app_dir / "streamlit_app.py"), run_name="__main__")
    finally:
        sys.path.remove(str(app_dir))
