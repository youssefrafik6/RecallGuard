"""Compatibility entry point for existing Streamlit deployments."""

from pathlib import Path
from runpy import run_path


if __name__ == "__main__":
    run_path(str(Path(__file__).resolve().parent / "app" / "streamlit_app.py"), run_name="__main__")
