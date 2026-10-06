"""
Hugging Face Spaces entrypoint.
Executes the main ResearchLens Streamlit application.
"""
import runpy
from pathlib import Path

_target_app = Path(__file__).resolve().parent / "streamlit_app.py"
runpy.run_path(str(_target_app), run_name="__main__")
