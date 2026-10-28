"""
Hugging Face Spaces entrypoint.
Executes the main ResearchLens Streamlit application.
"""
import runpy
from pathlib import Path

try:
    import torch
    if not hasattr(torch, "accelerator"):
        class _DummyAccelerator:
            @staticmethod
            def current_accelerator():
                return None
        torch.accelerator = _DummyAccelerator()
except ImportError:
    torch = None  # type: ignore[assignment]

_target_app = Path(__file__).resolve().parent / "streamlit_app.py"
runpy.run_path(str(_target_app), run_name="__main__")
