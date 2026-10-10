from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def test_app_loads_all_tabs():
    at = AppTest.from_file(APP, default_timeout=90).run()
    assert not at.exception
    assert len(at.tabs) == 5


def test_what_if_button_runs():
    at = AppTest.from_file(APP, default_timeout=90).run()
    btn = next(b for b in at.button if b.label == "Run what-if")
    btn.click().run()
    assert not at.exception
    assert "whatif_result" in at.session_state