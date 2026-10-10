from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app" / "ask_cfo_demo.py")


def test_chat_page_loads():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception

def test_offline_question_gets_an_answer():
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.toggle[0].set_value(True).run()
    at.chat_input[0].set_value("What is my cash risk?").run()
    assert not at.exception
    assert len(at.chat_message) >= 2