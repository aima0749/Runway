import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
from app.components.chat_ui import render_chat

st.set_page_config(page_title="Runway Radar: Ask CFO", page_icon="💬")
st.title("💬 Ask your CFO")
render_chat()