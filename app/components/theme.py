"""Visual styling only. No logic. Remove apply_theme() to get the old look back."""
import streamlit as st

CSS = """
<style>
.block-container {padding-top: 1.6rem; max-width: 1250px;}
#MainMenu, footer {visibility: hidden;}

.rr-hero {
  background: linear-gradient(135deg, #0f766e 0%, #1d4ed8 60%, #7c3aed 100%);
  border-radius: 20px; padding: 26px 32px; margin-bottom: 1rem;
  box-shadow: 0 10px 30px rgba(29,78,216,.28);
}
.rr-hero h1 {margin: 0; font-size: 2.1rem; color: #fff; padding: 0;}
.rr-hero p {margin: .35rem 0 0; color: #dbeafe; font-size: 1.05rem;}
.rr-chip {display: inline-block; margin-top: .8rem; padding: 4px 12px; border-radius: 999px;
  background: rgba(255,255,255,.16); color: #fff; font-size: .82rem;}

[data-testid="stMetric"] {
  background: #131c2e; border: 1px solid #22304a; border-radius: 16px;
  padding: 16px 18px; box-shadow: 0 4px 14px rgba(0,0,0,.25);
}
[data-testid="stMetricValue"] {font-weight: 700;}

.stButton > button, .stDownloadButton > button, .stLinkButton > a {
  border-radius: 12px; font-weight: 600; transition: all .15s ease;
}
.stButton > button:hover, .stLinkButton > a:hover {
  transform: translateY(-1px); box-shadow: 0 6px 16px rgba(34,211,166,.25);
  border-color: #22d3a6;
}

button[role="tab"] {border-radius: 999px; padding: 6px 18px; margin-right: 6px;}
button[role="tab"][aria-selected="true"] {background: #22d3a622;}

[data-testid="stAlert"] {border-radius: 14px;}
[data-testid="stDataFrame"], [data-testid="stExpander"] {border-radius: 14px; overflow: hidden;}
[data-testid="stChatMessage"] {
  background: #131c2e; border: 1px solid #22304a; border-radius: 16px;
}
[data-testid="stSidebar"] {border-right: 1px solid #22304a;}
</style>
"""


def apply_theme():
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title, subtitle, chip=""):
    chip_html = f'<span class="rr-chip">{chip}</span>' if chip else ""
    st.markdown(f'<div class="rr-hero"><h1>{title}</h1><p>{subtitle}</p>{chip_html}</div>',
                unsafe_allow_html=True)