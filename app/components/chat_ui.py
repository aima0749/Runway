"""Ask CFO chat tab. Dev B's streamlit_app.py only needs:
    from app.components.chat_ui import render_chat
    with chat_tab: render_chat()
"""
import json
import streamlit as st

from src.agent import get_llm
from src.offline import ask
from src.tools import save_invoice

EXAMPLES = [
    "What is the chance I run out of cash?",
    "Which invoice should I chase first?",
    "Can I afford to hire someone at Rs 60,000 per month starting next month?",
    "Mera paisa kab khatam hoga?",
]


@st.cache_resource
def _llm():
    return get_llm()


def _badge(m):
    if not m.get("ok", True):
        return "⚠️ The assistant could not finish this one"
    n = m.get("numbers_checked", 0)
    if m.get("source") == "cache":
        return f"📦 Pre-recorded answer. Numbers re-checked against the live engine ({n} verified)"
    if m.get("source") == "mock":
        return "🔌 Built-in offline answer. Numbers come straight from the engine"
    if m.get("fallback"):
        return "🟡 Template answer: the AI's wording failed the number check, so this was built from the data"
    if n == 0:
        return "ℹ️ No figures in this answer"
    return f"✅ {n} numbers verified against the engine"


def _show(m, idx):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] != "assistant":
            return
        st.caption(_badge(m))
        for t in m.get("trace", []):
            if t["tool"] == "add_invoice_from_text" and t["result"].get("can_save"):
                key = f"saved_{idx}"
                if key in st.session_state["saved"]:
                    st.success("Invoice saved. Your forecast has been updated.")
                elif st.button("✅ Save this invoice", key=f"btn_{idx}"):
                    res = save_invoice(t["result"]["parsed"])
                    if "error" in res:
                        st.error(res["error"])
                    else:
                        st.session_state["saved"].add(key)
                        st.rerun()
        if m.get("trace"):
            with st.expander("How I got this (tools used)"):
                for t in m["trace"]:
                    st.markdown(f"**{t['tool']}**  `{json.dumps(t['args'], default=str)}`")
                    st.json(t["result"], expanded=False)


def render_chat():
    st.session_state.setdefault("chat", [])
    st.session_state.setdefault("saved", set())

    c1, c2 = st.columns([3, 1])
    offline = c1.toggle("Offline demo mode (no AI service)", value=False)
    if c2.button("Clear chat"):
        st.session_state["chat"], st.session_state["saved"] = [], set()
        st.rerun()
    st.caption("Every number comes from the forecasting engine. The AI only picks the tool and explains it.")

    if not st.session_state["chat"]:
        cols = st.columns(2)
        for i, q in enumerate(EXAMPLES):
            if cols[i % 2].button(q, key=f"ex_{i}", use_container_width=True):
                st.session_state["pending"] = q

    for i, m in enumerate(st.session_state["chat"]):
        _show(m, i)

    prompt = st.chat_input("Ask about your cash flow (English or Roman Urdu)")
    prompt = prompt or st.session_state.pop("pending", None)
    if not prompt:
        return

    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state["chat"][-6:]]
    st.session_state["chat"].append({"role": "user", "content": prompt})
    _show(st.session_state["chat"][-1], len(st.session_state["chat"]) - 1)
    with st.spinner("Checking your numbers..."):
        out = ask(prompt, history, _llm(), offline=offline)
    st.session_state["chat"].append({
        "role": "assistant", "content": out["answer"], "trace": out["tool_trace"],
        "ok": out.get("ok", True), "source": out.get("source"), "fallback": out.get("fallback", False),
        "numbers_checked": out.get("numbers_checked", 0)})
    _show(st.session_state["chat"][-1], len(st.session_state["chat"]) - 1)