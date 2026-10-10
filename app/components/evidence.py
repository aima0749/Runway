"""Evidence tab. Reads saved evaluation outputs. Nothing here is hard-coded."""
import json

import pandas as pd
import streamlit as st

from config import ROOT

RESULTS = ROOT / "eval" / "results"
AGENT_FILES = {"agent_eval.json", "agent_eval.md"}


def _pct(v):
    return "n/a" if v is None else f"{v}%"


def _agent_section():
    st.subheader("AI assistant test")
    p = RESULTS / "agent_eval.json"
    if not p.exists():
        st.info("The assistant test has not been run yet. Run: python -m src.eval_agent")
        return
    data = json.loads(p.read_text(encoding="utf-8"))
    s = data["summary"]
    st.caption(f"Run on {s.get('when')} with {s.get('llm')} / {s.get('model')}, "
               f"{s.get('n_questions')} questions in English and Roman Urdu.")
    c = st.columns(4)
    c[0].metric("Right tool chosen", _pct(s.get("tool_choice_accuracy_pct")))
    c[1].metric("Out-of-scope refused", _pct(s.get("out_of_scope_refusal_pct")))
    c[2].metric("Clean on first try", _pct(s.get("clean_first_try_pct")))
    c[3].metric("Answers with untraceable numbers", s.get("final_answers_with_untraceable_numbers", "n/a"))
    st.caption(f"Retries: {s.get('retried')} · template fallbacks: {s.get('template_fallbacks')} · "
               f"API failures: {s.get('api_failures')} · average {s.get('avg_seconds')} s per answer")
    cat = s.get("by_category_tool_accuracy_pct") or {}
    if cat:
        st.dataframe(pd.DataFrame({"Category": list(cat), "Tool accuracy (%)": list(cat.values())}),
                     hide_index=True)
    with st.expander("Every question and result"):
        rows = [{k: r[k] for k in ("id", "category", "question", "called", "tool_ok",
                                   "verified", "retried", "fallback")} for r in data["rows"]]
        st.dataframe(pd.DataFrame(rows), hide_index=True)


def _backtest_section():
    st.subheader("Payment-timing and shortfall backtest")
    st.caption("Test design: opening cash is set so the on-paper (due-date) projection barely stays "
               "positive. This checks whether the model sees risk the paper view misses. "
               "It is not a general accuracy claim.")
    files = sorted(p for p in RESULTS.glob("*")
                   if p.is_file() and p.name not in AGENT_FILES and p.name != ".gitkeep")
    if not files:
        st.info("No backtest outputs found in eval/results. Run the backtest in src/evaluate.py first.")
        return
    for p in files:
        st.markdown(f"**{p.stem.replace('_', ' ')}**")
        if p.suffix == ".csv":
            st.dataframe(pd.read_csv(p), hide_index=True)
        elif p.suffix == ".md":
            st.markdown(p.read_text(encoding="utf-8"))
        elif p.suffix in (".png", ".jpg", ".jpeg"):
            st.image(str(p))
        elif p.suffix == ".json":
            st.json(json.loads(p.read_text(encoding="utf-8")), expanded=False)


def _how_it_works():
    st.subheader("How it works")
    st.markdown(
        "1. **Predict:** for each unpaid invoice, a survival model learns how that client pays and "
        "estimates when the money will really arrive.\n"
        "2. **Forecast:** the engine simulates 1,000 possible futures and reports a likely, best and "
        "worst case, plus the chance of running short.\n"
        "3. **Decide:** it ranks which invoice to chase first and answers what-if questions.\n"
        "4. **Act:** it drafts reminder messages in English and Roman Urdu.\n"
        "5. **Ask:** an AI assistant chooses a tool and explains the result in plain language. "
        "It never calculates. Every number in its answer is checked against the engine's output.\n\n"
        "**Limits:** committed money only (existing invoices and known expenses). New clients with "
        "fewer than 3 past invoices have wider ranges. All data is fictional.")


def render_evidence():
    _agent_section()
    st.divider()
    _backtest_section()
    st.divider()
    _how_it_works()