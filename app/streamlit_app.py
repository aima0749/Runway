"""Runway Radar. Run: streamlit run app/streamlit_app.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import urllib.parse

import pandas as pd
import streamlit as st

from config import GREEN_MAX, AMBER_MAX
from src import tools
from src.schemas import Settings
from app.components.chat_ui import render_chat
from app.components.charts import fan_chart, shortfall_chart
from app.components.data_tools import TEMPLATE_NAMES, template_csv, validate_upload
from app.components.evidence import render_evidence
from app.components.theme import apply_theme, hero

st.set_page_config(page_title="Runway Radar", page_icon="📡", layout="wide")
apply_theme()

AYESHA = "Ayesha Studio (scripted demo)"
EASY = "Synthetic business (easy)"
HARD = "Synthetic business (hard)"
UPLOAD = "Upload my own CSVs"
SCENARIOS = [AYESHA, EASY, HARD, UPLOAD]

# Policy knobs for reminder tone (not model outputs)
FINAL_AFTER_DAYS = 14
FIRM_AFTER_DAYS = 1
FIRM_IF_LATE_RISK = 0.5

WHAT_IF = {"Hire someone": "hire", "One-off expense": "add_expense",
           "A client pays late": "delay_client", "Extra sales every week": "weekly_sales"}


def rs(x):
    return ("-" if x < 0 else "") + f"Rs {abs(x):,.0f}"


def nice_date(s):
    return pd.Timestamp(s).strftime("%d %b %Y")


def reset_state():
    for k in ("chat", "saved", "pending", "whatif_result"):
        st.session_state.pop(k, None)


def base_dataset(name):
    if name == AYESHA:
        return tools.default_dataset()
    return tools.synthetic_dataset(seed=0, mode="easy" if name == EASY else "hard")


def suggest_tone(days_overdue, p_late14):
    if days_overdue >= FINAL_AFTER_DAYS:
        return "final", "it is well overdue"
    if days_overdue >= FIRM_AFTER_DAYS or p_late14 >= FIRM_IF_LATE_RISK:
        return "firm", "it is overdue or likely to be very late"
    return "friendly", "it is not overdue and the client usually pays"


# ---------------- Sidebar ----------------
st.sidebar.title("📡 Runway Radar")
scenario = st.sidebar.selectbox("Scenario", SCENARIOS, key="scenario")

if scenario != UPLOAD and st.session_state.get("active_scenario") != scenario:
    with st.spinner("Building the forecast..."):
        tools.set_engine(base_dataset(scenario))
    st.session_state["active_scenario"] = scenario
    reset_state()

if scenario == UPLOAD:
    with st.sidebar.expander("Upload your files", expanded=True):
        cf = st.file_uploader("clients.csv", type="csv", key="up_c")
        inf = st.file_uploader("invoices.csv", type="csv", key="up_i")
        ef = st.file_uploader("expenses.csv", type="csv", key="up_e")
        up_asof = st.date_input("As-of date", key="up_asof")
        up_cash = st.number_input("Opening cash (Rs)", min_value=0.0, step=10000.0, key="up_cash")
        up_buf = st.number_input("Minimum buffer (Rs)", min_value=0.0, step=10000.0, key="up_buf")
        if st.button("Load my files", key="up_go"):
            if not (cf and inf and ef):
                st.error("Please upload all three files.")
            else:
                ds, errors = validate_upload(cf, inf, ef, Settings(up_asof.isoformat(), float(up_cash), float(up_buf)))
                if errors:
                    for er in errors:
                        st.error(er)
                else:
                    tools.set_engine(ds)
                    st.session_state["active_scenario"] = UPLOAD
                    reset_state()
                    st.rerun()
    if st.session_state.get("active_scenario") != UPLOAD:
        st.info("Upload your three files in the sidebar to begin. Until then the previous scenario is shown.")

e = tools.get_engine()
s = e.ds.settings
with st.sidebar.form(f"settings_{scenario}"):
    as_of = st.date_input("As-of date (today in the data)", value=pd.Timestamp(s.as_of_date).date(),
                          key=f"asof_{scenario}")
    cash = st.number_input("Opening cash (Rs)", value=float(s.opening_cash_pkr), min_value=0.0,
                           step=10000.0, key=f"cash_{scenario}")
    buffer = st.number_input("Minimum buffer (Rs)", value=float(s.min_buffer_pkr), min_value=0.0,
                             step=10000.0, key=f"buf_{scenario}")
    applied = st.form_submit_button("Apply")
if applied:
    e.ds.settings = Settings(as_of.isoformat(), float(cash), float(buffer))
    with st.spinner("Updating the forecast..."):
        tools.set_engine(e.ds)
    reset_state()
    st.rerun()

with st.sidebar.expander("CSV templates for your own data"):
    for name in TEMPLATE_NAMES:
        st.download_button(f"Download {name}", template_csv(name), file_name=name,
                           mime="text/csv", key=f"tpl_{name}")
    st.caption("One row per recurring expense, not one row per month. Leave paid_date blank "
               "for unpaid invoices.")
st.sidebar.caption("All data is fictional. Numbers come from the forecasting engine, never from the AI.")

# ---------------- Main ----------------
e = tools.get_engine()
runway = tools.call_tool("get_runway")
names = dict(zip(e.ds.clients["client_id"].astype(str), e.ds.clients["name"]))

hero("📡 Runway Radar", "Know the day the cash runs out, before it does.",
     chip=f"{st.session_state.get('active_scenario', scenario)} · as of {e.as_of:%d %b %Y} · "
          "committed money only")

for note in getattr(e, "data_notes", []):
    st.info(note)
tab_run, tab_inv, tab_act, tab_chat, tab_ev = st.tabs(
    ["Runway", "Invoices", "Actions", "Ask CFO", "Evidence"])

# ---- Runway ----
with tab_run:
    if "error" in runway:
        st.error(runway["error"])
    else:
        pct = runway["p_shortfall_pct"]
        text = f"**{pct}% chance you run short of cash** in the next {runway['horizon_days']} days"
        if runway["likely_cash_out_date"]:
            text += f", most likely around **{nice_date(runway['likely_cash_out_date'])}**"
        {"green": st.success, "amber": st.warning, "red": st.error}[runway["status"]](text + ".")
        st.caption(f"Status: green below {GREEN_MAX:.0%}, amber up to {AMBER_MAX:.0%}, red above. "
                   "These are policy settings, not model outputs.")
        c = st.columns(3)
        c[0].metric("Cash today", rs(runway["opening_cash_pkr"]))
        c[1].metric("Typical lowest balance", rs(runway["min_cash_p50_pkr"]))
        c[2].metric("Earliest plausible cash-out",
                    nice_date(runway["earliest_plausible_cash_out_date"])
                    if runway["earliest_plausible_cash_out_date"] else "None in the window")
        st.plotly_chart(fan_chart(e.sim, e.as_of, e.ds.settings.min_buffer_pkr))
        st.caption("The shaded band is the range between a bad and a good run of payments. "
                   "The solid line is the likely case.")
        with st.expander("When does the risk build up?"):
            st.plotly_chart(shortfall_chart(e.sim, e.as_of))

# ---- Invoices ----
with tab_inv:
    if len(e.open_inv) == 0:
        st.info("There are no open invoices.")
    else:
        rows = []
        for inv_id in e.open_inv["invoice_id"]:
            r = tools.get_invoice(str(inv_id))
            if "error" in r:
                continue
            rows.append({
                "Invoice": r["invoice_id"], "Client": names.get(str(r["client_id"]), r["client_id"]),
                "Amount (Rs)": r["amount_pkr"], "Due": r["due_date"],
                "Days overdue": max(0, (e.as_of - pd.Timestamp(r["due_date"])).days),
                "Predicted pay date": r["predicted_pay_date"],
                "Likely range (P10 to P90)": f"{r['range_start']} to {r['range_end']}",
                "Chance >14 days late": r["p_more_than_14_days_late_pct"]})
        df = pd.DataFrame(rows).sort_values("Amount (Rs)", ascending=False)
        c = st.columns(2)
        c[0].metric("Open invoices", len(df))
        c[1].metric("Total owed to you", rs(df["Amount (Rs)"].sum()))
        st.dataframe(df, hide_index=True, column_config={
            "Amount (Rs)": st.column_config.NumberColumn(format="%.0f"),
            "Chance >14 days late": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d%%")})
        st.caption("Clients with fewer than 3 past invoices use fallbacks, so their ranges are wider.")

# ---- Actions ----
with tab_act:
    st.subheader("1. Who should I chase first?")
    top_n = st.slider("How many invoices to show", 1, 10, 5, key="chase_n")
    ch = tools.call_tool("rank_chase", {"top_n": top_n})
    if "error" in ch:
        st.warning(ch["error"])
    elif not ch["ranked"]:
        st.info("There is nothing to chase.")
    else:
        top = ch["ranked"][0]
        st.info(f"Best move: chase **{top['invoice_id']}** ({top['client']}). "
                f"Your shortfall risk goes from {top['p_before_pct']}% to {top['p_after_pct']}%.")
        st.dataframe(pd.DataFrame([{
            "Rank": i + 1, "Invoice": r["invoice_id"], "Client": r["client"],
            "Amount (Rs)": r["amount_pkr"],
            "Risk if collected 10 days earlier": f"{r['p_before_pct']}% to {r['p_after_pct']}%",
            "Risk reduction (points)": int(round(r["gain"] * 100))} for i, r in enumerate(ch["ranked"])]),
            hide_index=True, column_config={"Amount (Rs)": st.column_config.NumberColumn(format="%.0f")})

    st.divider()
    st.subheader("2. What if...?")
    label = st.radio("What do you want to test?", list(WHAT_IF), horizontal=True, key="wi_kind")
    kind = WHAT_IF[label]
    default_day = (e.as_of + pd.Timedelta(30, unit="D")).date()
    change = None
    if kind == "hire":
        monthly = st.number_input("Monthly salary (Rs)", min_value=0.0, value=60000.0, step=5000.0, key="wi_hire")
        start = st.date_input("Starting on", value=default_day, key="wi_hire_start")
        change = {"type": "hire", "monthly": float(monthly), "start": start.isoformat()}
    elif kind == "add_expense":
        amount = st.number_input("Amount (Rs)", min_value=0.0, value=200000.0, step=10000.0, key="wi_exp")
        due = st.date_input("Due on", value=(e.as_of + pd.Timedelta(7, unit="D")).date(), key="wi_exp_due")
        change = {"type": "add_expense", "amount": float(amount), "due_date": due.isoformat()}
    elif kind == "delay_client":
        ids = sorted(set(e.open_inv["client_id"].astype(str))) if len(e.open_inv) else []
        if ids:
            cid = st.selectbox("Client", ids, format_func=lambda i: names.get(i, i), key="wi_cl")
            days = st.number_input("Pays this many days later", min_value=1, max_value=60, value=14, key="wi_days")
            change = {"type": "delay_client", "client_id": cid, "days": int(days)}
        else:
            st.info("There are no open invoices to delay.")
    else:
        amount = st.number_input("Extra sales per week (Rs)", min_value=0.0, value=50000.0, step=10000.0, key="wi_sales")
        change = {"type": "weekly_sales", "amount": float(amount)}

    if st.button("Run what-if", key="wi_run") and change:
        st.session_state["whatif_result"] = tools.call_tool("run_what_if", {"change": change})
    res = st.session_state.get("whatif_result")
    if res:
        if "error" in res:
            st.error(res["error"])
        else:
            c = st.columns(3)
            c[0].metric("Risk before", f"{res['p_before_pct']}%")
            c[1].metric("Risk after", f"{res['p_after_pct']}%",
                        delta=f"{res['p_after_pct'] - res['p_before_pct']} points", delta_color="inverse")
            c[2].metric("Status after", res["status_after"])
            if "min_cash_p50_before" in res and "min_cash_p50_after" in res:
                st.caption(f"Typical lowest balance: {rs(res['min_cash_p50_before'])} before, "
                           f"{rs(res['min_cash_p50_after'])} after.")
            if res.get("warning"):
                st.warning(res["warning"])

    st.divider()
    st.subheader("3. Draft a reminder")
    if len(e.open_inv) == 0:
        st.info("There are no open invoices to remind.")
    else:
        opts = e.open_inv.sort_values("amount_pkr", ascending=False)
        amt = dict(zip(opts["invoice_id"].astype(str), opts["amount_pkr"]))
        cl = dict(zip(opts["invoice_id"].astype(str), opts["client_id"].astype(str)))
        inv_id = st.selectbox("Invoice", list(amt), key="rem_inv",
                              format_func=lambda i: f"{i} · {names.get(cl[i], cl[i])} · {rs(amt[i])}")
        c = st.columns(2)
        lang_label = c[0].radio("Language", ["English", "Roman Urdu"], horizontal=True, key="rem_lang")
        tone_pick = c[1].selectbox("Tone", ["Auto (suggested)", "friendly", "firm", "final"], key="rem_tone")
        info = tools.get_invoice(inv_id)
        reason = ""
        tone = tone_pick
        if tone_pick.startswith("Auto"):
            tone, why = suggest_tone(max(0, (e.as_of - pd.Timestamp(info["due_date"])).days),
                                     info["p_more_than_14_days_late"])
            reason = f" (suggested because {why})"
        msg = tools.call_tool("draft_reminder", {"invoice_id": inv_id,
                                                 "lang": "ur" if lang_label == "Roman Urdu" else "en",
                                                 "tone": tone})
        if "error" in msg:
            st.error(msg["error"])
        else:
            st.caption(f"Tone: {tone}{reason}")
            st.code(msg["message"], language=None)
            st.link_button("Open in WhatsApp", "https://wa.me/?text=" + urllib.parse.quote(msg["message"]))
            if lang_label == "Roman Urdu":
                st.caption("Roman Urdu templates should be reviewed by a native speaker before real use.")

# ---- Ask CFO ----
with tab_chat:
    render_chat()

# ---- Evidence ----
with tab_ev:
    render_evidence()