"""Agent tools. Pure functions, JSON in / JSON out. The LLM never calculates; it calls these."""
from dataclasses import asdict
import numpy as np
import pandas as pd

from config import SEED, N_SIMS, CHASE_TOP_N, LANGS, TONES
from src.generator import generate_business
from src.features import build_features
from src.delay_model import DelayModel
from src.simulate import simulate_cash
from src import decisions
from src.messages import draft_reminder as _draft


# ---------- Engine: builds the pipeline once ----------
class Engine:
    def __init__(self, dataset=None):
        self.ds = dataset or generate_business()
        self.rebuild()

    def rebuild(self):
        ds = self.ds
        self.as_of = pd.Timestamp(ds.settings.as_of_date)
        inv = ds.invoices.copy()
        for c in ("issue_date", "due_date", "paid_date"):
            inv[c] = pd.to_datetime(inv[c])
        self.inv = inv

        paid = inv["paid_date"].notna()
        age = (self.as_of - inv["issue_date"]).dt.days.clip(lower=0)
        T = np.where(paid, (inv["paid_date"] - inv["issue_date"]).dt.days, age)
        E = paid.astype(int).values

        X_all = build_features(inv)
        self.model = DelayModel().fit(X_all, T, E)

        self.open_inv = inv[~paid].reset_index(drop=True)
        self.s_obs = (self.as_of - self.open_inv["issue_date"]).dt.days.clip(lower=0).values
        X_open = build_features(self.open_inv)
        rng = np.random.default_rng(SEED)
        self.samples = self.model.sample(X_open, self.s_obs, N_SIMS, rng)

        self.sim = simulate_cash(self.open_inv, self.samples, ds.expenses,
                                 ds.settings.opening_cash_pkr,
                                 as_of=self.as_of, s_obs=self.s_obs,
                                 buffer=ds.settings.min_buffer_pkr)

    def day_to_date(self, day_idx):
        return (self.as_of + pd.Timedelta(days=int(day_idx) + 1)).strftime("%Y-%m-%d")


_engine = None

def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = Engine()
    return _engine

def set_engine(dataset):
    """Swap the dataset (scenario picker / CSV upload)."""
    global _engine
    _engine = Engine(dataset)
    return _engine


def _pct(p):  # 0.683 -> 68
    return int(round(float(p) * 100))


# ---------- Tool 1 ----------
def get_runway() -> dict:
    e = get_engine()
    s = e.sim
    below_p50 = np.where(s.p50 < 0)[0]
    below_p10 = np.where(s.p10 < 0)[0]
    return {
        "p_shortfall": round(float(s.p_shortfall), 2),
        "p_shortfall_pct": _pct(s.p_shortfall),
        "status": decisions.status(s.p_shortfall),
        "likely_cash_out_date": e.day_to_date(below_p50[0]) if len(below_p50) else None,
        "earliest_plausible_cash_out_date": e.day_to_date(below_p10[0]) if len(below_p10) else None,
        "opening_cash_pkr": round(float(e.ds.settings.opening_cash_pkr)),
        "min_cash_p50_pkr": round(float(np.median(s.min_cash_dist))),
        "horizon_days": int(s.paths.shape[1]),
        "assumption": "Committed money only: existing invoices and known expenses.",
    }

# ---------- Tool 2 ----------
WHAT_IF_TYPES = {"hire", "add_expense", "delay_client", "weekly_sales"}

def run_what_if(change: dict) -> dict:
    t = (change or {}).get("type")
    if t not in WHAT_IF_TYPES:
        return {"error": f"Unknown change type '{t}'. Use one of {sorted(WHAT_IF_TYPES)}."}
    res = decisions.run_what_if(get_engine(), change)
    res = dict(res)
    res["change"] = change
    res["p_before_pct"] = _pct(res["p_before"])
    res["p_after_pct"] = _pct(res["p_after"])
    res["status_after"] = decisions.status(res["p_after"])
    return res


# ---------- Tool 3 ----------
def rank_chase(top_n: int = CHASE_TOP_N) -> dict:
    top_n = max(1, min(int(top_n), 10))
    rows = decisions.rank_chase(get_engine(), top_n)
    out = []
    for r in rows:
        d = asdict(r)
        d["p_before_pct"] = _pct(r.p_before)
        d["p_after_pct"] = _pct(r.p_after)
        out.append(d)
    return {"ranked": out, "note": "Each row assumes that invoice is collected 10 days earlier."}


# ---------- Tool 4 ----------
def get_invoice(invoice_id: str) -> dict:
    e = get_engine()
    m = e.open_inv.index[e.open_inv["invoice_id"] == invoice_id]
    if len(m) == 0:
        return {"error": f"No open invoice with id '{invoice_id}'."}
    i = m[0]
    row = e.open_inv.loc[i]
    t = e.samples[:, i]
    q10, q50, q90 = np.percentile(t, [10, 50, 90])
    to_date = lambda d: (row["issue_date"] + pd.Timedelta(days=float(d))).strftime("%Y-%m-%d")
    terms = (row["due_date"] - row["issue_date"]).days
    return {
        "invoice_id": invoice_id,
        "client_id": row["client_id"],
        "amount_pkr": float(row["amount_pkr"]),
        "due_date": row["due_date"].strftime("%Y-%m-%d"),
        "predicted_pay_date": to_date(q50),
        "range_start": to_date(q10),
        "range_end": to_date(q90),
        "p_more_than_14_days_late": round(float((t > terms + 14).mean()), 2),
        "p_more_than_14_days_late_pct": _pct((t > terms + 14).mean()),
    }


# ---------- Tool 5 ----------
def draft_reminder(invoice_id: str, lang: str = "en", tone: str = "friendly") -> dict:
    if lang not in LANGS:
        return {"error": f"lang must be one of {LANGS}"}
    if tone not in TONES:
        return {"error": f"tone must be one of {TONES}"}
    inv = get_invoice(invoice_id)
    if "error" in inv:
        return inv
    e = get_engine()
    name_row = e.ds.clients[e.ds.clients["client_id"] == inv["client_id"]]
    client = name_row["name"].iloc[0] if len(name_row) else inv["client_id"]
    text = _draft(client, inv, tone, lang)
    return {"invoice_id": invoice_id, "lang": lang, "tone": tone, "message": text}


# ---------- Tool 6 (real logic comes in Step 6) ----------
def add_invoice_from_text(text: str) -> dict:
    """STUB: parser.py arrives in Step 6. Never saves; the user must confirm first."""
    return {"status": "needs_confirmation",
            "parsed": {"client": None, "amount_pkr": None, "due_date": None},
            "note": "Parser not implemented yet."}


# ---------- Registry + schemas for the LLM ----------
TOOLS = {
    "get_runway": get_runway,
    "run_what_if": run_what_if,
    "rank_chase": rank_chase,
    "get_invoice": get_invoice,
    "draft_reminder": draft_reminder,
    "add_invoice_from_text": add_invoice_from_text,
}

TOOL_SCHEMAS = [
    {"name": "get_runway",
     "description": "Get the probability of running out of cash, status, and likely cash-out dates.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "run_what_if",
     "description": "Simulate a business decision and return shortfall risk before and after.",
     "parameters": {"type": "object", "properties": {
         "change": {"type": "object", "description":
             "One of: {type:'hire', monthly:number, start:'YYYY-MM-DD'}, "
             "{type:'add_expense', amount:number, due_date:'YYYY-MM-DD'}, "
             "{type:'delay_client', client_id:string, days:number}, "
             "{type:'weekly_sales', amount:number}"}},
         "required": ["change"]}},
    {"name": "rank_chase",
     "description": "Rank open invoices by how much collecting them earlier reduces shortfall risk.",
     "parameters": {"type": "object", "properties": {
         "top_n": {"type": "integer", "description": "How many to return, 1 to 10"}}}},
    {"name": "get_invoice",
     "description": "Get predicted pay date, range, and chance of being over 14 days late for one open invoice.",
     "parameters": {"type": "object", "properties": {
         "invoice_id": {"type": "string"}}, "required": ["invoice_id"]}},
    {"name": "draft_reminder",
     "description": "Draft a payment reminder message for an invoice.",
     "parameters": {"type": "object", "properties": {
         "invoice_id": {"type": "string"},
         "lang": {"type": "string", "enum": ["en", "ur"]},
         "tone": {"type": "string", "enum": ["friendly", "firm", "final"]}},
         "required": ["invoice_id"]}},
    {"name": "add_invoice_from_text",
     "description": "Extract an invoice from messy pasted text. Returns a proposal the user must confirm.",
     "parameters": {"type": "object", "properties": {
         "text": {"type": "string"}}, "required": ["text"]}},
]


def call_tool(name: str, args: dict | None = None) -> dict:
    """Single entry point for agent.py. Never raises; returns {'error': ...} instead."""
    fn = TOOLS.get(name)
    if fn is None:
        return {"error": f"Unknown tool '{name}'."}
    try:
        return fn(**(args or {}))
    except TypeError as ex:
        return {"error": f"Bad arguments: {ex}"}
    except Exception as ex:
        return {"error": f"Tool failed: {ex}"}