"""Guardrail: every number, percentage and date in an answer must trace back to
a tool result, a tool argument, or the user's own words."""
import math
import re
from dataclasses import dataclass, field
from typing import Any

from config import HORIZON_DAYS, CHASE_SHIFT_DAYS, LATE_THRESHOLD_DAYS

# ---- policy knobs ----
SMALL_OK = {1.0, 2.0, 3.0}   # harmless counts like "top 2 invoices"
REL_TOL = 0.005              # amounts >= 1000 may differ by 0.5% (rounding)

MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sept": 9, "sep": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}
MONTH_PAT = "|".join(sorted(MONTHS, key=len, reverse=True))
MULT = {"k": 1e3, "lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5,
        "million": 1e6, "crore": 1e7}

ID_RE = re.compile(r"\b[A-Za-z]{1,5}[-_]?\d+\b")                       # INV3, C1
ISO_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
DMY_RE = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({MONTH_PAT})\b\.?(?:,?\s+(\d{{4}}))?", re.I)
MDY_RE = re.compile(rf"\b({MONTH_PAT})\b\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b(?:,?\s+(\d{{4}}))?", re.I)
ORD_RE = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)\b", re.I)
PTS_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:percentage\s+)?points?\b", re.I)
NUM_RE = re.compile(
    r"(?<!\w)(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?:\s*(%|percent\b|k\b|lakhs?\b|lacs?\b|million\b|crore\b))?", re.I)


@dataclass
class Item:
    kind: str      # date | ordinal | pct | num
    value: Any
    raw: str


@dataclass
class GuardrailResult:
    ok: bool
    checked: int
    unverified: list = field(default_factory=list)


def _extract(text: str) -> list:
    """Pull dates, ordinals, percentages and plain numbers out of text."""
    items = []
    t = ID_RE.sub(" ", text or "")

    def iso(m):
        items.append(Item("date", (int(m[1]), int(m[2]), int(m[3])), m[0])); return " "

    def dmy(m):
        items.append(Item("date", (int(m[3]) if m[3] else None,
                                   MONTHS[m[2].lower()], int(m[1])), m[0])); return " "

    def mdy(m):
        items.append(Item("date", (int(m[3]) if m[3] else None,
                                   MONTHS[m[1].lower()], int(m[2])), m[0])); return " "

    def ordn(m):
        items.append(Item("ordinal", int(m[1]), m[0])); return " "

    def pts(m):
        items.append(Item("pct", float(m[1]), m[0])); return " "

    def num(m):
        v = float(m[1].replace(",", ""))
        suf = (m[2] or "").lower()
        if suf in ("%", "percent"):
            items.append(Item("pct", v, m[0].strip()))
        else:
            items.append(Item("num", v * MULT.get(suf, 1), m[0].strip()))
        return " "

    for rx, fn in ((ISO_RE, iso), (DMY_RE, dmy), (MDY_RE, mdy),
                   (ORD_RE, ordn), (PTS_RE, pts), (NUM_RE, num)):
        t = rx.sub(fn, t)
    return items


class _Allowed:
    def __init__(self):
        self.nums = {float(HORIZON_DAYS), float(CHASE_SHIFT_DAYS), float(LATE_THRESHOLD_DAYS)}
        self.pcts, self.dates, self.days, self.years = set(), set(), set(), set()

    def add_item(self, it: Item):
        if it.kind == "date":
            y, m, d = it.value
            self.dates.add((y, m, d)); self.days.add(d)
            if y:
                self.years.add(y)
        elif it.kind == "ordinal":
            self.days.add(it.value)
        elif it.kind == "pct":
            self.pcts.add(int(round(it.value)))
        else:
            self.nums.add(abs(it.value))

    def add_text(self, s: str):
        for it in _extract(s):
            self.add_item(it)

    def add_obj(self, obj, key=""):
        if obj is None or isinstance(obj, bool):
            return
        if isinstance(obj, (int, float)):
            v = abs(float(obj))
            if math.isnan(v) or math.isinf(v):
                return
            if key.endswith("_pct"):
                self.pcts.add(int(round(v)))
            else:
                self.nums.add(v)
                if v <= 1:                      # fractions can be quoted as percentages
                    self.pcts.add(int(round(v * 100)))
                    self.pcts.add(int(math.floor(v * 100)))
        elif isinstance(obj, str):
            self.add_text(obj)
        elif isinstance(obj, dict):
            for k, v in obj.items():
                self.add_obj(v, str(k))
        elif isinstance(obj, (list, tuple)):
            self.nums.add(float(len(obj)))      # "3 invoices"
            for x in obj:
                self.add_obj(x, key)


def _close(a: float, b: float) -> bool:
    if b >= 1000:
        tol = REL_TOL * b
    elif b >= 10:
        tol = 0.5
    else:
        tol = 0.006
    return abs(a - b) <= tol


def _item_ok(it: Item, A: _Allowed) -> bool:
    if it.kind == "date":
        y, m, d = it.value
        return any(am == m and ad == d and (y is None or y == ay) for ay, am, ad in A.dates)
    if it.kind == "ordinal":
        return it.value in A.days
    if it.kind == "pct":
        return int(round(it.value)) in A.pcts or int(math.floor(it.value)) in A.pcts
    v = it.value
    if v in SMALL_OK:
        return True
    if v == int(v) and int(v) in A.years:
        return True
    return any(_close(v, a) for a in A.nums)


def verify_answer(answer: str, trace: list, extra_texts=()) -> GuardrailResult:
    """trace: [{'tool','args','result'}]. extra_texts: user message, history, as-of date."""
    A = _Allowed()
    for t in trace:
        A.add_obj(t.get("args") or {})
        A.add_obj(t.get("result") or {})
    for s in extra_texts:
        A.add_text(s or "")
    items = _extract(answer)
    bad = [it.raw for it in items if not _item_ok(it, A)]
    return GuardrailResult(ok=not bad, checked=len(items), unverified=bad)


# ---------------- language + template fallback ----------------
_UR_WORDS = {"mera", "meri", "mere", "kab", "kya", "kaise", "kitna", "kitne", "paisa", "paise",
             "paisay", "hai", "hain", "hoga", "hogi", "mujhe", "aap", "apna", "apni", "nahi",
             "nahin", "pehle", "pehla", "karo", "karna", "chahiye", "bhai", "wala", "wali",
             "ko", "ka", "ki", "ke", "mein", "khatam", "rakh", "rakhna", "sakta", "sakti",
             "batao", "bata", "dikhao", "bhejo", "likho", "kaun", "kis", "agar", "toh"}


def detect_lang(text: str) -> str:
    words = re.findall(r"[a-z]+", (text or "").lower())
    return "ur" if sum(w in _UR_WORDS for w in words) >= 2 else "en"


def fallback_answer(trace: list, lang: str = "en") -> str:
    """Answer built only from the last tool result. No LLM text, so no invented numbers."""
    ur = lang == "ur"
    if not trace:
        return ("Main sirf aap ke business ke cash flow ke baare mein madad kar sakta hoon. "
                "Aap apna runway, invoices ya hiring ke baare mein pooch sakte hain." if ur else
                "I can only report numbers from your cash-flow data. Try asking about your "
                "runway, which invoice to chase, or a hiring decision.")
    tool, r = trace[-1]["tool"], trace[-1]["result"]
    if "error" in r:
        return f"Maazrat, yeh kaam nahi ho saka: {r['error']}" if ur else f"Sorry, that did not work: {r['error']}"
    if tool == "get_runway":
        d = r.get("likely_cash_out_date")
        if ur:
            s = f"Aap ke paise khatam hone ka {r['p_shortfall_pct']}% khatra hai (status: {r['status']})."
            return s + (f" Cash khatam hone ki mumkin tareekh {d} hai." if d else "")
        s = f"There is a {r['p_shortfall_pct']}% chance you run short of cash (status: {r['status']})."
        return s + (f" The likely cash-out date is {d}." if d else
                    f" No cash-out is expected in the next {r['horizon_days']} days.")
    if tool == "run_what_if":
        if ur:
            return (f"Is faislay se cash kam par jane ka khatra {r['p_before_pct']}% se "
                    f"{r['p_after_pct']}% ho jayega (status: {r['status_after']}).")
        return (f"Your shortfall risk would change from {r['p_before_pct']}% to "
                f"{r['p_after_pct']}% (status after: {r['status_after']}).")
    if tool == "rank_chase" and r.get("ranked"):
        t = r["ranked"][0]
        if ur:
            return (f"Sab se pehle {t['invoice_id']} ({t['client']}) se paisa wasool karein. "
                    f"Khatra {t['p_before_pct']}% se kam ho kar {t['p_after_pct']}% reh jata hai.")
        return (f"Chase {t['invoice_id']} ({t['client']}) first. Your shortfall risk drops "
                f"from {t['p_before_pct']}% to {t['p_after_pct']}%.")
    if tool == "get_invoice":
        if ur:
            return (f"Invoice {r['invoice_id']} ka payment {r['predicted_pay_date']} ke qareeb aane ka "
                    f"andaza hai (range {r['range_start']} se {r['range_end']}). "
                    f"{r['p_more_than_14_days_late_pct']}% imkaan hai ke 14 din se zyada late ho.")
        return (f"Invoice {r['invoice_id']} is expected around {r['predicted_pay_date']} "
                f"(range {r['range_start']} to {r['range_end']}), with a "
                f"{r['p_more_than_14_days_late_pct']}% chance of being more than 14 days late.")
    if tool == "draft_reminder":
        return r.get("message", "")
    if tool == "add_invoice_from_text":
        return ("Maine invoice samajh liya hai. Save karne se pehle please confirm karein." if ur else
                "I extracted an invoice. Please confirm it before it is saved.")
    return "Please check the tool details below." if not ur else "Tafseelat neeche dekh lein."