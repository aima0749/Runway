"""Offline demo support. Cached answers are only served if their numbers still match the live engine."""
import difflib
import json
import re
import sys
import time
from datetime import datetime

from config import ROOT
from src.guardrail import verify_answer
from src.tools import call_tool, get_engine

CACHE_PATH = ROOT / "app" / "cache" / "demo_responses.json"

DEMO_QUESTIONS = [
    "What is the chance I run out of cash?",
    "Which invoice should I chase first?",
    "Which client should I call first?",
    "Can I afford to hire someone at Rs 60,000 per month starting next month?",
    "What happens if I buy equipment for Rs 200,000 next week?",
    "Mera paisa kab khatam hoga?",
    "Mujhe sab se pehle kis client se paisa lena chahiye?",
    "Kya main agle mahine se 60,000 per month par kisi ko hire kar sakta hoon?",
]


def normalize(q):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", (q or "").lower())).strip()


def _canon(x):
    return json.loads(json.dumps(x, sort_keys=True, default=str))


def load_cache():
    if not CACHE_PATH.exists():
        return {}
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _find(question, cache):
    key = normalize(question)
    if key in cache:
        return cache[key]
    digits = re.findall(r"\d+", key)
    for cand in difflib.get_close_matches(key, list(cache), n=3, cutoff=0.88):
        if re.findall(r"\d+", cand) == digits:          # never reuse an answer for different numbers
            return cache[cand]
    return None


def _fresh_trace(entry):
    new = []
    for t in entry["tool_trace"]:
        res = call_tool(t["tool"], t["args"])
        if _canon(res) != _canon(t["result"]):
            return None
        new.append({"tool": t["tool"], "args": t["args"], "result": res})
    return new


def lookup(question):
    """-> (answer dict or None, status in 'hit' | 'miss' | 'stale')"""
    entry = _find(question, load_cache())
    if not entry:
        return None, "miss"
    trace = _fresh_trace(entry)
    if trace is None:
        return None, "stale"
    extra = [question, entry["question"], get_engine().as_of.strftime("%Y-%m-%d")]
    g = verify_answer(entry["answer"], trace, extra)
    if not g.ok:
        return None, "stale"
    return ({"answer": entry["answer"], "tool_trace": trace, "steps": entry.get("steps", 1),
             "ok": True, "llm": "cache", "verified": True, "numbers_checked": g.checked,
             "unverified": [], "retried": False, "fallback": False, "source": "cache"}, "hit")


def ask(question, history=None, llm=None, offline=False):
    """Single entry point for the UI. Never raises."""
    from src.agent import run_agent, MockLLM, get_llm
    history = history or []
    if offline:
        hit, why = lookup(question)
        if hit:
            return hit
        out = run_agent(question, history, MockLLM())
        out["source"], out["cache_status"] = "mock", why
        return out
    llm = llm or get_llm()
    out = run_agent(question, history, llm)
    out["source"] = "mock" if out.get("llm") == "mock" else "llm"
    if not out.get("ok"):                                # API down, rate limit, step limit
        hit, _ = lookup(question)
        if hit:
            return hit
        alt = run_agent(question, history, MockLLM())
        if alt.get("ok") and alt["tool_trace"]:
            alt["source"] = "mock"
            return alt
    return out


def record(questions=None):
    from src.agent import run_agent, get_llm
    llm = get_llm()
    if llm.name == "mock":
        raise SystemExit("Recording needs the real LLM. Check .env and your network.")
    cache = {}
    for q in (questions or DEMO_QUESTIONS):
        out = run_agent(q, [], llm)
        good = (out["ok"] and out["verified"] and not out["fallback"] and out["tool_trace"]
                and not any("error" in t["result"] for t in out["tool_trace"]))
        print(("KEEP " if good else "SKIP ") + q)
        if good:
            cache[normalize(q)] = {"question": q, "answer": out["answer"], "tool_trace": out["tool_trace"],
                                   "steps": out["steps"], "model": getattr(llm, "model", "?"),
                                   "recorded_at": datetime.now().isoformat(timespec="seconds")}
        time.sleep(1.5)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"Saved {len(cache)} answers to {CACHE_PATH}")


def check():
    for q in DEMO_QUESTIONS:
        _, why = lookup(q)
        print(f"{why.upper():6} {q}")


if __name__ == "__main__":
    {"record": record}.get(sys.argv[1] if len(sys.argv) > 1 else "check", check)()