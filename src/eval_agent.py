"""Agent evaluation: tool choice, number grounding, refusals. Quote only numbers from your own run."""
import argparse
import json
import time
from collections import defaultdict
from datetime import datetime

from config import ROOT
from src.agent import get_llm, run_agent
from src.guardrail import detect_lang
from src.tools import get_engine

QFILE = ROOT / "eval" / "agent_questions.json"
OUT_DIR = ROOT / "eval" / "results"


def _rate(vals):
    v = [x for x in vals if x is not None]
    return round(100 * sum(v) / len(v), 1) if v else None


def _fill(q):
    inv = get_engine().open_inv["invoice_id"]
    return q.replace("{inv1}", str(inv.iloc[0])) if len(inv) else q


def run_eval(llm, delay=0.0, limit=None):
    items = json.loads(QFILE.read_text(encoding="utf-8"))
    items = items[:limit] if limit else items
    rows = []
    for it in items:
        q = _fill(it["q"])
        t0 = time.time()
        try:
            out = run_agent(q, [], llm)
        except Exception as ex:
            out = {"answer": f"CRASH: {ex}", "tool_trace": [], "ok": False, "verified": False,
                   "retried": False, "fallback": False, "steps": 0, "unverified": []}
        called = [t["tool"] for t in out["tool_trace"]]
        tool_ok = (called == []) if it.get("strict_none") else all(t in called for t in it["expect_tools"])
        rows.append({
            "id": it["id"], "category": it["category"], "lang": it["lang"], "question": q,
            "called": called, "tool_ok": tool_ok, "api_ok": bool(out["ok"]),
            "verified": bool(out.get("verified")), "retried": bool(out["retried"]),
            "fallback": bool(out["fallback"]),
            "clean_first_try": bool(out["ok"] and out.get("verified") and not out["retried"] and not out["fallback"]),
            "unverified": out.get("unverified", []),
            "reply_in_roman_urdu": (detect_lang(out["answer"]) == "ur") if it["lang"] == "ur" else None,
            "seconds": round(time.time() - t0, 1), "answer": out["answer"]})
        print(f"{'OK  ' if tool_ok else 'FAIL'} {it['id']} {called}")
        if delay:
            time.sleep(delay)

    cats = defaultdict(list)
    for r in rows:
        cats[r["category"]].append(r["tool_ok"])
    strict = [r["tool_ok"] for r, it in zip(rows, items) if it.get("strict_none")]
    summary = {
        "when": datetime.now().isoformat(timespec="seconds"), "llm": llm.name,
        "model": getattr(llm, "model", None), "n_questions": len(rows),
        "tool_choice_accuracy_pct": _rate([r["tool_ok"] for r in rows]),
        "out_of_scope_refusal_pct": _rate(strict),
        "clean_first_try_pct": _rate([r["clean_first_try"] for r in rows]),
        "retried": sum(r["retried"] for r in rows),
        "template_fallbacks": sum(r["fallback"] for r in rows),
        "final_answers_with_untraceable_numbers": sum(1 for r in rows if r["api_ok"] and not r["verified"]),
        "api_failures": sum(not r["api_ok"] for r in rows),
        "roman_urdu_reply_pct_heuristic": _rate([r["reply_in_roman_urdu"] for r in rows]),
        "avg_seconds": round(sum(r["seconds"] for r in rows) / max(len(rows), 1), 1),
        "by_category_tool_accuracy_pct": {k: _rate(v) for k, v in cats.items()},
    }
    return {"summary": summary, "rows": rows}


def write_reports(result):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "agent_eval.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    s = result["summary"]
    lines = [f"# Agent evaluation ({s['when']})", f"LLM: {s['llm']} / {s['model']}", "",
             "| Metric | Value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in s.items() if k not in ("when", "llm", "model", "by_category_tool_accuracy_pct")]
    lines += ["", "| Category | Tool accuracy % |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in s["by_category_tool_accuracy_pct"].items()]
    bad = [r for r in result["rows"] if not r["tool_ok"] or not r["clean_first_try"]]
    if bad:
        lines += ["", "## Questions to inspect", "| id | called | verified | retried | fallback |", "|---|---|---|---|---|"]
        lines += [f"| {r['id']} | {r['called']} | {r['verified']} | {r['retried']} | {r['fallback']} |" for r in bad]
    (OUT_DIR / "agent_eval.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="smoke test only; mock accuracy is not meaningful")
    ap.add_argument("--delay", type=float, default=2.0, help="seconds between questions (rate limits)")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    res = run_eval(get_llm(force_mock=a.mock), a.delay if not a.mock else 0, a.limit)
    write_reports(res)
    print(json.dumps(res["summary"], indent=2))