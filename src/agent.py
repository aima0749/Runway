"""AI CFO agent. The LLM chooses tools and explains results. It never calculates."""
import json
import os
import re
from dotenv import load_dotenv

from config import MAX_AGENT_STEPS
from src.tools import TOOL_SCHEMAS, call_tool, get_engine
from src.guardrail import verify_answer, fallback_answer, detect_lang

load_dotenv()

DEFAULT_MODEL = "llama-3.3-70b-versatile"


def _system_prompt() -> str:
    as_of = get_engine().as_of.strftime("%Y-%m-%d")
    return f"""You are Runway Radar, a cash-flow assistant for small business owners in Pakistan.
Today's date in the business data is {as_of}.

RULES
1. Every number, percentage, date, and amount in your answer MUST come from a tool result. Never calculate, estimate, or guess a number yourself.
2. For any question about cash, risk, invoices, hiring, or spending, call a tool first.
3. If a tool returns an error, tell the user plainly what is wrong. Do not invent an answer.
4. If the question is not about this business's cash flow, politely say you can only help with cash flow.
5. Reply in the user's language. If they write Roman Urdu (Urdu in English letters), reply in Roman Urdu. Otherwise reply in English.
6. Keep answers short: 2 to 4 sentences, plain words, no jargon. Then suggest one next step, such as chasing an invoice or drafting a reminder.
7. Use percentages as given (for example 68%) and amounts as 'Rs 60,000'.
8. Never save an invoice. add_invoice_from_text only proposes; the user must confirm.
9. Do not give yes/no verdicts such as "safe" or "not affordable". State the risk change and the status, and let the owner decide.
10. If a tool result contains a "warning", tell the user about it in plain words. If an invoice proposal has problems or warnings, mention them and ask the user to confirm or correct before anything is saved.
"""


def _openai_tools() -> list:
    return [{"type": "function", "function": s} for s in TOOL_SCHEMAS]


# ---------------- LLM backends ----------------
# Both return: {"content": str | None, "tool_calls": [{"id", "name", "arguments": dict}]}

class GroqLLM:
    name = "groq"

    def __init__(self, api_key=None, model=None):
        from groq import Groq
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        if not self.api_key:
            raise RuntimeError("LLM_API_KEY is missing. Add it to your .env file.")
        self.model = model or os.getenv("LLM_MODEL") or DEFAULT_MODEL
        self.client = Groq(api_key=self.api_key)

        def chat(self, messages, tools):
        kwargs = dict(model=self.model, messages=messages, temperature=0, max_tokens=2000)
        if tools:
            kwargs.update(tools=tools, tool_choice="auto")
        resp = self.client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        calls = []
        for tc in (msg.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append({"id": tc.id, "name": tc.function.name, "arguments": args})
        return {"content": msg.content, "tool_calls": calls}

class MockLLM:
    """Keyword-routed fake LLM. Lets you test the loop with no key and no network."""
    name = "mock"

    def chat(self, messages, tools):
        last = messages[-1]
        if last["role"] == "tool":
            return {"content": self._summarize(messages), "tool_calls": []}

        text = last["content"].lower()
        if any(k in text for k in ("dena hai", "dena he", "owes", "owe me")):
           return self._call("add_invoice_from_text", {"text": last["content"]})
        if any(k in text for k in ("hire", "salary", "afford", "rakh")):
            m = re.search(r"(\d[\d,]*)\s*(k)?", text)
            amount = 60000.0
            if m:
                amount = float(m.group(1).replace(",", "")) * (1000 if m.group(2) else 1)
            start = (get_engine().as_of + __import__("pandas").Timedelta(30, unit="D")).strftime("%Y-%m-%d")
            return self._call("run_what_if",
                              {"change": {"type": "hire", "monthly": amount, "start": start}})
        if any(k in text for k in ("chase", "collect", "pehle", "first", "which client")):
            return self._call("rank_chase", {"top_n": 3})
        if any(k in text for k in ("remind", "message", "whatsapp")):
            lang = "ur" if "urdu" in text else "en"
            return self._call("draft_reminder", {"invoice_id": "INV3", "lang": lang, "tone": "friendly"})
        if any(k in text for k in ("cash", "runway", "short", "risk", "paisa", "paise")):
            return self._call("get_runway", {})
        return {"content": "I can only help with this business's cash flow.", "tool_calls": []}

    @staticmethod
    def _call(name, args):
        return {"content": None, "tool_calls": [{"id": f"mock_{name}", "name": name, "arguments": args}]}

    @staticmethod
    def _summarize(messages):
        # Use only tool results from the current turn (after the last user message).
        idx = max(i for i, m in enumerate(messages) if m["role"] == "user")
        results = [json.loads(m["content"]) for m in messages[idx:] if m["role"] == "tool"]
        r = results[-1]
        if "parsed" in r:
            p = r["parsed"]
            if r["problems"]:
                return "I could not read that invoice fully: " + "; ".join(r["problems"])
            return (f"I found an invoice for {p['client_name']}: Rs {p['amount_pkr']:,.0f}, "
                    f"due {p['due_date']}. Please confirm before I save it.")
        if "p_after_pct" in r:
            return (f"Your risk of running short would go from {r['p_before_pct']}% "
                    f"to {r['p_after_pct']}%. Check whether you can absorb that before committing.")
        if "ranked" in r and r["ranked"]:
            t = r["ranked"][0]
            return (f"Chase {t['invoice_id']} ({t['client']}) first. Your shortfall risk drops "
                    f"from {t['p_before_pct']}% to {t['p_after_pct']}%. Want me to draft a reminder?")
        if "message" in r:
            return r["message"]
        if "p_shortfall_pct" in r:
            return (f"There is a {r['p_shortfall_pct']}% chance you run short of cash "
                    f"(status: {r['status']}). Want to see which invoice to chase first?")
        return json.dumps(r)


def get_llm(force_mock: bool = False):
    if force_mock:
        return MockLLM()
    try:
        return GroqLLM()
    except Exception as ex:
        print(f"[agent] Falling back to MockLLM: {ex}")
        return MockLLM()


# ---------------- The agent loop ----------------
def _result(answer, trace, steps, ok, llm, g=None, retried=False, fallback=False):
    return {"answer": answer, "tool_trace": trace, "steps": steps, "ok": ok, "llm": llm.name,
            "verified": bool(g and g.ok), "numbers_checked": g.checked if g else 0,
            "unverified": g.unverified if g else [], "retried": retried, "fallback": fallback}


def run_agent(user_message: str, history: list | None = None, llm=None) -> dict:
    """history: list of {'role': 'user'|'assistant', 'content': str} from earlier turns."""
    llm = llm or get_llm()
    messages = [{"role": "system", "content": _system_prompt()}]
    messages += history or []
    messages.append({"role": "user", "content": user_message})
    trace, retried = [], False
    extra_texts = [user_message, get_engine().as_of.strftime("%Y-%m-%d")] + \
                  [h["content"] for h in (history or [])]

    for step in range(MAX_AGENT_STEPS):
        reply, last_err = None, None
        for attempt in range(2):                       # one retry on API hiccups
            try:
                reply = llm.chat(messages, _openai_tools())
                break
            except Exception as ex:
                last_err = ex
        if reply is None:
            return _result(f"The AI service had a problem: {last_err}", trace, step, False, llm)

        calls = reply["tool_calls"]
        if not calls:
            answer = reply["content"] or ""
            g = verify_answer(answer, trace, extra_texts)
            if g.ok:
                return _result(answer, trace, step + 1, True, llm, g, retried)
            if not retried:                            # ask the LLM to fix itself once
                retried = True
                messages.append({"role": "assistant", "content": answer})
                messages.append({"role": "user", "content":
                    "Your answer contains values that do not appear in the tool results: "
                    f"{', '.join(g.unverified)}. Rewrite it using only numbers from the tool "
                    "results. If you need data, call a tool."})
                continue
            fb = fallback_answer(trace, detect_lang(user_message))
            g2 = verify_answer(fb, trace, extra_texts)
            return _result(fb, trace, step + 1, True, llm, g2, retried, fallback=True)

        messages.append({
            "role": "assistant", "content": reply["content"] or "",
            "tool_calls": [{"id": c["id"], "type": "function",
                            "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
                           for c in calls],
        })
        for c in calls:
            result = call_tool(c["name"], c["arguments"])
            trace.append({"tool": c["name"], "args": c["arguments"], "result": result})
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(result)})

    return _result("I could not finish that. Please try a simpler question.",
                   trace, MAX_AGENT_STEPS, False, llm)


# ---------------- Terminal chat ----------------
if __name__ == "__main__":
    import sys
    llm = get_llm(force_mock="--mock" in sys.argv)
    print(f"Runway Radar agent (LLM: {llm.name}). Type 'quit' to exit.")
    history = []
    while True:
        q = input("\nYou: ").strip()
        if q.lower() in ("quit", "exit", ""):
            break
        out = run_agent(q, history, llm)
        print("Tools:", [t["tool"] for t in out["tool_trace"]])
        print("CFO:", out["answer"])
        print(f"[verified={out['verified']} checked={out['numbers_checked']} "
              f"retried={out['retried']} fallback={out['fallback']}]")
        history += [{"role": "user", "content": q}, {"role": "assistant", "content": out["answer"]}]