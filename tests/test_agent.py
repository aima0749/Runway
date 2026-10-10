import json


from src.agent import run_agent, MockLLM


class LoopLLM:
    """Always asks for a tool. Used to prove the step limit works."""
    name = "loop"
    def chat(self, messages, tools):
        return {"content": None, "tool_calls": [
            {"id": "x", "name": "get_runway", "arguments": {}}]}


class BrokenLLM:
    name = "broken"
    def chat(self, messages, tools):
        raise RuntimeError("network down")


def test_runway_question_uses_tool():
    out = run_agent("What is my cash risk?", llm=MockLLM())
    assert out["ok"]
    assert out["tool_trace"][0]["tool"] == "get_runway"
    pct = out["tool_trace"][0]["result"]["p_shortfall_pct"]
    assert f"{pct}%" in out["answer"]

def test_chase_question():
    out = run_agent("Which client should I chase first?", llm=MockLLM())
    assert out["tool_trace"][0]["tool"] == "rank_chase"

def test_hire_question():
    out = run_agent("Can I afford to hire someone at 60,000?", llm=MockLLM())
    t = out["tool_trace"][0]
    assert t["tool"] == "run_what_if"
    assert t["args"]["change"]["monthly"] == 60000.0

def test_out_of_scope_uses_no_tool():
    out = run_agent("Tell me a joke about cricket", llm=MockLLM())
    assert out["tool_trace"] == []

def test_step_limit():
    out = run_agent("anything", llm=LoopLLM())
    assert not out["ok"] and out["steps"] == 4

def test_llm_failure_does_not_crash():
    out = run_agent("hello", llm=BrokenLLM())
    assert not out["ok"] and "problem" in out["answer"]

def _tool_call():
    return {"content": None, "tool_calls": [{"id": "a", "name": "get_runway", "arguments": {}}]}

class LiarLLM:
    name = "liar"
    def chat(self, messages, tools):
        if not any(m["role"] == "tool" for m in messages):
            return _tool_call()
        return {"content": "You have a 99% chance of running short.", "tool_calls": []}

class FixesItself:
    name = "fixes"
    def __init__(self): self.n = 0
    def chat(self, messages, tools):
        if not any(m["role"] == "tool" for m in messages):
            return _tool_call()
        self.n += 1
        if self.n == 1:
            return {"content": "Risk is 99%.", "tool_calls": []}
        res = json.loads([m for m in messages if m["role"] == "tool"][-1]["content"])
        return {"content": f"Risk is {res['p_shortfall_pct']}%.", "tool_calls": []}

def test_liar_gets_replaced_by_fallback():
    out = run_agent("What is my cash risk?", llm=LiarLLM())
    assert out["fallback"] and out["retried"] and out["verified"]
    assert "99" not in out["answer"]

def test_llm_can_fix_itself_on_retry():
    out = run_agent("What is my cash risk?", llm=FixesItself())
    assert out["retried"] and not out["fallback"] and out["verified"]    