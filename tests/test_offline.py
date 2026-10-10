import json
from src import offline
from src.agent import run_agent, MockLLM


class BrokenLLM:
    name = "broken"
    def chat(self, messages, tools): raise RuntimeError("network down")


def _cache_with(tmp_path, monkeypatch, q):
    out = run_agent(q, [], MockLLM())
    p = tmp_path / "cache.json"
    p.write_text(json.dumps({offline.normalize(q): {"question": q, "answer": out["answer"],
                 "tool_trace": out["tool_trace"], "steps": out["steps"]}}, default=str))
    monkeypatch.setattr(offline, "CACHE_PATH", p)
    return p


def test_hit_and_punctuation_insensitive(tmp_path, monkeypatch):
    _cache_with(tmp_path, monkeypatch, "What is my cash risk?")
    hit, why = offline.lookup("what is my cash risk")
    assert why == "hit" and hit["source"] == "cache" and hit["verified"]

def test_stale_when_engine_numbers_differ(tmp_path, monkeypatch):
    p = _cache_with(tmp_path, monkeypatch, "What is my cash risk?")
    data = json.loads(p.read_text())
    key = next(iter(data))
    data[key]["tool_trace"][0]["result"]["p_shortfall_pct"] += 1
    p.write_text(json.dumps(data))
    assert offline.lookup("What is my cash risk?") == (None, "stale")

def test_different_digits_never_match(tmp_path, monkeypatch):
    _cache_with(tmp_path, monkeypatch, "Can I hire someone at 60,000?")
    assert offline.lookup("Can I hire someone at 70,000?")[1] == "miss"

def test_offline_miss_uses_mock(tmp_path, monkeypatch):
    monkeypatch.setattr(offline, "CACHE_PATH", tmp_path / "none.json")
    out = offline.ask("What is my cash risk?", offline=True)
    assert out["source"] == "mock" and out["tool_trace"]

def test_api_failure_falls_back_to_cache(tmp_path, monkeypatch):
    _cache_with(tmp_path, monkeypatch, "What is my cash risk?")
    out = offline.ask("What is my cash risk?", llm=BrokenLLM())
    assert out["source"] == "cache"