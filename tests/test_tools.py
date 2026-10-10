import json
from src.tools import call_tool, TOOLS, TOOL_SCHEMAS

def test_all_tools_have_schema():
    assert set(TOOLS) == {s["name"] for s in TOOL_SCHEMAS}

def test_runway():
    r = call_tool("get_runway")
    assert 0 <= r["p_shortfall"] <= 1
    assert r["status"] in ("green", "amber", "red")
    json.dumps(r)

def test_rank_chase():
    r = call_tool("rank_chase", {"top_n": 2})
    assert len(r["ranked"]) <= 2
    json.dumps(r)

def test_what_if_valid_and_invalid():
    ok = call_tool("run_what_if", {"change": {"type": "hire", "monthly": 60000, "start": "2027-03-01"}})
    assert "p_after_pct" in ok
    bad = call_tool("run_what_if", {"change": {"type": "nonsense"}})
    assert "error" in bad

def test_invoice_and_reminder():
    from src.tools import get_engine
    inv_id = get_engine().open_inv["invoice_id"].iloc[0]
    inv = call_tool("get_invoice", {"invoice_id": inv_id})
    assert "predicted_pay_date" in inv
    assert "error" in call_tool("get_invoice", {"invoice_id": "NOPE"})
    msg = call_tool("draft_reminder", {"invoice_id": inv_id, "lang": "ur", "tone": "firm"})
    assert "message" in msg

def test_unknown_tool_does_not_crash():
    assert "error" in call_tool("does_not_exist")