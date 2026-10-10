from src.agent import MockLLM
from src.eval_agent import run_eval

def test_eval_runs_end_to_end_with_mock():
    res = run_eval(MockLLM(), limit=6)
    s = res["summary"]
    assert s["n_questions"] == 6 and len(res["rows"]) == 6
    assert s["final_answers_with_untraceable_numbers"] == 0