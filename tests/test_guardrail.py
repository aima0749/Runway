from src.guardrail import verify_answer, fallback_answer, detect_lang

TRACE = [{"tool": "run_what_if",
          "args": {"change": {"type": "hire", "monthly": 60000.0, "start": "2027-03-01"}},
          "result": {"p_before": 0.68, "p_after": 0.81, "p_before_pct": 68, "p_after_pct": 81,
                     "min_cash_p50_before": -50000.0, "min_cash_p50_after": -110000.0,
                     "status_after": "red"}}]

RUNWAY = [{"tool": "get_runway", "args": {},
           "result": {"p_shortfall": 0.57, "p_shortfall_pct": 57, "status": "red",
                      "likely_cash_out_date": "2026-12-09", "horizon_days": 60}}]


def test_good_answer_passes():
    g = verify_answer("Hiring at Rs 60,000 per month from March raises risk from 68% to 81%, "
                      "with a shortfall of Rs 110,000.", TRACE)
    assert g.ok and g.checked >= 4

def test_invented_percentage_fails():
    g = verify_answer("Risk would rise to 92%.", TRACE)
    assert not g.ok and "92%" in g.unverified

def test_invented_amount_fails():
    assert not verify_answer("You would be short by Rs 75,000.", TRACE).ok

def test_numbers_from_user_message_are_allowed():
    assert verify_answer("Rs 60,000 is a big commitment.", [], ["can I hire at 60k?"]).ok

def test_fraction_quoted_as_percent():
    assert verify_answer("There is a 57% chance.", RUNWAY).ok

def test_dates():
    assert verify_answer("Around 9 December 2026.", RUNWAY).ok
    assert verify_answer("Around December 9th.", RUNWAY).ok
    assert verify_answer("Around the 9th.", RUNWAY).ok
    assert not verify_answer("Around 12 December.", RUNWAY).ok
    assert not verify_answer("Around the 20th.", RUNWAY).ok

def test_ids_and_small_counts_ignored():
    assert verify_answer("Chase INV3 first. Top 2 invoices matter most.", RUNWAY).ok

def test_fallback_is_itself_verified():
    for lang in ("en", "ur"):
        fb = fallback_answer(RUNWAY, lang)
        assert verify_answer(fb, RUNWAY).ok
        assert "57%" in fb

def test_fallback_without_tools_has_no_numbers():
    assert verify_answer(fallback_answer([], "en"), []).checked == 0

def test_detect_lang():
    assert detect_lang("Mera paisa kab khatam hoga?") == "ur"
    assert detect_lang("Which invoice should I chase first?") == "en"