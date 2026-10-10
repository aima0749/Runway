import subprocess
import sys

import pytest

from config import LANGS, TONES
from src.messages import draft_reminder


INVOICE = {
    "invoice_id": "INV-2026-42",
    "client_id": "C42",
    "amount_pkr": 300000.0,
    "due_date": "2026-10-20",
    "predicted_pay_date": "2026-10-25",
    "range_start": "2026-10-22",
    "range_end": "2026-11-02",
    "p_more_than_14_days_late": 0.25,
}


@pytest.mark.parametrize("lang", LANGS)
@pytest.mark.parametrize("tone", TONES)
def test_every_language_and_tone_has_required_invoice_details(lang, tone):
    message = draft_reminder("Ayesha Khan", INVOICE, tone=tone, lang=lang)

    assert isinstance(message, str)
    assert "Ayesha Khan" in message
    assert "INV-2026-42" in message
    assert "Rs 300,000" in message
    assert "20 Oct 2026" in message


def test_module_import_does_not_import_tools_or_streamlit():
    code = (
        "import sys; import src.messages; "
        "assert 'src.tools' not in sys.modules; "
        "assert 'streamlit' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_unknown_language_raises_value_error():
    with pytest.raises(ValueError, match="Valid languages"):
        draft_reminder("Ayesha", INVOICE, lang="fr")


def test_unknown_tone_raises_value_error():
    with pytest.raises(ValueError, match="Valid tones"):
        draft_reminder("Ayesha", INVOICE, tone="urgent")


def test_missing_invoice_key_raises_value_error():
    invoice = dict(INVOICE)
    del invoice["amount_pkr"]

    with pytest.raises(ValueError, match="amount_pkr"):
        draft_reminder("Ayesha", invoice)


@pytest.mark.parametrize("lang", LANGS)
@pytest.mark.parametrize("tone", TONES)
def test_every_message_is_under_600_characters(lang, tone):
    assert len(draft_reminder("Ayesha Khan", INVOICE, tone=tone, lang=lang)) < 600


@pytest.mark.parametrize(
    ("lang", "expected_overdue", "expected_deadline"),
    [
        ("en", "is 3 days overdue", "by 27 Oct 2026"),
        ("ur", "3 din overdue hai", "27 Oct 2026 tak"),
    ],
)
def test_final_tone_uses_both_valid_optional_values(lang, expected_overdue, expected_deadline):
    invoice = {**INVOICE, "days_overdue": 3, "pay_by": "2026-10-27"}

    message = draft_reminder("Ayesha Khan", invoice, tone="final", lang=lang)

    assert expected_overdue in message
    assert expected_deadline in message
    assert len(message) < 600


@pytest.mark.parametrize(
    ("lang", "expected_overdue", "fallback_deadline"),
    [
        ("en", "is 3 days overdue", "within 7 days"),
        ("ur", "3 din overdue hai", "7 din ke andar"),
    ],
)
def test_final_tone_with_only_days_overdue(lang, expected_overdue, fallback_deadline):
    invoice = {**INVOICE, "days_overdue": 3}

    message = draft_reminder("Ayesha Khan", invoice, tone="final", lang=lang)

    assert expected_overdue in message
    assert fallback_deadline in message
    assert len(message) < 600


@pytest.mark.parametrize(
    ("lang", "fallback_overdue", "expected_deadline"),
    [
        ("en", "is overdue", "by 27 Oct 2026"),
        ("ur", "overdue hai", "27 Oct 2026 tak"),
    ],
)
def test_final_tone_with_only_pay_by(lang, fallback_overdue, expected_deadline):
    invoice = {**INVOICE, "pay_by": "2026-10-27"}

    message = draft_reminder("Ayesha Khan", invoice, tone="final", lang=lang)

    assert fallback_overdue in message
    assert expected_deadline in message
    assert len(message) < 600


@pytest.mark.parametrize(
    ("lang", "fallback_overdue", "fallback_deadline"),
    [
        ("en", "is overdue", "within 7 days"),
        ("ur", "overdue hai", "7 din ke andar"),
    ],
)
def test_final_tone_without_optional_values_uses_fallbacks(lang, fallback_overdue, fallback_deadline):
    message = draft_reminder("Ayesha Khan", INVOICE, tone="final", lang=lang)

    assert fallback_overdue in message
    assert fallback_deadline in message
    assert len(message) < 600


@pytest.mark.parametrize(
    ("lang", "fallback_overdue", "fallback_deadline"),
    [
        ("en", "is overdue", "within 7 days"),
        ("ur", "overdue hai", "7 din ke andar"),
    ],
)
def test_invalid_optional_values_use_fallbacks(lang, fallback_overdue, fallback_deadline):
    invoice = {
        **INVOICE,
        "days_overdue": -2,
        "pay_by": "not-a-date",
        "days_late": 99,
    }

    message = draft_reminder("Ayesha Khan", invoice, tone="final", lang=lang)

    assert fallback_overdue in message
    assert fallback_deadline in message
    assert "99" not in message


@pytest.mark.parametrize("lang", LANGS)
def test_final_tone_uses_singular_day(lang):
    message = draft_reminder(
        "Ayesha Khan", {**INVOICE, "days_overdue": 1}, tone="final", lang=lang
    )

    assert ("is 1 day overdue" if lang == "en" else "1 din overdue hai") in message
    assert len(message) < 600