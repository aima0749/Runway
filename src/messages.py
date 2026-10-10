from datetime import datetime

from config import LANGS, TONES


_TEMPLATES = {
    ("en", "friendly"):
        "Hi {name}, hope you're doing well. Invoice {invoice_id} for {amount} was due on {due}. "
        "Could you please send the payment by the end of this week? Thank you!",
    ("en", "firm"):
        "Hello {name}, this is a reminder that invoice {invoice_id} for {amount} was due on {due}. "
        "Please arrange payment at your earliest convenience. Thank you.",
    ("en", "final"):
        "Hello {name}, invoice {invoice_id} for {amount}, due on {due}, is overdue. "
        "Please arrange payment within 7 days. Thank you for your prompt attention.",
    ("ur", "friendly"):
        "Assalam o Alaikum {name}, umeed hai aap khairiyat se hain. Invoice {invoice_id} ki raqam "
        "{amount} {due} ko due thi. Meharbani karke is hafte tak payment bhej dein. Shukriya!",
    ("ur", "firm"):
        "Assalam o Alaikum {name}, invoice {invoice_id} ki {amount} raqam {due} ko due thi. "
        "Meharbani karke jald payment ka intezam kar dein. Shukriya.",
    ("ur", "final"):
        "Assalam o Alaikum {name}, invoice {invoice_id} ki {amount} raqam, jis ki due date {due} thi, "
        "ab overdue hai. Meharbani karke 7 din ke andar payment bhej dein. Shukriya.",
}


def draft_reminder(client: str, invoice: dict, tone: str = "friendly", lang: str = "en") -> str:
    """Create a concise invoice reminder in English or Roman Urdu."""
    if lang not in LANGS:
        raise ValueError(f"Unknown language {lang!r}. Valid languages: {', '.join(LANGS)}.")
    if tone not in TONES:
        raise ValueError(f"Unknown tone {tone!r}. Valid tones: {', '.join(TONES)}.")

    for key in ("invoice_id", "amount_pkr", "due_date"):
        if key not in invoice:
            raise ValueError(f"Missing required invoice key: {key}.")

    try:
        due_date = datetime.strptime(str(invoice["due_date"]), "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError("'due_date' must be a valid date in YYYY-MM-DD format.") from exc

    values = {
        "name": str(client),
        "invoice_id": str(invoice["invoice_id"]),
        "amount": f"Rs {float(invoice['amount_pkr']):,.0f}",
        "due": f"{due_date.day} {due_date.strftime('%b %Y')}",
    }
    return _TEMPLATES[(lang, tone)].format_map(values)