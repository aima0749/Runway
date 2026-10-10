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
        "Hello {name}, invoice {invoice_id} for {amount}, due on {due}, {overdue}. "
        "Please arrange payment {deadline}. Thank you for your prompt attention.",
    ("ur", "friendly"):
        "Assalam o Alaikum {name}, umeed hai aap khairiyat se hain. Invoice {invoice_id} ki raqam "
        "{amount} {due} ko due thi. Meharbani karke is hafte tak payment bhej dein. Shukriya!",
    ("ur", "firm"):
        "Assalam o Alaikum {name}, invoice {invoice_id} ki {amount} raqam {due} ko due thi. "
        "Meharbani karke jald payment ka intezam kar dein. Shukriya.",
    ("ur", "final"):
        "Assalam o Alaikum {name}, invoice {invoice_id} ki {amount} raqam, jis ki due date {due} thi, "
        "ab {overdue}. Meharbani karke {deadline} payment bhej dein. Shukriya.",
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
    if tone == "final":
        overdue = "is overdue" if lang == "en" else "overdue hai"
        days_overdue = invoice.get("days_overdue")
        if isinstance(days_overdue, int) and not isinstance(days_overdue, bool) and days_overdue >= 0:
            if lang == "en":
                unit = "day" if days_overdue == 1 else "days"
                overdue = f"{days_overdue} {unit} overdue"
            else:
                overdue = f"{days_overdue} din overdue"

        deadline = "within 7 days" if lang == "en" else "7 din ke andar"
        pay_by = invoice.get("pay_by")
        if isinstance(pay_by, str):
            try:
                pay_by_date = datetime.strptime(pay_by, "%Y-%m-%d")
            except ValueError:
                pass
            else:
                formatted_pay_by = pay_by_date.strftime("%d %b %Y")
                deadline = f"by {formatted_pay_by}" if lang == "en" else f"{formatted_pay_by} tak"
        values.update(overdue=overdue, deadline=deadline)

    return _TEMPLATES[(lang, tone)].format_map(values)