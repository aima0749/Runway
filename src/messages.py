def draft_reminder(client: str, invoice: dict, tone: str = "friendly", lang: str = "en") -> str:
    """STUB (Dev B): placeholder text."""
    return f"[{lang}/{tone}] Reminder to {client} for invoice {invoice.get('invoice_id', '?')}."