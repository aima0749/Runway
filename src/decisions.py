from config import GREEN_MAX, AMBER_MAX
from src.schemas import ChaseRow

def status(p: float) -> str:
    if p < GREEN_MAX:
        return "green"
    if p <= AMBER_MAX:
        return "amber"
    return "red"

def rank_chase(engine=None, top_n=10) -> list:
    """STUB (Dev B): dummy ranked list."""
    return [
        ChaseRow("INV3", "Client One", 300000.0, 0.68, 0.31, 0.37),
        ChaseRow("INV2", "Client Two", 150000.0, 0.68, 0.55, 0.13),
    ][:top_n]

def run_what_if(engine=None, change=None) -> dict:
    """STUB (Dev B): dummy before/after."""
    return {"p_before": 0.68, "p_after": 0.81, "min_cash_p50_before": -50000.0,
            "min_cash_p50_after": -110000.0}