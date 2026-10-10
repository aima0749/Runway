from src.schemas import Dataset

def load_and_validate(files, settings):
    """STUB (Dev A): returns (Dataset, errors). errors is a list of plain-language strings."""
    from src.generator import generate_business
    return generate_business(), []