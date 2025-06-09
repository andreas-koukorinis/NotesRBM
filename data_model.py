import pandas as pd

class ParquetStore:
    def __init__(self, base_path: str):
        self.base = base_path

    def read(self, dataset: str, date: str) -> pd.DataFrame:
        """Load cleaned Parquet for a given dataset + date."""
        raise NotImplementedError

    def write(self, dataset: str, df: pd.DataFrame, date: str):
        """Write cleaned DataFrame → Parquet."""
        raise NotImplementedError
