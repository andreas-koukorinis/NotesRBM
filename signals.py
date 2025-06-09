from abc import ABC, abstractmethod
import pandas as pd

class BaseSignal(ABC):
    def __init__(self, ts_provider: TimeSeriesProvider, params: dict):
        self.ts = ts_provider
        self.params = params

    @abstractmethod
    def compute(self, start_date, end_date, universe) -> pd.DataFrame:
        """
        Returns DataFrame: index=date, columns=assets, values=position weights.
        """
        pass

class ETFFlowSignal(BaseSignal):
    def compute(self, start_date, end_date, universe):
        # 1) fetch data
        # 2) compute 6m ΔAUM / mcap
        # 3) cross-section decile → +1/-1
        # 4) normalize to dollar-neutral
        raise NotImplementedError

# stub for any future signals:
class PriceMomentumSignal(BaseSignal):
    def compute(self, start_date, end_date, universe):
        raise NotImplementedError
