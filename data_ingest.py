class RawDataIngestor:
    def __init__(self, config):
        pass  # read API keys, file paths, etc.

    def fetch_etf_holdings(self, date_range):
        """Download raw ETF holdings CSVs → store in raw lake."""
        raise NotImplementedError

    def fetch_price_history(self, tickers, date_range):
        """Download raw price history → store in raw lake."""
        raise NotImplementedError

    def run_all(self, date_range):
        """Orchestrate all fetch_* calls."""
        self.fetch_etf_holdings(date_range)
        self.fetch_price_history(...)
