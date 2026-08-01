from dataclasses import dataclass


@dataclass
class TradingDeps:
    user_id: str
    paper_trading: bool = True

    max_risk_per_trade_pct: float = 1.0
    minimum_reward_risk_ratio: float = 2.0

    # News policy
    news_db_path: str = "crypto_news.db"
    require_news_for_trade: bool = True
    minimum_news_confidence: float = 0.25

    # Production dependencies can be added here:
    # market_data_client: MarketDataClient
    # portfolio_repository: PortfolioRepository
    # exchange_client: ExchangeClient
