from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field


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
    # ------------------------------------------------------------------
    # Shared dependencies
    # ------------------------------------------------------------------




    # ------------------------------------------------------------------
    # Structured outputs
    # ------------------------------------------------------------------

class TechnicalReport(BaseModel):
    symbol: str
    timeframe: str
    trend: Literal["bullish", "bearish", "sideways"]

    fast_ema: float | None = None
    slow_ema: float | None = None

    observations: list[str] = Field(default_factory=list)
    invalidation_conditions: list[str] = Field(default_factory=list)


class NewsReport(BaseModel):
    symbol: str
    time_filter: str
    keywords: list[str] = Field(default_factory=list)

    sentiment: Literal[
        "bullish",
        "bearish",
        "neutral",
        "mixed",
        "unknown",
    ]

    expected_impact: Literal[
        "high",
        "medium",
        "low",
        "unknown",
    ]

    confidence: float = Field(ge=0, le=1)
    article_count: int = Field(ge=0)

    key_items: list[str] = Field(default_factory=list)
    bullish_catalysts: list[str] = Field(default_factory=list)
    bearish_catalysts: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    data_quality_warnings: list[str] = Field(default_factory=list)


class TradeDraft(BaseModel):
    symbol: str
    timeframe: str
    action: Literal["buy", "sell", "hold"]

    entry: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None

    risk_per_trade_pct: float = Field(ge=0, le=100)

    rationale: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class CoordinatedPlan(BaseModel):
    technical_analysis: TechnicalReport | None = None
    news_analysis: NewsReport | None = None
    draft: TradeDraft


class RiskReview(BaseModel):
    approved: bool
    concerns: list[str] = Field(default_factory=list)
    recommended_changes: list[str] = Field(default_factory=list)


class HardRiskGate(BaseModel):
    approved: bool
    reasons: list[str] = Field(default_factory=list)


class TradingResponse(BaseModel):
    technical_analysis: TechnicalReport | None = None
    news_analysis: NewsReport | None = None

    draft: TradeDraft
    risk_review: RiskReview
    hard_risk_gate: HardRiskGate

    executable: bool
