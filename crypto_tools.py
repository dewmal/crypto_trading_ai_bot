from pathlib import Path

from pydantic_ai.tools import RunContext
from talipp.indicators import EMA

from data import TradingDeps
from initicators import get_binance_spot_ema
from news import get_latest_news


def get_ta_ema(ctx: RunContext[str],
    symbol: str = "BTCUSDT",
    interval: str = "1m",
    period: int = 14,
    limit: int = 100,) -> str:
  """Get Market EMA"""
  ema = get_binance_spot_ema(
      symbol="BTCUSDT",
      interval="1m",
      period=3,
      limit=20,
  )
  return f"{ema}"

def get_news(
    ctx: RunContext[TradingDeps],
    keywords: list[str] | None = None,
    time_filter: str = "24h",
    limit: int = 10,
    match_all: bool = False,
) -> list[str]:
    """Retrieve recent news from the application-owned crypto-news database.

    Args:
        keywords: Search terms such as ["bitcoin", "ETF"].
        time_filter: Recency window supported by get_latest_news.
        limit: Maximum number of news records.
        match_all: Whether every keyword must match.
    """
    cleaned_keywords = list(
        dict.fromkeys(
            keyword.strip()
            for keyword in (keywords or [])
            if keyword and keyword.strip()
        )
    )

    bounded_limit = max(1, min(limit, 25))

    # The model cannot select an arbitrary filesystem path.
    db_path = Path(ctx.deps.news_db_path)

    if not db_path.is_file():
        raise RuntimeError(
            "The configured crypto-news database does not exist."
        )

    try:
        articles = get_latest_news(
            db_path=str(db_path),
            keywords=cleaned_keywords or None,
            time_filter=time_filter,
            limit=bounded_limit,
            match_all=match_all,
        )
    except Exception as exc:
        raise RuntimeError("The crypto-news query failed.") from exc

    # Limit each item to avoid an unexpectedly large tool response.
    results: list[str] = []

    for article in articles:
        text = str(article).strip()

        if text:
            results.append(text[:2_000])

    return results
