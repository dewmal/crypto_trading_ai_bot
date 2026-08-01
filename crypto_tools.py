from pydantic_ai.tools import RunContext
from talipp.indicators import EMA

from initicators import get_binance_spot_ema


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
