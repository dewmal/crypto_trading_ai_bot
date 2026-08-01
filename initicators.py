import time

import requests
from talipp.indicators import EMA, RSI


def get_binance_spot_ema(
    symbol: str = "BTCUSDT",
    interval: str = "1m",
    period: int = 3,
    limit: int = 100,
) -> EMA:
    """
    Fetch closed Binance Spot candles and calculate their EMA.

    Args:
        symbol: Binance Spot symbol, such as "BTCUSDT".
        interval: Candle interval, such as "1m", "5m", "1h", or "1d".
        period: EMA period.
        limit: Maximum number of closed candles to use.

    Returns:
        A talipp EMA indicator object.
    """
    if period < 1:
        raise ValueError("period must be at least 1")

    if not period <= limit <= 1000:
        raise ValueError("limit must be between period and 1000")

    url = "https://api.binance.com/api/v3/klines"

    # Request an additional candle because the newest candle may still be open.
    request_limit = min(limit + 1, 1000)

    response = requests.get(
        url,
        params={
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": request_limit,
        },
        timeout=10,
    )

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(
            f"Binance API error {response.status_code}: {response.text}"
        ) from exc

    klines = response.json()
    current_time_ms = int(time.time() * 1000)

    # kline[4] = close price
    # kline[6] = candle close time
    closed_candles = [
        kline for kline in klines
        if int(kline[6]) < current_time_ms
    ][-limit:]

    close_prices = [float(kline[4]) for kline in closed_candles]

    if len(close_prices) < period:
        raise ValueError(
            f"Only {len(close_prices)} closed candles were returned; "
            f"at least {period} are required."
        )

    return EMA(period=period, input_values=close_prices)

def get_binance_spot_rsi(
    symbol: str = "BTCUSDT",
    interval: str = "1m",
    period: int = 14,
    limit: int = 100,
) -> RSI:
    """
    Fetch closed Binance Spot candles and calculate RSI.

    Returns:
        talipp RSI indicator object.
    """
    if period < 1:
        raise ValueError("period must be at least 1")

    if limit < period + 1:
        raise ValueError("limit must be greater than the RSI period")

    if limit > 1000:
        raise ValueError("limit cannot exceed 1000")

    response = requests.get(
        "https://data-api.binance.vision/api/v3/klines",
        params={
            "symbol": symbol.upper(),
            "interval": interval,
            # Extra candle because the latest candle may still be open.
            "limit": min(limit + 1, 1000),
        },
        timeout=10,
    )

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(
            f"Binance API error {response.status_code}: {response.text}"
        ) from exc

    klines = response.json()

    if not isinstance(klines, list):
        raise RuntimeError(f"Unexpected Binance response: {klines}")

    current_time_ms = int(time.time() * 1000)

    # kline[4] = closing price
    # kline[6] = candle closing time
    closed_klines = [
        kline
        for kline in klines
        if int(kline[6]) < current_time_ms
    ][-limit:]

    close_prices = [float(kline[4]) for kline in closed_klines]

    if len(close_prices) < period + 1:
        raise ValueError(
            f"RSI({period}) requires at least {period + 1} prices, "
            f"but Binance returned {len(close_prices)} closed candles."
        )

    return RSI(
        period=period,
        input_values=close_prices,
    )
