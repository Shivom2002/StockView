from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def compute_sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def compute_ema(series: pd.Series, window: int) -> pd.Series:
    """Compute Exponential Moving Average using pandas EWM."""
    return series.ewm(span=window, adjust=False).mean()


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_atr(history: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Compute Average True Range (ATR).

    True Range is the maximum of:
    - Current High - Current Low
    - abs(Current High - Previous Close)
    - abs(Current Low - Previous Close)

    ATR is the EMA of True Range over the specified window.
    """
    high = history["High"]
    low = history["Low"]
    close = history["Close"]
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(span=window, adjust=False).mean()
    return atr


def compute_annualized_volatility(returns: pd.Series) -> float:
    return float(returns.std() * np.sqrt(252))


def compute_max_drawdown(returns: pd.Series, window: int = 252) -> float:
    recent = returns.dropna().tail(window)
    if recent.empty:
        return float("nan")
    cumulative = (1 + recent).cumprod()
    peak = cumulative.cummax()
    drawdown = cumulative / peak - 1
    return float(drawdown.min())


def compute_long_term_pullback_signals(
    history: pd.DataFrame,
    pullback_band: float = 0.005,
    rsi_min: int = 40,
    rsi_max: int = 55,
) -> pd.DataFrame:
    """
    Compute Long-Term Pullback-in-Trend strategy signals.

    Strategy Rules (Daily timeframe):
    - Trend filter: Close > EMA200
    - Pullback zone: Low <= EMA50 * (1 + pullback_band)
    - RSI confirmation: RSI14 between rsi_min and rsi_max
    - Entry trigger: Close crosses above EMA20 OR (RSI14[t] > RSI14[t-1] AND Close > EMA20)
    - Exit: Close < EMA200 (trend break)

    Parameters:
        history: DataFrame with OHLCV data
        pullback_band: Pullback tolerance above EMA50 (default 0.005 = 0.5%)
        rsi_min: Minimum RSI for entry (default 40)
        rsi_max: Maximum RSI for entry (default 55)

    Returns:
        DataFrame with original columns plus:
        - ema20, ema50, ema200: Exponential moving averages
        - rsi14: 14-period RSI
        - atr14: 14-period ATR
        - ltp_trend_ok: Boolean, True if Close > EMA200
        - ltp_pullback_ok: Boolean, True if Low <= EMA50 * (1 + pullback_band)
        - ltp_rsi_ok: Boolean, True if rsi_min <= RSI14 <= rsi_max
        - ltp_entry: Boolean, True on entry signal
        - ltp_exit: Boolean, True on exit signal
        - ltp_position: Integer, 1 if in position, 0 otherwise
    """
    df = history.copy()

    # Compute indicators
    df["ema20"] = compute_ema(df["Close"], 20)
    df["ema50"] = compute_ema(df["Close"], 50)
    df["ema200"] = compute_ema(df["Close"], 200)
    df["rsi14"] = compute_rsi(df["Close"], 14)
    df["atr14"] = compute_atr(df, 14)

    # Strategy conditions
    df["ltp_trend_ok"] = df["Close"] > df["ema200"]
    df["ltp_pullback_ok"] = df["Low"] <= df["ema50"] * (1 + pullback_band)
    df["ltp_rsi_ok"] = (df["rsi14"] >= rsi_min) & (df["rsi14"] <= rsi_max)

    # Entry trigger: Close crosses above EMA20 OR (RSI uptick AND Close > EMA20)
    # Using explicit cross detection: previous close was <= ema20, current close > ema20
    close_cross_above_ema20 = (df["Close"].shift(1) <= df["ema20"].shift(1)) & (
        df["Close"] > df["ema20"]
    )
    rsi_uptick = df["rsi14"] > df["rsi14"].shift(1)
    close_above_ema20 = df["Close"] > df["ema20"]

    entry_trigger = close_cross_above_ema20 | (rsi_uptick & close_above_ema20)

    # Combined entry signal: All conditions must be met
    df["ltp_entry"] = (
        df["ltp_trend_ok"] & df["ltp_pullback_ok"] & df["ltp_rsi_ok"] & entry_trigger
    )

    # Exit signal: Close < EMA200
    df["ltp_exit"] = df["Close"] < df["ema200"]

    # Track position state using vectorized logic
    # Start with no position
    df["ltp_position"] = 0

    # Use iterative approach to track position state (cannot be fully vectorized due to state dependency)
    position = 0
    positions = []
    for idx in range(len(df)):
        if position == 0:  # Not in position
            if df["ltp_entry"].iloc[idx]:
                position = 1  # Enter position
        else:  # In position (position == 1)
            if df["ltp_exit"].iloc[idx]:
                position = 0  # Exit position
        positions.append(position)

    df["ltp_position"] = positions

    return df


def compute_indicators(history: pd.DataFrame) -> Dict[str, object]:
    close = history["Close"].dropna()
    sma20 = compute_sma(close, 20)
    sma50 = compute_sma(close, 50)
    sma200 = compute_sma(close, 200)
    rsi = compute_rsi(close, 14)

    returns = close.pct_change().dropna()
    volatility = compute_annualized_volatility(returns) if not returns.empty else float("nan")
    max_drawdown = compute_max_drawdown(returns, 252)

    return {
        "sma20": sma20,
        "sma50": sma50,
        "sma200": sma200,
        "rsi": rsi,
        "volatility": volatility,
        "max_drawdown": max_drawdown,
    }
