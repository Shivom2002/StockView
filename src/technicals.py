from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def compute_sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


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
