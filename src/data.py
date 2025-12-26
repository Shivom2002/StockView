from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st
import yfinance as yf


try:
    _cache_data = st.cache_data
except AttributeError:  # Streamlit < 1.18
    _cache_data = st.cache


@_cache_data(show_spinner=False)
def fetch_price_history(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    stock = yf.Ticker(ticker)
    history = stock.history(period=period, interval=interval, auto_adjust=False)
    if history.empty:
        return history
    history = history.copy()
    if history.index.tz is not None:
        history.index = history.index.tz_localize(None)
    return history


@_cache_data(show_spinner=False)
def fetch_fundamentals(ticker: str) -> Dict[str, Any]:
    stock = yf.Ticker(ticker)
    info = stock.info or {}

    cashflow = None
    try:
        cashflow = stock.get_cashflow(freq="annual")
    except Exception:
        try:
            cashflow = stock.cashflow
        except Exception:
            cashflow = None

    return {
        "info": info,
        "cashflow": cashflow,
    }
