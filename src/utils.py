from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import math

import numpy as np


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def safe_get(mapping: Optional[dict], key: str, default: Any = None) -> Any:
    if not mapping:
        return default
    return mapping.get(key, default)


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (float, np.floating)) and math.isnan(value):
        return True
    return False


def format_number(value: Any, decimals: int = 2) -> str:
    if is_missing(value):
        return "N/A"
    try:
        return f"{float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return "N/A"


def format_percent(value: Any, decimals: int = 1) -> str:
    if is_missing(value):
        return "N/A"
    try:
        return f"{float(value) * 100:.{decimals}f}%"
    except (TypeError, ValueError):
        return "N/A"


def format_currency(value: Any, decimals: int = 2) -> str:
    if is_missing(value):
        return "N/A"
    try:
        return f"${float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return "N/A"


def abbreviate_number(value: Any) -> str:
    if is_missing(value):
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"

    sign = "-" if number < 0 else ""
    number = abs(number)
    for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if number >= threshold:
            return f"{sign}{number / threshold:.2f}{suffix}"
    return f"{sign}{number:,.2f}"


def format_currency_abbrev(value: Any) -> str:
    if is_missing(value):
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return f"${abbreviate_number(number)}"


def as_of_timestamp() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")


def to_float(value: Any) -> Optional[float]:
    if is_missing(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_divide(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator
