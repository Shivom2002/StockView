from __future__ import annotations

from typing import Dict, Iterable
import pandas as pd

from .utils import clamp


def _normalize_terminal_growth(discount_rate: float, terminal_growth: float) -> float:
    if discount_rate <= terminal_growth:
        return max(0.0, discount_rate - 0.005)
    return terminal_growth


def dcf_valuation(
    fcf0: float,
    shares: float,
    growth_rate: float,
    discount_rate: float,
    terminal_growth: float,
    years: int = 5,
) -> Dict[str, object]:
    terminal_growth = _normalize_terminal_growth(discount_rate, terminal_growth)

    cashflows = []
    pv_cashflows = []
    for year in range(1, years + 1):
        fcf_t = fcf0 * ((1 + growth_rate) ** year)
        cashflows.append(fcf_t)
        pv_cashflows.append(fcf_t / ((1 + discount_rate) ** year))

    fcf_year_n = cashflows[-1]
    terminal_value = (fcf_year_n * (1 + terminal_growth)) / (discount_rate - terminal_growth)
    pv_terminal = terminal_value / ((1 + discount_rate) ** years)

    enterprise_value = sum(pv_cashflows) + pv_terminal
    value_per_share = enterprise_value / shares if shares else float("nan")

    return {
        "enterprise_value": enterprise_value,
        "value_per_share": value_per_share,
        "cashflows": cashflows,
        "pv_cashflows": pv_cashflows,
        "terminal_value": terminal_value,
        "pv_terminal": pv_terminal,
        "discount_rate": discount_rate,
        "terminal_growth": terminal_growth,
    }


def build_sensitivity_table(
    fcf0: float,
    shares: float,
    growth_rate: float,
    discount_rates: Iterable[float],
    terminal_growth_rates: Iterable[float],
) -> pd.DataFrame:
    data: dict[str, list[float]] = {}
    for tg in terminal_growth_rates:
        values = []
        for dr in discount_rates:
            valuation = dcf_valuation(
                fcf0=fcf0,
                shares=shares,
                growth_rate=growth_rate,
                discount_rate=dr,
                terminal_growth=tg,
            )
            values.append(valuation["value_per_share"])
        data[f"{tg:.1%}"] = values

    index = [f"{dr:.1%}" for dr in discount_rates]
    return pd.DataFrame(data, index=index)


def apply_scenario(
    base_growth: float,
    base_discount: float,
    base_terminal: float,
    growth_delta: float,
    discount_delta: float,
    terminal_delta: float,
    growth_range: tuple[float, float],
    discount_range: tuple[float, float],
    terminal_range: tuple[float, float],
) -> Dict[str, float]:
    growth = clamp(base_growth + growth_delta, *growth_range)
    discount = clamp(base_discount + discount_delta, *discount_range)
    terminal = clamp(base_terminal + terminal_delta, *terminal_range)
    return {
        "growth": growth,
        "discount": discount,
        "terminal": terminal,
    }
