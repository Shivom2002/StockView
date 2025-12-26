from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from .utils import safe_divide, to_float


def _extract_cashflow_value(cashflow: Optional[pd.DataFrame], labels: list[str]) -> Optional[float]:
    if cashflow is None or cashflow.empty:
        return None
    for label in labels:
        if label in cashflow.index:
            series = cashflow.loc[label]
            if isinstance(series, pd.Series):
                series = series.dropna()
                if series.empty:
                    continue
                return to_float(series.iloc[0])
    return None


def parse_fundamentals(info: Dict[str, Any], cashflow: Optional[pd.DataFrame], price: Optional[float]) -> Dict[str, Any]:
    market_cap = to_float(info.get("marketCap"))
    trailing_pe = to_float(info.get("trailingPE"))
    forward_pe = to_float(info.get("forwardPE"))
    price_to_sales = to_float(info.get("priceToSalesTrailing12Months"))
    profit_margin = to_float(info.get("profitMargins"))
    revenue_growth = to_float(info.get("revenueGrowth"))
    total_revenue = to_float(info.get("totalRevenue"))
    total_debt = to_float(info.get("totalDebt"))

    fcf = to_float(info.get("freeCashflow"))
    if fcf is None:
        fcf = _extract_cashflow_value(cashflow, ["Free Cash Flow", "FreeCashFlow"])

    if fcf is None:
        operating_cf = _extract_cashflow_value(
            cashflow,
            [
                "Total Cash From Operating Activities",
                "Operating Cash Flow",
                "Net Cash Provided by Operating Activities",
            ],
        )
        capex = _extract_cashflow_value(
            cashflow,
            [
                "Capital Expenditures",
                "CapitalExpenditures",
            ],
        )
        if operating_cf is not None and capex is not None:
            fcf = operating_cf - capex

    fcf_margin = safe_divide(fcf, total_revenue) if fcf is not None and total_revenue else None

    shares_outstanding = to_float(info.get("sharesOutstanding"))
    if shares_outstanding is None and market_cap and price:
        shares_outstanding = market_cap / price

    return {
        "market_cap": market_cap,
        "trailing_pe": trailing_pe,
        "forward_pe": forward_pe,
        "price_to_sales": price_to_sales,
        "profit_margin": profit_margin,
        "revenue_growth": revenue_growth,
        "fcf": fcf,
        "fcf_margin": fcf_margin,
        "total_debt": total_debt,
        "shares_outstanding": shares_outstanding,
        "total_revenue": total_revenue,
    }
