from __future__ import annotations

from typing import Any, Dict, Optional

import math

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from src.data import fetch_fundamentals, fetch_price_history
# from src.dcf import apply_scenario, build_sensitivity_table, dcf_valuation
from src.fundamentals import parse_fundamentals
from src.gemini_summarizer import summarize_news
from src.news import fetch_news_yfinance, filter_recent_news, pick_top_links
from src.report import render_markdown
from src.technicals import compute_indicators, compute_long_term_pullback_signals
from src.utils import (
    as_of_timestamp,
    clamp,
    format_currency,
    format_currency_abbrev,
    format_number,
    format_percent,
    safe_divide,
    to_float,
)

st.set_page_config(page_title="StockView", layout="wide", initial_sidebar_state="collapsed")

st.markdown("<h1 style='text-align: center;'>StockView</h1>", unsafe_allow_html=True)

# Custom CSS for styling
st.markdown("""
<style>
    /* Main app background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }

    /* Main content area background */
    .main .block-container {
        background-color: rgba(255, 255, 255, 0.95);
        padding: 2rem 3rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
    }

    /* Title styling */
    h1 {
        color: #1a1a2e;
        font-weight: 700;
        margin-bottom: 2rem;
    }

    /* Form styling - remove border and make transparent */
    .stForm {
        background-color: transparent;
        border: none !important;
        padding: 0;
    }

    /* Remove form border container */
    [data-testid="stForm"] {
        border: none !important;
        background-color: transparent !important;
    }

    /* Text input styling */
    .stTextInput > div > div > input {
        background-color: #ffffff;
        border: 2px solid #e0e0e0;
        border-radius: 50px;
        padding: 12px 24px;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        height: 48px;
    }

    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
        outline: none;
    }

    /* Remove label spacing for text input */
    .stTextInput > label {
        display: none;
    }

    .stTextInput {
        margin-bottom: 0px;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 12px 32px;
        font-size: 16px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        width: 100%;
        height: 48px;
        margin-top: 0px;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(102, 126, 234, 0.4);
    }

    /* Form button vertical alignment - force same height */
    .stForm .stButton {
        margin-top: 0px;
        padding-top: 0px;
        display: flex;
        align-items: center;
    }

    /* Align columns in form */
    .stForm [data-testid="column"] {
        display: flex;
        align-items: center;
    }

    /* Tab container styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 10px;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.05);
    }

    /* Individual tab styling */
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        padding: 0px 30px;
        background-color: #ffffff;
        border-radius: 8px;
        font-size: 20px;
        font-weight: 600;
        color: #262730;
        border: 2px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }

    /* Tab hover effect */
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e8eaf0;
        border-color: #4CAF50;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }

    /* Active/selected tab styling */
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-color: #667eea;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }

    /* Tab panel (content area) styling */
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 20px;
    }

    /* Metric cards */
    .stMetric {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    /* News tab typography improvements */
    .stTabs [data-baseweb="tab-panel"] h2 {
        font-size: 28px;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 1.5rem;
        margin-top: 1rem;
    }

    .stTabs [data-baseweb="tab-panel"] h3 {
        font-size: 24px;
        font-weight: 600;
        color: #2d3748;
        margin-bottom: 1rem;
        margin-top: 1.5rem;
    }

    /* News headlines styling */
    .stTabs [data-baseweb="tab-panel"] a {
        font-size: 17px;
        line-height: 1.6;
        color: #667eea;
        text-decoration: none;
        font-weight: 500;
    }

    .stTabs [data-baseweb="tab-panel"] a:hover {
        color: #764ba2;
        text-decoration: underline;
    }

    /* News metadata (publisher, time) */
    .stTabs [data-baseweb="tab-panel"] em {
        font-size: 14px;
        color: #718096;
    }

    /* Bullet points styling */
    .stTabs [data-baseweb="tab-panel"] li {
        font-size: 17px;
        line-height: 1.8;
        color: #2d3748;
        margin-bottom: 0.75rem;
    }

    /* General text in tabs */
    .stTabs [data-baseweb="tab-panel"] p {
        font-size: 16px;
        line-height: 1.7;
        color: #2d3748;
    }

    /* Bold text emphasis */
    .stTabs [data-baseweb="tab-panel"] strong {
        color: #1a1a2e;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar controls for Long-Term Pullback strategy
st.sidebar.header("Strategy Parameters")
st.sidebar.markdown("**Long-Term Pullback-in-Trend**")
pullback_band = st.sidebar.number_input(
    "Pullback Band (%)",
    min_value=0.0,
    max_value=5.0,
    value=0.5,
    step=0.1,
    help="Tolerance above EMA50 for pullback detection (default 0.5%)",
) / 100  # Convert to decimal
rsi_min = st.sidebar.number_input(
    "RSI Min",
    min_value=20,
    max_value=60,
    value=40,
    step=5,
    help="Minimum RSI for entry confirmation (default 40)",
)
rsi_max = st.sidebar.number_input(
    "RSI Max",
    min_value=40,
    max_value=80,
    value=55,
    step=5,
    help="Maximum RSI for entry confirmation (default 55)",
)
atr_k = st.sidebar.number_input(
    "ATR Multiplier (Stop)",
    min_value=1.0,
    max_value=5.0,
    value=2.0,
    step=0.5,
    help="ATR multiplier for stop loss calculation (default 2.0)",
)

with st.form("analyze_form"):
    col1, col2 = st.columns([4, 1])
    with col1:
        ticker_input = st.text_input("", placeholder="Enter Ticker")
    with col2:
        submitted = st.form_submit_button("Analyze")


def _last_value(series: pd.Series) -> Optional[float]:
    if series is None:
        return None
    series = series.dropna()
    if series.empty:
        return None
    return float(series.iloc[-1])


def _is_valid_number(value: Optional[float]) -> bool:
    return value is not None and not pd.isna(value) and math.isfinite(value)


def _unique_clamped(values: list[float], min_value: float, max_value: float) -> list[float]:
    seen = set()
    ordered = []
    for value in values:
        clamped = clamp(value, min_value, max_value)
        if clamped not in seen:
            ordered.append(clamped)
            seen.add(clamped)
    return ordered


def _build_recommendation(
    trend_signal: str,
    momentum_signal: str,
    valuation_signal: str,
    fundamentals: Dict[str, Any],
    tech: Dict[str, Any],
    dcf: Dict[str, Any],
) -> Dict[str, Any]:
    notes = []
    profit_margin = to_float(fundamentals.get("profit_margin"))
    revenue_growth = to_float(fundamentals.get("revenue_growth"))
    trailing_pe = to_float(fundamentals.get("trailing_pe"))

    if profit_margin is not None:
        if profit_margin >= 0.2:
            notes.append("healthy profit margins")
        elif profit_margin <= 0.05:
            notes.append("thin profit margins")

    if revenue_growth is not None:
        if revenue_growth > 0:
            notes.append("positive revenue growth")
        elif revenue_growth < 0:
            notes.append("revenue contraction")

    if trailing_pe is not None:
        if trailing_pe >= 25:
            notes.append("elevated trailing P/E")
        elif trailing_pe <= 15:
            notes.append("relatively low trailing P/E")

    notes = notes[:2]
    notes_text = " and ".join(notes) if notes else "mixed fundamentals based on available data"

    # valuation_text = (
    #     f"Valuation looks {valuation_signal.lower()} versus the DCF range." if dcf.get("enabled") else "DCF is unavailable due to missing FCF data."
    # )

    summary = (
        f"Trend is {trend_signal.lower()} with momentum signaling {momentum_signal.lower()}. "
        # f"{valuation_text} "
        f"Key fundamentals show {notes_text}."
    )

    reasons = []
    if trend_signal == "Bullish":
        reasons.append("Price holds above long-term averages")
    elif trend_signal == "Bearish":
        reasons.append("Price remains below long-term averages")
    else:
        reasons.append("Trend is mixed across averages")

    if momentum_signal == "Oversold":
        reasons.append("RSI indicates oversold conditions")
    elif momentum_signal == "Overbought":
        reasons.append("RSI indicates strong momentum")
    else:
        reasons.append("RSI is neutral")

    if profit_margin is not None:
        reasons.append(f"Profit margin at {format_percent(profit_margin)}")

    fallback_reasons = [
        "Revenue growth is a key watch for re-rating",
        "FCF strength supports valuation assumptions",
        "Market sentiment can shift quickly",
    ]
    for item in fallback_reasons:
        if len(reasons) >= 3:
            break
        if item not in reasons:
            reasons.append(item)

    risks = []
    volatility = to_float(tech.get("volatility"))
    max_drawdown = to_float(tech.get("max_drawdown"))

    # if valuation_signal == "Overvalued":
    #     risks.append("DCF suggests limited upside at current price")
    if volatility is not None and volatility > 0.4:
        risks.append("Elevated volatility can pressure near-term returns")
    if max_drawdown is not None and max_drawdown < -0.3:
        risks.append("Recent drawdowns highlight downside risk")

    # if dcf.get("enabled") is False:
    #     risks.append("DCF disabled due to missing FCF data")

    fallback_risks = [
        "Macro conditions and rates can shift valuation",
        "Fundamentals may change with upcoming earnings",
        "Data availability is limited in Phase 1",
    ]
    for item in fallback_risks:
        if len(risks) >= 3:
            break
        if item not in risks:
            risks.append(item)

    return {
        "summary": summary,
        "reasons": reasons[:3],
        "risks": risks[:3],
    }


analysis_result = None

if submitted:
    ticker = ticker_input.strip().upper()
    if not ticker:
        st.error("Please enter a valid ticker symbol.")
    else:
        history = fetch_price_history(ticker)
        if history.empty:
            st.error("No price history found for this ticker. Please check the symbol and try again.")
        else:
            indicators = compute_indicators(history)

            close_series = history["Close"].dropna()
            current_price = _last_value(close_series)
            previous_price = _last_value(close_series.iloc[:-1]) if len(close_series) > 1 else None
            price_change_pct = safe_divide(current_price - previous_price, previous_price) if previous_price else None

            fundamentals_raw = fetch_fundamentals(ticker)
            fundamentals = parse_fundamentals(
                fundamentals_raw.get("info", {}),
                fundamentals_raw.get("cashflow"),
                current_price,
            )

            rsi_value = _last_value(indicators.get("rsi"))
            sma20 = _last_value(indicators.get("sma20"))
            sma50 = _last_value(indicators.get("sma50"))
            sma200 = _last_value(indicators.get("sma200"))
            volatility = indicators.get("volatility")
            max_drawdown = indicators.get("max_drawdown")

            trend_signal = "Neutral"
            if current_price is not None and sma200 is not None and sma50 is not None:
                if current_price > sma200 and sma50 > sma200:
                    trend_signal = "Bullish"
                elif current_price < sma200 and sma50 < sma200:
                    trend_signal = "Bearish"

            momentum_signal = "Neutral"
            if rsi_value is not None:
                if rsi_value < 30:
                    momentum_signal = "Oversold"
                elif rsi_value > 70:
                    momentum_signal = "Overbought"

            # base_growth = growth_pct / 100
            # base_discount = discount_pct / 100
            # base_terminal = terminal_pct / 100
            # margin_of_safety = margin_pct / 100

            # growth_range = (0.0, 0.25)
            # discount_range = (0.07, 0.14)
            # terminal_range = (0.01, 0.04)

            # fcf = fundamentals.get("fcf")
            # shares = fundamentals.get("shares_outstanding")

            # dcf_enabled = fcf is not None and shares is not None and shares > 0
            dcf_enabled = False
            dcf_details: Dict[str, Any] = {
                "enabled": dcf_enabled,
                # "base_growth": base_growth,
                # "base_discount": base_discount,
                # "base_terminal": base_terminal,
            }
            sensitivity_table = None

            # if dcf_enabled:
            #     base = dcf_valuation(
            #         fcf0=fcf,
            #         shares=shares,
            #         growth_rate=base_growth,
            #         discount_rate=base_discount,
            #         terminal_growth=base_terminal,
            #     )

            #     bear_inputs = apply_scenario(
            #         base_growth,
            #         base_discount,
            #         base_terminal,
            #         growth_delta=-0.04,
            #         discount_delta=0.015,
            #         terminal_delta=-0.005,
            #         growth_range=growth_range,
            #         discount_range=discount_range,
            #         terminal_range=terminal_range,
            #     )
            #     bull_inputs = apply_scenario(
            #         base_growth,
            #         base_discount,
            #         base_terminal,
            #         growth_delta=0.04,
            #         discount_delta=-0.01,
            #         terminal_delta=0.005,
            #         growth_range=growth_range,
            #         discount_range=discount_range,
            #         terminal_range=terminal_range,
            #     )

            #     bear = dcf_valuation(
            #         fcf0=fcf,
            #         shares=shares,
            #         growth_rate=bear_inputs["growth"],
            #         discount_rate=bear_inputs["discount"],
            #         terminal_growth=bear_inputs["terminal"],
            #     )
            #     bull = dcf_valuation(
            #         fcf0=fcf,
            #         shares=shares,
            #         growth_rate=bull_inputs["growth"],
            #         discount_rate=bull_inputs["discount"],
            #         terminal_growth=bull_inputs["terminal"],
            #     )

            #     discount_rates = _unique_clamped(
            #         [base_discount - 0.02, base_discount - 0.01, base_discount, base_discount + 0.01, base_discount + 0.02],
            #         *discount_range,
            #     )
            #     terminal_growths = _unique_clamped([0.01, 0.02, 0.025, 0.03, 0.04], *terminal_range)
            #     sensitivity_table = build_sensitivity_table(
            #         fcf0=fcf,
            #         shares=shares,
            #         growth_rate=base_growth,
            #         discount_rates=discount_rates,
            #         terminal_growth_rates=terminal_growths,
            #     )

            #     dcf_details.update(
            #         {
            #             "base_value": base["value_per_share"],
            #             "bear_value": bear["value_per_share"],
            #             "bull_value": bull["value_per_share"],
            #             "bear_inputs": bear_inputs,
            #             "bull_inputs": bull_inputs,
            #         }
            #     )

            valuation_signal = "N/A"
            # if dcf_enabled and current_price is not None:
            #     base_value = dcf_details.get("base_value")
            #     bull_value = dcf_details.get("bull_value")
            #     if _is_valid_number(base_value) and _is_valid_number(bull_value):
            #         if current_price < base_value * (1 - margin_of_safety):
            #             valuation_signal = "Undervalued"
            #         elif current_price > bull_value:
            #             valuation_signal = "Overvalued"
            #         else:
            #             valuation_signal = "Fair"

            recommendation = _build_recommendation(
                trend_signal,
                momentum_signal,
                valuation_signal,
                fundamentals,
                {"volatility": volatility, "max_drawdown": max_drawdown},
                dcf_details,
            )

            results = {
                "ticker": ticker,
                "timestamp": as_of_timestamp(),
                "price": current_price,
                "price_change_pct": price_change_pct,
                "technicals": {
                    "rsi": rsi_value,
                    "volatility": volatility,
                    "max_drawdown": max_drawdown,
                },
                "fundamentals": fundamentals,
                "dcf": dcf_details,
                "recommendation": recommendation,
                "signals": {
                    "trend": trend_signal,
                    "momentum": momentum_signal,
                    "valuation": valuation_signal,
                },
                # "margin_of_safety": margin_of_safety,
            }

            analysis_result = {
                "history": history,
                "indicators": indicators,
                "sensitivity": sensitivity_table,
                "results": results,
            }
            st.session_state["analysis_result"] = analysis_result

if "analysis_result" in st.session_state:
    analysis_result = st.session_state["analysis_result"]

if analysis_result:
    results = analysis_result["results"]
    history = analysis_result["history"]
    indicators = analysis_result["indicators"]
    sensitivity_table = analysis_result["sensitivity"]

    st.caption(f"Last updated: {results.get('timestamp')}")

    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["📈 Stock", "📰 News", "📊 Strategy"])

    # Tab 1: Stock Analysis (Fundamentals + Technicals + Chart)
    with tab1:
        snapshot_cols = st.columns(4)
        snapshot_cols[0].metric(
            "Current Price",
            format_currency(results.get("price")),
            format_percent(results.get("price_change_pct")),
        )
        snapshot_cols[1].metric("Market Cap", format_currency_abbrev(results["fundamentals"].get("market_cap")))
        snapshot_cols[2].metric("Trailing P/E", format_number(results["fundamentals"].get("trailing_pe")))
        snapshot_cols[3].metric("Forward P/E", format_number(results["fundamentals"].get("forward_pe")))

        signal_cols = st.columns(3)
        signal_cols[0].markdown(f"**Trend:** {results['signals']['trend']}")
        signal_cols[1].markdown(f"**Momentum:** {results['signals']['momentum']}")
        signal_cols[2].markdown(f"**Valuation:** {results['signals']['valuation']}")

        st.subheader("Price & Moving Averages")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(history.index, history["Close"], label="Close", linewidth=1.5, color="black")

        # Plot SMAs
        if indicators.get("sma20") is not None:
            ax.plot(indicators["sma20"].index, indicators["sma20"], label="SMA 20", alpha=0.7)
        if indicators.get("sma50") is not None:
            ax.plot(indicators["sma50"].index, indicators["sma50"], label="SMA 50", alpha=0.7)
        if indicators.get("sma200") is not None:
            ax.plot(indicators["sma200"].index, indicators["sma200"], label="SMA 200", alpha=0.7)

        ax.set_ylabel("Price")
        ax.legend(loc="upper left")
        ax.grid(alpha=0.2)
        st.pyplot(fig, clear_figure=True)

        tech_col, fund_col = st.columns([1, 1])
        with tech_col:
            st.subheader("Technicals")
            rsi_value = results["technicals"].get("rsi")
            rsi_comment = "RSI measures momentum; values below 30 suggest oversold, above 70 suggest overbought."
            st.metric("RSI (14)", format_number(rsi_value))
            st.caption(rsi_comment)
            st.metric("Annualized Volatility", format_percent(results["technicals"].get("volatility")))
            st.metric("1Y Max Drawdown", format_percent(results["technicals"].get("max_drawdown")))

        with fund_col:
            st.subheader("Fundamentals")
            fundamentals = results["fundamentals"]
            st.metric("Market Cap", format_currency_abbrev(fundamentals.get("market_cap")))
            st.metric("Trailing P/E", format_number(fundamentals.get("trailing_pe")))
            st.metric("Forward P/E", format_number(fundamentals.get("forward_pe")))
            st.metric("P/S Ratio", format_number(fundamentals.get("price_to_sales")))
            st.metric("Profit Margin", format_percent(fundamentals.get("profit_margin")))
            st.metric("Revenue Growth", format_percent(fundamentals.get("revenue_growth")))
            st.metric("Free Cash Flow", format_currency_abbrev(fundamentals.get("fcf")))
            st.metric("FCF Margin", format_percent(fundamentals.get("fcf_margin")))
            st.metric("Total Debt", format_currency_abbrev(fundamentals.get("total_debt")))

        st.subheader("Recommendation")
        recommendation = results.get("recommendation", {})
        st.write(recommendation.get("summary", "N/A"))
        reasons_col, risks_col = st.columns(2)
        with reasons_col:
            st.markdown("**Reasons**")
            for item in recommendation.get("reasons", []):
                st.write(f"- {item}")
        with risks_col:
            st.markdown("**Risks / Watch items**")
            for item in recommendation.get("risks", []):
                st.write(f"- {item}")

    # Tab 2: News & AI Summary
    with tab2:
        ticker = results.get("ticker")

        # Fetch and filter news
        all_news = fetch_news_yfinance(ticker)
        recent_news = filter_recent_news(all_news, max_age_hours=72)
        top_links = pick_top_links(recent_news, k=5)

        if not recent_news:
            st.info("No recent news found for this ticker in the last 3 days.")
        else:
            # Display top 5 articles with summaries
            st.subheader("Recent Headlines")
            for item in top_links:
                title = item.get("title", "")
                link = item.get("link", "")
                publisher = item.get("publisher", "Unknown")
                published_at = item.get("published_at")
                summary = item.get("summary", "")

                time_str = published_at.strftime("%b %d, %H:%M UTC") if published_at else "Unknown"

                # Article card with title, metadata, and summary
                st.markdown(f"**[{title}]({link})**")
                st.markdown(f"*{publisher} • {time_str}*")
                if summary:
                    st.markdown(f"{summary}")
                st.markdown("---")  # Divider between articles

            # AI Summary section
            st.subheader("AI Summary")

            # Try to generate summary with more context (15 articles)
            summary_data, error_msg = summarize_news(recent_news, ticker, max_items_for_context=15)

            if summary_data:
                # Display bullets
                bullets = summary_data.get("bullets", [])
                if bullets:
                    for bullet in bullets:
                        st.write(f"- {bullet}")

                # Display takeaway
                takeaway = summary_data.get("takeaway", "")
                if takeaway:
                    st.markdown("")  # Add spacing
                    st.markdown(f"**Overall Takeaway:** {takeaway}")

                # Display watch items if available
                watch_items = summary_data.get("watch_items", [])
                if watch_items:
                    st.markdown("")  # Add spacing
                    st.markdown("**Watch Items:**")
                    for item in watch_items:
                        st.write(f"- {item}")

                # Show info if there was a parsing issue
                if error_msg and "parsing" in error_msg.lower():
                    st.caption("ℹ️ Summary generated with partial parsing. Refresh page if content appears incomplete.")
            else:
                # Show error message
                if "GEMINI_API_KEY" in error_msg:
                    st.info(
                        "💡 To enable AI-powered news summarization, set the `GEMINI_API_KEY` environment variable. "
                        "See README for instructions."
                    )
                else:
                    st.warning(f"Could not generate AI summary: {error_msg}")
                    st.caption("Headlines are still available above.")

    # Tab 3: Strategy Analysis
    with tab3:
        try:
            # Compute strategy signals on demand
            strategy_df = compute_long_term_pullback_signals(
                history,
                pullback_band=pullback_band,
                rsi_min=int(rsi_min),
                rsi_max=int(rsi_max),
            )

            # Get latest status
            latest_row = strategy_df.iloc[-1]
            trend_ok = bool(latest_row["ltp_trend_ok"])
            pullback_ok = bool(latest_row["ltp_pullback_ok"])
            rsi_ok = bool(latest_row["ltp_rsi_ok"])
            entry_signal_today = bool(latest_row["ltp_entry"])
            in_position = bool(latest_row["ltp_position"] == 1)

            # Display current status
            st.subheader("Current Status")
            status_cols = st.columns(5)
            status_cols[0].metric("Trend OK", "Yes" if trend_ok else "No")
            status_cols[1].metric("Pullback OK", "Yes" if pullback_ok else "No")
            status_cols[2].metric("RSI OK", "Yes" if rsi_ok else "No")
            status_cols[3].metric("Entry Signal", "Yes" if entry_signal_today else "No")
            status_cols[4].metric("In Position", "Yes" if in_position else "No")

            # If in position, show entry details
            if in_position:
                # Find the most recent entry
                entries = strategy_df[strategy_df["ltp_entry"]].copy()
                if not entries.empty:
                    latest_entry = entries.iloc[-1]
                    entry_date = entries.index[-1]
                    entry_price = float(latest_entry["Close"])
                    entry_atr = float(latest_entry["atr14"])

                    # Calculate stop and target
                    atr_k_value = float(atr_k)
                    stop_price = entry_price - atr_k_value * entry_atr
                    target_price = entry_price + 3 * entry_atr

                    # Calculate R multiple
                    risk_per_share = entry_price - stop_price
                    reward_per_share = target_price - entry_price
                    r_multiple = safe_divide(reward_per_share, risk_per_share)

                    st.subheader("Current Position Details")
                    pos_cols = st.columns(4)
                    pos_cols[0].metric("Entry Date", entry_date.strftime("%Y-%m-%d"))
                    pos_cols[1].metric("Entry Price", format_currency(entry_price))
                    pos_cols[2].metric("Suggested Stop", format_currency(stop_price))
                    pos_cols[3].metric("Target (3R)", format_currency(target_price))

                    st.markdown(
                        f"**Risk Metrics:** Stop = Entry - {atr_k_value:.1f}x ATR, "
                        f"Target = Entry + 3x ATR, R:R = {format_number(r_multiple)}"
                    )
                    ema200_val = float(latest_row["ema200"])
                    st.caption(f"Exit condition: Close < EMA200 (currently ${ema200_val:.2f})")
            else:
                st.info("No active position. Waiting for entry signal.")

            # Strategy chart with EMAs and entry/exit markers
            st.subheader("Strategy Chart")
            fig_strategy, ax_strategy = plt.subplots(figsize=(10, 4))
            ax_strategy.plot(history.index, history["Close"], label="Close", linewidth=1.5, color="black")

            # Plot EMAs
            ax_strategy.plot(strategy_df.index, strategy_df["ema20"], label="EMA 20", linewidth=1, alpha=0.7)
            ax_strategy.plot(strategy_df.index, strategy_df["ema50"], label="EMA 50", linewidth=1, alpha=0.7)
            ax_strategy.plot(strategy_df.index, strategy_df["ema200"], label="EMA 200", linewidth=1, alpha=0.7)

            # Mark entry and exit points (last 2 only)
            entry_dates = strategy_df[strategy_df["ltp_entry"]].index
            entry_prices = strategy_df.loc[entry_dates, "Close"]
            if not entry_dates.empty:
                last_2_entry_dates = entry_dates[-2:]
                last_2_entry_prices = entry_prices[-2:]
                ax_strategy.scatter(last_2_entry_dates, last_2_entry_prices, marker="^", color="green", s=100, label="Entry", zorder=5)

            exit_dates = strategy_df[strategy_df["ltp_exit"] & (strategy_df["ltp_position"].shift(1) == 1)].index
            exit_prices = strategy_df.loc[exit_dates, "Close"]
            if not exit_dates.empty:
                last_2_exit_dates = exit_dates[-2:]
                last_2_exit_prices = exit_prices[-2:]
                ax_strategy.scatter(last_2_exit_dates, last_2_exit_prices, marker="v", color="red", s=100, label="Exit", zorder=5)

            ax_strategy.set_ylabel("Price")
            ax_strategy.legend(loc="upper left")
            ax_strategy.grid(alpha=0.2)
            st.pyplot(fig_strategy, clear_figure=True)

            # Show last 2 entry/exit events
            st.subheader("Recent Entry/Exit Events")
            events = []

            # Collect all entry events
            entries = strategy_df[strategy_df["ltp_entry"]].copy()
            for idx in entries.index:
                events.append({
                    "Date": idx.strftime("%Y-%m-%d"),
                    "Event": "Entry",
                    "Price": format_currency(entries.loc[idx, "Close"]),
                    "RSI": format_number(entries.loc[idx, "rsi14"]),
                    "EMA20": format_currency(entries.loc[idx, "ema20"]),
                })

            # Collect all exit events (where ltp_exit is True and previous position was 1)
            exits = strategy_df[strategy_df["ltp_exit"] & (strategy_df["ltp_position"].shift(1) == 1)].copy()
            for idx in exits.index:
                events.append({
                    "Date": idx.strftime("%Y-%m-%d"),
                    "Event": "Exit",
                    "Price": format_currency(exits.loc[idx, "Close"]),
                    "RSI": format_number(exits.loc[idx, "rsi14"]),
                    "EMA20": format_currency(exits.loc[idx, "ema20"]),
                })

            if events:
                # Sort by date descending and take last 2
                events_df = pd.DataFrame(events)
                events_df = events_df.sort_values("Date", ascending=False).head(2).reset_index(drop=True)
                st.dataframe(events_df)
            else:
                st.info("No entry/exit events found in the selected timeframe.")
        except Exception as e:
            st.warning(f"Strategy section error: {str(e)}")

    # st.subheader("DCF Lite")
    # if results["dcf"].get("enabled"):
    #     base_value = results["dcf"].get("base_value")
    #     bear_value = results["dcf"].get("bear_value")
    #     bull_value = results["dcf"].get("bull_value")
    #     current_price = results.get("price")

    #     dcf_cols = st.columns(3)
    #     dcf_cols[0].metric("Bear Value / Share", format_currency(bear_value))
    #     dcf_cols[1].metric("Base Value / Share", format_currency(base_value))
    #     dcf_cols[2].metric("Bull Value / Share", format_currency(bull_value))

    #     if current_price is not None and base_value is not None:
    #         upside_base = safe_divide(base_value - current_price, current_price)
    #         upside_bear = safe_divide(bear_value - current_price, current_price) if bear_value is not None else None
    #         upside_bull = safe_divide(bull_value - current_price, current_price) if bull_value is not None else None
    #         st.write(
    #             f"Implied upside/downside vs current price: Bear {format_percent(upside_bear)}, Base {format_percent(upside_base)}, Bull {format_percent(upside_bull)}"
    #         )

    #     st.caption("DCF assumes no net debt adjustment or dilution in Phase 1.")

    #     if sensitivity_table is not None:
    #         st.markdown("**Sensitivity (Value per Share)**")
    #         formatted_sensitivity_table = sensitivity_table.applymap(format_currency)
    #         try:
    #             st.dataframe(formatted_sensitivity_table, use_container_width=True)
    #         except TypeError:
    #             st.dataframe(formatted_sensitivity_table)
    # else:
    #     st.warning("FCF unavailable; DCF disabled (Phase 1).")

    st.subheader("Export")
    markdown_report = render_markdown(results)
    st.download_button(
        "Export as Markdown",
        data=markdown_report,
        file_name=f"{results.get('ticker', 'report')}_report.md",
        mime="text/markdown",
    )
