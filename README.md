# Ticker Analyzer (Phase 1)

A Streamlit app that generates a lightweight research report for a single stock ticker. Phase 1 uses `yfinance` for price history and basic fundamentals, plus a simplified DCF model.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Notes and Limitations

- Data comes from `yfinance` and may be missing, stale, or inaccurate for some tickers.
- Fundamentals coverage varies by company; missing fields are shown as `N/A`.
- The DCF model is simplified: it uses a 5-year FCF forecast, a terminal value, and does **not** adjust for net debt, dilution, or other balance sheet items.
- This app is for educational use only and is **not** financial advice.
