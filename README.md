# StockView

A Streamlit app that generates a lightweight research report for a single stock ticker. Phase 1 uses `yfinance` for price history and basic fundamentals, plus a simplified DCF model.

## Features

- **Price Charts & Technical Indicators**: Moving averages, RSI, volatility metrics
- **Fundamentals**: P/E ratios, profit margins, revenue growth, cash flow
- **AI-Powered News Summary**: Recent news headlines with AI-generated summaries (optional)
- **Recommendations**: Automated analysis combining technicals and fundamentals

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Optional: Enable AI News Summarization

The app can use Google's Gemini API to generate AI-powered summaries of recent news. To enable this feature:

1. Get a Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Set the environment variable:

```bash
export GEMINI_API_KEY="your-api-key-here"
```

3. (Optional) Customize the Gemini model:

```bash
export GEMINI_MODEL="gemini-1.5-flash"  # Default: gemini-2.5-flash
```

**Available models:**
- `gemini-2.5-flash` (default) - Latest fast model, good for summarization
- `gemini-1.5-flash` - Stable, fast, cost-effective
- `gemini-1.5-pro` - More capable, higher quality, slower and more expensive
- `gemini-2.0-flash-exp` - Experimental version

**Note**: The app works without these keys — you'll still see news headlines and links, just without AI summaries.

## Run

```bash
streamlit run app.py
```

## Notes and Limitations

- Data comes from `yfinance` and may be missing, stale, or inaccurate for some tickers.
- Fundamentals coverage varies by company; missing fields are shown as `N/A`.
- The DCF model is simplified: it uses a 5-year FCF forecast, a terminal value, and does **not** adjust for net debt, dilution, or other balance sheet items.
- **News**: News quality from `yfinance` varies by ticker; some tickers may have limited or no news coverage.
- **AI Summaries**: AI-generated summaries are based on headlines and snippets only (not full article text) and may be incomplete or miss context.
- This app is for educational use only and is **not** financial advice.
