# Quick Streamlit UI

This repository includes a small Streamlit app to quickly run the linear-regression next-day stock close predictor.

How to run locally:

1. Install dependencies:

	pip install -r requirements.txt

2. Start the Streamlit UI:

	streamlit run app.py

Enter a ticker (e.g. AAPL) and a date range, then click "Run prediction". The app will show evaluation metrics, an actual-vs-predicted plot, and a next-day predicted close.

