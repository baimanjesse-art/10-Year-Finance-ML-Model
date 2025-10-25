import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="Quick Stock Predictor", layout="centered")

st.title("10-Year Finance ML — Quick Stock Predictor")
st.write("A quick linear-regression next-day stock close predictor using yfinance.")

ticker = st.text_input("Ticker", value="AAPL").upper()
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=pd.to_datetime("2015-01-01"))
with col2:
    end_date = st.date_input("End Date", value=pd.to_datetime("2025-10-01"))

run = st.button("Run prediction")

@st.cache_data
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
    return df

if run:
    with st.spinner("Downloading data and training model..."):
        data = load_data(ticker, start_date, end_date)
        if data is None or data.empty:
            st.error("No data found for that ticker / date range.")
        else:
            data['Target'] = data['Close'].shift(-1)
            data = data.dropna()
            X = data[['Open','High','Low','Close','Volume']]
            y = data['Target']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            st.metric("Mean Squared Error", f"{mse:.4f}")
            st.metric("R² Score", f"{r2:.4f}")
            # plot
            fig, ax = plt.subplots(figsize=(8,4))
            ax.plot(y_test.values, label='Actual', color='blue')
            ax.plot(y_pred, label='Predicted', color='red')
            ax.set_title(f"{ticker} Actual vs Predicted")
            ax.set_xlabel("Days")
            ax.set_ylabel("Price")
            ax.legend()
            st.pyplot(fig)
            latest = data[['Open','High','Low','Close','Volume']].iloc[-1].values.reshape(1,-1)
            next_pred = model.predict(latest)[0]
            st.success(f"Predicted next day's close for {ticker}: ${next_pred:.2f}")
            st.write("Latest data sample:")
            st.dataframe(data.tail(5))
