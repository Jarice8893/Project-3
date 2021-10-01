import streamlit as st
from datetime import date
import yfinance as yf

from plotly import graph_objs as go

START = "1980-12-12"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock prediction")

stocks = ("AAPL", "MSFT")
selected_stocks = st.selectbox("Select Dataset for prediction", stocks)

n_years = st.slider("Years of Prediction", 1, 4)
prediction = n_years * 365
