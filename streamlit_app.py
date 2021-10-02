from os import lseek
import streamlit as st
from datetime import date
import yfinance as yf
from plotly import graph_objs as go

START = "1980-12-12"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock prediction")

stocks = ("AAPL", "MSFT", "AP", "SPY", "TSLA", "USO")
selected_stocks = st.selectbox("Select Dataset for prediction", stocks)

n_years = st.slider("Years of Prediction", 1, 4)
prediction = n_years * 365

@st.cache

def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Load data..")
data = load_data(selected_stocks)
data_load_state.text("Loading Data... done!")

st.subheader('Raw data')
st.write(data.tail())











