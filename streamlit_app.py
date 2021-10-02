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

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_Close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visable=True)
    st.plotly_chart(fig)

plot_raw_data()










