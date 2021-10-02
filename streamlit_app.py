import streamlit as st
import datetime as dt
from datetime import date
import yfinance as yf
import pandas as pd
from plotly import graph_objs as go
import  plotly.express as px
import math
import numpy as np


START = "2014-01-01"
TODAY = dt.datetime.now().strftime("%Y-%m-%d")

st.title("Stock Prediction App")

stocks = ["Select the Stock", "AAPL", "AR", "MSFT", "USO", "TSLA", "SPY", "GME"]


# Loading RAW data

def load_data(ticker):
    data = yf.download(ticker, START,  TODAY)
    data.reset_index(inplace=True)
    return data


#Additional Information on Stocks

def stock_financials(stock):
    df_ticker = yf.Ticker(stock)
    sector = df_ticker.info['sector']
    prevClose = df_ticker.info['previousClose']
    twoHunDayAvg = df_ticker.info['twoHundredDayAverage']
    Name = df_ticker.info['longName']
    averageVolume = df_ticker.info['averageVolume']
    website = df_ticker.info['website']

    st.write('Company Name -', Name)
    st.write('Sector -', sector)
    st.write('Company Website -', website)
    st.write('Average Volume -', averageVolume)
    st.write('Previous Close -', prevClose)
    st.write('200 Day Average -', twoHunDayAvg)


#Plotting/fixing raw data

def plot_raw_data(stock, data_1):
    df_ticker = yf.Ticker(stock)
    Name = df_ticker.info['longName']
    data_1.reset_index()
    numeric_df = data_1.select_dtypes(['float', 'int'])
    numeric_cols = numeric_df.columns.tolist()
    st.markdown('')
    st.markdown('**_Features_** you want to **_Plot_**')
    features_selected = st.multiselect("", numeric_cols)
    if st.button("Generate Plot"):
        cust_data = data_1[features_selected]
        plotly_figure = px.line(data_frame=cust_data, x=data_1['Date'], y=features_selected,
                                title= Name + ' ' + '<i>timeline</i>')
        plotly_figure.update_layout(title = {'y':0.9,'x':0.5, 'xanchor': 'center', 'yanchor': 'top'})
        plotly_figure.update_xaxes(title_text='Date')
        plotly_figure.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, title="Price"), width=800, height=550)
        st.plotly_chart(plotly_figure)

#Testing out data

def create_train_test_data(df1):

    data = df1.sort_index(ascending=True, axis=0)
    new_data = pd.DataFrame(index=range(0, len(df1)), columns=['Date', 'High', 'Low', 'Open', 'Volume', 'Close'])

    for i in range(0, len(data)):
        new_data['Date'][i] = data['Date'][i]
        new_data['High'][i] = data['High'][i]
        new_data['Low'][i] = data['Low'][i]
        new_data['Open'][i] = data['Open'][i]
        new_data['Volume'][i] = data['Volume'][i]
        new_data['Close'][i] = data['Close'][i]

    #fixing time
    new_data['Date'] = pd.to_datetime(new_data['Date']).dt.date

    train_data_len = math.ceil(len(new_data) * .8)

    train_data = new_data[:train_data_len]
    test_data = new_data[train_data_len:]

    return train_data, test_data


#Linear Regression

def Linear_Regression_model(train_data, test_data):

    x_train = train_data.drop(columns=['Date', 'Close'], axis=1)
    x_test = test_data.drop(columns=['Date', 'Close'], axis=1)
    y_train = train_data['Close']
    y_test = test_data['Close']

    from sklearn.linear_model import LinearRegression

    model = LinearRegression()
    model.fit(x_train, y_train)

    prediction = model.predict(x_test)

    return prediction


#Plot Predictions
def prediction_plot(pred_data, test_data, models, ticker_name):

    test_data['Predicted'] = 0
    test_data['Predicted'] = pred_data

    #Resetting the index
    test_data.reset_index(inplace=True, drop=True)
    st.success("Your Data Prediction is Successful!")
    st.markdown('')
    st.write("Predicted Price vs Actual Close - " ,models)
    st.write("Stock Prediction Data for - ", ticker_name)
    st.write(test_data[['Date', 'Close', 'Predicted']])
    st.write("Plot Close Price vs Predicted Price for - ", models)

    #Plotting the Graph
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_data['Date'], y=test_data['Close'], mode='lines', name='Close'))
    fig.add_trace(go.Scatter(x=test_data['Date'], y=test_data['Predicted'], mode='lines', name='Predicted'))
    fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01), height=550, width=800,
                      autosize=False, margin=dict(l=25, r=75, b=100, t=0))

    st.plotly_chart(fig)

# Sidebar Menu -----------------------

menu=["Stock Info, Dataframes, & Testing", "Stock Predictions"]
st.sidebar.title("Menu")
choices = st.sidebar.selectbox("Choices:", menu,index=0)


if choices == 'Stock Info, Dataframes, & Testing':
    st.subheader("Extracting Data")
    st.markdown('Enter Stock of Your Choosing')
    user_input = st.text_input("", '')

    if not user_input:
        pass
    else:
        data = load_data(user_input)
        st.markdown('Select from the options below..')

        selected_explore = st.selectbox("", options=['Select your Option', 'Stock Information and Testing',
                                                     'DataFrames'], index=0)
        if selected_explore == 'Stock Information and Testing':
            st.markdown('')
            st.markdown('Extracting Stock Information')
            st.markdown('')
            st.markdown('')
            stock_financials(user_input)
            plot_raw_data(user_input, data)
            st.markdown('')

        elif selected_explore == 'DataFrames':
            st.markdown('Select **_Start_ _Date_ _for_ _Historical_ Stock** Data & features')
            start_date = st.date_input("", date(2014, 1, 1))
            st.write('You Selected Data From - ', start_date)
            submit_button = st.button("Extract Features")

            start_row = 0
            if submit_button:
                st.write('Extracted Features Dataframe for ', user_input)
                for i in range(0, len(data)):
                    if start_date <= pd.to_datetime(data['Date'][i]):
                        start_row = i
                        break
                st.write(data.iloc[start_row:, :])

elif choices == 'Stock Predictions':
    st.subheader("Train Machine Learning Models for Stock Prediction")
    st.markdown('')
    st.markdown('**_Select_ _Stocks_ _to_ Train**')
    stock_select = st.selectbox("", stocks, index=0)
    df1 = load_data(stock_select)
    df1 = df1.reset_index()
    df1['Date'] = pd.to_datetime(df1['Date']).dt.date
    options = ['Select your option', 'Linear Regression']
    st.markdown('')
    st.markdown('**_Select_ _Machine_ _Learning_ _Algorithms_ to Train**')
    models = st.selectbox("", options)
    submit = st.button('Train Model')


    if models == 'Linear Regression':
        if submit:
            st.write('**Your _final_ _dataframe_ _for_ Training**')
            st.write(df1[['Date','Close']])
            train_data, test_data = create_train_test_data(df1)
            pred_data = Linear_Regression_model(train_data, test_data)
            prediction_plot(pred_data, test_data, models, stock_select)




