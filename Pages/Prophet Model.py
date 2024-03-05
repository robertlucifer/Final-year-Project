#importing the required libraries
import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd
st.set_page_config(
        page_title="Stock Price Prediction",
        page_icon="chart_with_upwards_trend"
    )
start = "2015-01-01"
today=date.today()

st.header('Stock prediction Using Keras Model')


ticker=st.selectbox( 'search a company',   ('Apple Inc. (AAPL)','Reliance (RELI)',
'Tata Consultancy Services Limited (TCS.NS)',
'Tata Consultancy Services Limited (TCS.BO)',
'HDFC Bank Limited (HDFCBANK.NS)',
'ICICI Bank Limited (IBN)',
'ICICI Bank Limited (ICICIBANK.BO)',
'State Bank of India (SBIN.NS)',
'Infosys Limited (INFY.NS)',
'Life Insurance Corporation of India (LICI.NS)',
'Bharti Airtel Limited (BHARTIARTL.NS)',
'Larsen & Toubro Limited (LT.NS)',
'ITC Limited (ITC.NS)',
'HCL Technologies Limited (HCLTECH.NS)',
'Bajaj Finance Limited (BAJFINANCE.NS)',
'Adani Enterprises Limited (ADANIENT.NS)',
'Sun Pharmaceutical Industries Limited (SUNPHARMA.NS)',
'Maruti Suzuki India Limited (MARUTI.NS)',
'Oil and Natural Gas Corporation Limited (ONGC.NS)',
'NTPC Limited (NTPC.NS)',
'Kotak Mahindra Bank Limited (KOTAKBANK.NS)',
'Axis Bank Limited (AXISBANK.NS)',
'Titan Company Limited (TITAN.NS)',
'Tata Motors Limited (TATAMOTORS.NS)',
'Adani Green Energy Limited (ADANIGREEN.NS)',
'UltraTech Cement Limited (ULTRACEMCO.NS)',
'Coal India Limited (COALINDIA.NS)',
'Wipro Limited (WIPRO.NS)',
'Asian Paints Limited (ASIANPAINT.NS)',

'Samsung Electronics Co., Ltd. (005930.KS)'))

value = ticker.split('(')
value=value[1].split(')')
value=value[0]

n_years = st.slider("Years of prediction:",0,4)
n_week =st.slider("weeks of prediction:",0,52)
n_day=st.slider("Days of prediction:",0,30)
period=(n_years*365)+(n_week*7)+(n_day)
def load_data(ticker):
    data=yf.download(ticker,start,today)
    data.reset_index(inplace=True)
    return data
data_load_state = st.text("Load data...")
data= load_data(value )
data_load_state.text("Loading data...done!")

st.subheader('Raw data')
st.write(data.tail())

#ploting the raw data   
def plot_raw_data():
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Open'],name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Close'],name='stock_close'))
    fig.layout.update(title_text="Time Series Data",xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
plot_raw_data()

#forecasting the data
df_train=data[['Date','Close']]
df_train=df_train.rename(columns={"Date":"ds","Close":"y"})

m=Prophet()
m.fit(df_train)
future=m.make_future_dataframe(periods=period)
forecast=m.predict(future)
st.subheader('Forecast data')
st.write(forecast.tail(100))

#ploting the forecast data
st.write("Forecast data")
fig1=plot_plotly(m,forecast)
st.plotly_chart(fig1)