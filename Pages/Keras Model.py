import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from prophet.plot import plot_plotly
st.set_page_config(
        page_title="Keras Model",
        page_icon="chart_with_upwards_trend"
        
    )
model = load_model('Stock Predictions Model.keras')

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



start = '2012-01-01'
end=date.today()
value = ticker.split('(')
value=value[1].split(')')
ticker1=value[0]
data = yf.download(ticker1, start ,end)
data1=data.sort_values(by='Date',ascending=False)
st.write('You selected:', ticker ,'and its stock ticker is ',ticker1)
st.subheader('Stock Data')
st.write(data1)


data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

st.subheader('Price vs MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8,4))
plt.plot(ma_50_days, 'r', label="50 day average")
plt.plot(data.Close, 'g', label="Stock Trend")
plt.xlabel('Years')
plt.title("Closing trend and 50 day average", fontsize = 20,loc = 'center')
plt.legend(loc="lower right")
plt.ylabel('Stock Trend')
plt.show()
st.pyplot(fig1)

st.subheader('Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8,4))
plt.plot(ma_50_days, 'r', label="50 day average")
plt.plot(ma_100_days, 'b', label="100 day average")
plt.plot(data.Close, 'g' , label="Stock Trend")
plt.title("Closing trend, 50 & 100 day average", fontsize = 20,loc = 'center')
plt.xlabel('Years')
plt.ylabel('Stock Trend')   
plt.legend(loc="lower right")
plt.show()
st.pyplot(fig2)

st.subheader('Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8,4))
plt.plot(ma_100_days, 'r', label="100 day average")
plt.plot(ma_200_days, 'b', label="200 day average")
plt.plot(data.Close, 'g' ,label="Stock Trend")
plt.title("Closing trend and 50,100,200 day average", fontsize = 20,loc = 'center')
plt.xlabel('Years')
plt.ylabel('Stock Trend')
plt.legend(loc="lower right")
plt.show()
st.pyplot(fig3)

x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

x,y = np.array(x), np.array(y)

predict = model.predict(x)

scale = 1/scaler.scale_

predict = predict * scale
y = y * scale

st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(8,4))
plt.plot(predict, 'r', label='Original Price')
plt.plot(y, 'g', label = 'Predicted Price')
plt.title("Closing trend and Predicted Trend", fontsize = 20,loc = 'center')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend(loc="lower right")
plt.show()
st.pyplot(fig4)
