#importing the required libraries
import streamlit as st

st.set_page_config(
        page_title="Stock Price Prediction",
        page_icon="chart_with_upwards_trend",
        
    )
st.title('Robert Antony Final Year Project' )
st.header('Stock Price Prediction (using prophet and keras)', divider='green')
st.subheader('Prophet Model:')
st.text("The Prophet model is an open-source tool developed by Facebook for univariate  ") 
st.text(" time series forecasting ,meaning it predicts the future values of a single variable  ")
st.text("based on historical data. It's particularly well-suited for data with trends, ")
st.text("seasonality, and holidays. ")
code='''
from prophet import Prophet
m=Prophet()
m.fit(df_train)
future=m.make_future_dataframe(periods=period)
forecast=m.predict(future)
'''
st.code(code, language='python')
st.text("This model can predict future values as it is a predict function to predict future")
st.text("values.")
st.subheader('Keras Model:')
st.text("The keras model is custom made for stock price prediction. It is built on ") 
st.text(" sequentail model with custom layers  and optimised the stock prediction. ")
st.subheader("model layers")
layers='''
model = Sequential()
model.add(LSTM(units=50, activation='relu',return_sequences=True,input_shape=((x.shape[1],1))))
model.add(Dropout(0.2))
model.add(LSTM(units=60, activation='relu',return_sequences=True ))
model.add(Dropout(0.3))
model.add(LSTM(units=80, activation='relu',return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(units=120, activation='relu', return_sequences=True))
model.add(Dropout(0.5))
model.add(Dense(units=1))'''
st.code(layers, language='python')
st.subheader("Conclusion")
st.text("By this project we can understand that machine learning can be implemented in ") 
st.text("platform and industry where there is more  number of data that be fed to the ")
st.text("machine learning model and can be used to predict the future values")