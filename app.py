
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from login_config import check_login

# Login
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    check_login()
    st.stop()

# Main app
st.set_page_config(page_title="7-Day AI Forecast", layout="centered")
st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/0f/McDonald%27s_logo.svg/512px-McDonald%27s_logo.svg.png", width=120)
st.title("Welcome admin, McDonald's San Carlos Forecast Ready")

uploaded_file = st.file_uploader("Upload your historical data (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[['Date', 'Sales', 'Customers']]
    df.sort_values('Date', inplace=True)
    st.subheader("Preview")
    st.dataframe(df.tail())

    data = df[['Sales', 'Customers']].values
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    def create_dataset(data, time_step=3):
        X, y = [], []
        for i in range(len(data)-time_step-1):
            X.append(data[i:(i+time_step)])
            y.append(data[i+time_step])
        return np.array(X), np.array(y)

    X, y = create_dataset(data_scaled, 5)
    X = X.reshape(X.shape[0], X.shape[1], 2)

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 2)))
    model.add(LSTM(50))
    model.add(Dense(2))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=100, verbose=0)

    input_seq = data_scaled[-5:]
    predictions = []
    for _ in range(7):
        input_reshaped = input_seq.reshape(1, 5, 2)
        pred = model.predict(input_reshaped, verbose=0)
        predictions.append(pred[0])
        input_seq = np.vstack([input_seq[1:], pred])

    forecast = scaler.inverse_transform(predictions)
    forecast_dates = [df['Date'].max() + timedelta(days=i+1) for i in range(7)]
    result_df = pd.DataFrame({
        'Date': forecast_dates,
        'Forecasted Sales': forecast[:,0].round().astype(int),
        'Forecasted Customers': forecast[:,1].round().astype(int)
    })

    st.subheader("7-Day Forecast")
    st.dataframe(result_df)

    csv = result_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Forecast", data=csv, file_name="forecast_7_days.csv", mime='text/csv')
