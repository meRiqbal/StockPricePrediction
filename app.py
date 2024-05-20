# Import important libraries.
import tensorflow as tf
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

# Title of the web app
st.title("Stock Price Trend Prediction Web App")

# Input for stock ID
stock = st.text_input("Enter the Stock ID", "GOOG")

# Set the date range for downloading stock data
end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)

# Download stock data
try:
    data = yf.download(stock, start=start, end=end)
    if data.empty:
        st.error("No data found for the entered stock ID. Please try a different stock symbol.")
        st.stop()
except Exception as e:
    st.error(f"Error downloading data: {e}")
    st.stop()

# Display stock data
st.subheader("Stock Data")
st.write(data)

# Load the pre-trained model
try:
    model = tf.keras.models.load_model("Stock_price_trend_model.keras")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Split the data for testing
splitting_len = int(len(data) * 0.7)
x_test = pd.DataFrame(data.Close[splitting_len:])

# Define a function to plot graphs
def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset=None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values, 'Orange')
    plt.plot(full_data.Close, 'b')
    if extra_data:
        plt.plot(extra_dataset)
    return fig

# Plot different moving averages
st.subheader('Original Close Price and MA for 250 days')
data['MA_for_250_days'] = data.Close.rolling(250).mean()
st.pyplot(plot_graph((15, 6), data['MA_for_250_days'], data))

st.subheader('Original Close Price and MA for 200 days')
data['MA_for_200_days'] = data.Close.rolling(200).mean()
st.pyplot(plot_graph((15, 6), data['MA_for_200_days'], data))

st.subheader('Original Close Price and MA for 100 days')
data['MA_for_100_days'] = data.Close.rolling(100).mean()
st.pyplot(plot_graph((15, 6), data['MA_for_100_days'], data))

st.subheader('Original Close Price and MA for 100 days and MA for 250 days')
st.pyplot(plot_graph((15, 6), data['MA_for_100_days'], data, 1, data['MA_for_250_days']))

# Check if x_test is not empty
if not x_test.empty and 'Close' in x_test.columns:
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(x_test[['Close']])

    x_data = []
    y_data = []

    for i in range(100, len(scaled_data)):
        x_data.append(scaled_data[i-100:i])
        y_data.append(scaled_data[i])

    x_data, y_data = np.array(x_data), np.array(y_data)

    # Predict using the model
    predictions = model.predict(x_data)

    # Inverse transform the predictions and actual values
    inv_pre = scaler.inverse_transform(predictions)
    inv_y_test = scaler.inverse_transform(y_data)

    # Prepare data for plotting
    plotting_data = pd.DataFrame(
        {
            'original_test_data': inv_y_test.reshape(-1),
            'predictions': inv_pre.reshape(-1)
        },
        index=data.index[splitting_len + 100:]
    )

    # Display original vs predicted values
    st.subheader("Original values vs Predicted values")
    st.write(plotting_data)

    # Plot original close price vs predicted close price
    st.subheader('Original Close Price vs Predicted Close price')
    fig = plt.figure(figsize=(15, 6))
    plt.plot(pd.concat([data.Close[:splitting_len + 100], plotting_data], axis=0))
    plt.legend(["Data- not used", "Original Test data", "Predicted Test data"])
    st.pyplot(fig)
else:
    st.error("No data available for scaling or the 'Close' column is missing.")
