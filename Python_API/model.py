import math
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

plt.style.use('fivethirtyeight')

def predict_stock_price(stock_symbol):
    today = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=2*365)).strftime('%Y-%m-%d')
    
    # Get the stock quote
    df = yf.download(stock_symbol, start=start_date, end=today)
    
    # Prepare data
    data = df[['Close']]
    dataset = data.values
    training_data_len = math.ceil(len(dataset) * 0.8)
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)
    
    # Create training dataset
    training_data = scaled_data[:training_data_len, :]
    x_train, y_train = [], []
    for i in range(60, len(training_data)):
        x_train.append(training_data[i-60:i, 0])
        y_train.append(training_data[i, 0])
    
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    # Build LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    
    # Compile and train model
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=1)
    
    # Prepare testing data
    test_data = scaled_data[training_data_len - 60:, :]
    x_test, y_test = [], dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])
    
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    
    # Predict
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions

    save_dir = "plots"
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(16,8))
    plt.title('Model Prediction')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Valid', 'Predictions'], loc='lower right')
    #plt.show()

    save_path = os.path.join(save_dir, "model_prediction.png")
    plt.savefig(save_path)
    
    # Predict next closing price
    last_60_days = data[-60:].values
    last_60_days_scaled = scaler.transform(last_60_days)
    X_test = np.array([last_60_days_scaled])
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    pred_price = model.predict(X_test)
    pred_price = scaler.inverse_transform(pred_price)
    
    print(f'Predicted Closing Price for {stock_symbol} on {today}: {pred_price[0][0]:.2f} USD')
    return pred_price[0][0]

# Example usage:
predict_stock_price("GOOG")
