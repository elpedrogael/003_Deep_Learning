import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna
import ta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam


# Prepare_lstm_data is defined
def prepare_lstm_data(dataset, lookback=20, features=None):
    if features is None:
        features = ["Close", "RSI", "BB", "MACD", "MACD_signal"]
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(dataset[features])
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i - lookback:i])
        y.append(1 if scaled_data[i, 0] > scaled_data[i - 1, 0] else 0)  # Binary classification based on Close
    X, y = np.array(X), np.array(y)

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    return X_train, y_train, X_test, y_test, scaler


# Updated train_lstm with Input layer
def train_lstm(X_train, y_train, lookback, units=50):
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(units, return_sequences=False))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=3, batch_size=32, verbose=0)
    return model


# Updated calculate_metrics to handle NaN
def calculate_metrics(portfolio_value, wins, losses):
    returns = pd.Series(portfolio_value).pct_change().dropna()
    mean_ret = returns.mean()
    std_ret = returns.std()
    if std_ret == 0 or np.isnan(mean_ret) or np.isnan(std_ret):
        return {"Sharpe": -float('inf')}  # Penalize invalid cases
    sharpe = (mean_ret / std_ret) * np.sqrt(252)  # Annualized Sharpe Ratio
    return {"Sharpe": sharpe}


