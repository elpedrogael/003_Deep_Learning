import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna
import ta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

# Optimization for RSI

# Objective function for RSI
def objective_rsi(trial, data, verbose=False):
    rsi_window = trial.suggest_int("rsi_window", 10, 100)
    rsi_lower = trial.suggest_int("rsi_lower", 5, 35)
    rsi_upper = trial.suggest_int("rsi_upper", 65, 95)
    stop_loss = trial.suggest_float("stop_loss", 0.01, 0.2)
    take_profit = trial.suggest_float("take_profit", 0.01, 0.2)
    n_shares = trial.suggest_categorical("n_shares", [1000, 2000, 3000, 3500, 4000])
    lookback = trial.suggest_int("lookback", 10, 50)

    dataset = data.copy()
    rsi = ta.momentum.RSIIndicator(dataset.Close, window=rsi_window)
    dataset["RSI"] = rsi.rsi()

    X_train, y_train, X_test, y_test, scaler = prepare_lstm_data(dataset.dropna(), lookback=lookback, features=["Close", "RSI"])
    model = train_lstm(X_train, y_train, lookback=lookback)

    scaled_data = scaler.transform(dataset[["Close", "RSI"]])
    X_full = [scaled_data[i - lookback:i] for i in range(lookback, len(scaled_data))]
    X_full = np.array(X_full)
    lstm_preds = (model.predict(X_full, verbose=0) > 0.5).astype(int).flatten()

    dataset = dataset.iloc[lookback:].reset_index(drop=True)
    dataset["LSTM_BUY"] = pd.Series(lstm_preds) == 1
    dataset["LSTM_SELL"] = pd.Series(lstm_preds) == 0

    dataset["RSI_BUY"] = dataset["RSI"] < rsi_lower
    dataset["RSI_SELL"] = dataset["RSI"] > rsi_upper

    dataset["BUY_SIGNAL"] = dataset["RSI_BUY"]
    dataset["SELL_SIGNAL"] = dataset["RSI_SELL"]

    dataset = dataset.dropna()
    capital = 1000000
    com = 0.5 / 100
    portfolio_value = [capital]
    active_long_pos = None
    active_short_pos = None
    wins = 0
    losses = 0

    for i, row in dataset.iterrows():
        if active_long_pos:
            if row.Close < active_long_pos["stop_loss"]:
                pnl = row.Close * n_shares * (1 - com) - active_long_pos["cost"]
                capital += active_long_pos["cost"] + pnl
                if pnl > 0:
                    wins += 1
                else:
                    losses += 1
                active_long_pos = None
            elif row.Close > active_long_pos["take_profit"]:
                pnl = row.Close * n_shares * (1 - com) - active_long_pos["cost"]
                capital += active_long_pos["cost"] + pnl
                if pnl > 0:
                    wins += 1
                else:
                    losses += 1
                active_long_pos = None

        if active_short_pos:
            if row.Close > active_short_pos["stop_loss"]:
                pnl = active_short_pos["revenue"] - row.Close * n_shares * (1 + com)
                capital += pnl
                if pnl > 0:
                    wins += 1
                else:
                    losses += 1
                active_short_pos = None
            elif row.Close < active_short_pos["take_profit"]:
                pnl = active_short_pos["revenue"] - row.Close * n_shares * (1 + com)
                capital += pnl
                if pnl > 0:
                    wins += 1
                else:
                    losses += 1
                active_short_pos = None

        if row["BUY_SIGNAL"] and active_long_pos is None and active_short_pos is None:
            cost = row.Close * n_shares * (1 + com)
            if capital > cost:
                capital -= cost
                active_long_pos = {
                    "datetime": row.Datetime,
                    "cost": cost,
                    "take_profit": row.Close * (1 + take_profit),
                    "stop_loss": row.Close * (1 - stop_loss)
                }

        if row["SELL_SIGNAL"] and active_short_pos is None and active_long_pos is None:
            revenue = row.Close * n_shares * (1 - com)
            capital += revenue
            active_short_pos = {
                "datetime": row.Datetime,
                "revenue": revenue,
                "take_profit": row.Close * (1 - take_profit),
                "stop_loss": row.Close * (1 + stop_loss)
            }

        long_value = row.Close * n_shares if active_long_pos else 0
        short_value = (active_short_pos["revenue"] - row.Close * n_shares) if active_short_pos else 0
        portfolio_value.append(capital + long_value + short_value)

    if len(portfolio_value) <= 1:
        return -float('inf')
    metrics = calculate_metrics(portfolio_value, wins, losses)
    return metrics["Sharpe"]
