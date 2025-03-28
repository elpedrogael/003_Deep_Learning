import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna
import ta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

# Best parameters from your input
RSI_PARAMS = {
    'rsi_window': 46,
    'rsi_lower': 13,
    'rsi_upper': 65,
    'stop_loss': 0.06425843746848879,
    'take_profit': 0.02250238758183155,
    'n_shares': 1000,
    'lookback': 20  # Default value since not provided
}

BB_PARAMS = {
    'bb_window': 26,
    'bb_window_dev': 1.6419616128479513,
    'stop_loss': 0.020183998738358935,
    'take_profit': 0.03515466268130313,
    'n_shares': 1000,
    'lookback': 20  # Default value since not provided
}

MACD_PARAMS = {
    'macd_window_slow': 25,
    'macd_window_fast': 20,
    'macd_window_sign': 9,
    'stop_loss': 0.011148111728088396,
    'take_profit': 0.012560961178120339,
    'n_shares': 3000,
    'lookback': 20
}


# Supporting functions
def prepare_lstm_data(dataset, lookback=20, features=None):
    if features is None:
        features = ["Close", "RSI", "BB", "MACD", "MACD_signal"]
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(dataset[features])
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i - lookback:i])
        y.append(1 if scaled_data[i, 0] > scaled_data[i - 1, 0] else 0)
    X, y = np.array(X), np.array(y)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    return X_train, y_train, X_test, y_test, scaler


def train_lstm(X_train, y_train, lookback, units=50):
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(units, return_sequences=False))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=3, batch_size=32, verbose=0)
    return model


def calculate_metrics(portfolio_value, wins, losses):
    returns = pd.Series(portfolio_value).pct_change().dropna()
    mean_ret = returns.mean()
    std_ret = returns.std()
    downside_std = returns[returns < 0].std() if len(returns[returns < 0]) > 0 else 0
    max_drawdown = (pd.Series(portfolio_value).cummax() - portfolio_value).max() / pd.Series(
        portfolio_value).cummax().max()

    sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else -float('inf')
    sortino = (mean_ret / downside_std) * np.sqrt(252) if downside_std > 0 else -float('inf')
    calmar = (mean_ret * 252) / max_drawdown if max_drawdown > 0 else -float('inf')
    win_loss_pct = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0

    return {
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": calmar,
        "Win_Loss_Percentage": win_loss_pct,
        "Wins": wins,
        "Losses": losses
    }


# Preprocessing function
def preprocess(data: pd.DataFrame, rsi_window: int, bb_window: int, bb_window_dev: float,
               macd_window_slow: int, macd_window_fast: int, macd_window_sign: int) -> pd.DataFrame:
    dataset = data.copy()

    # RSI
    rsi = ta.momentum.RSIIndicator(dataset.Close, window=rsi_window)
    dataset['RSI'] = rsi.rsi()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(dataset.Close, window=bb_window, window_dev=bb_window_dev)
    dataset['BB'] = bb.bollinger_mavg()
    dataset['BB_LOWER'] = bb.bollinger_lband()
    dataset['BB_UPPER'] = bb.bollinger_hband()

    # MACD
    macd = ta.trend.MACD(dataset.Close, window_slow=macd_window_slow, window_fast=macd_window_fast,
                         window_sign=macd_window_sign)
    dataset['MACD'] = macd.macd()
    dataset['MACD_signal'] = macd.macd_signal()

    dataset = dataset.dropna()
    return dataset


# Trading simulation function
def simulate_trading(dataset: pd.DataFrame, strategy: str, stop_loss: float, take_profit: float, n_shares: int,
                     lookback: int = None) -> dict:
    if strategy == 'MACD' and lookback is not None:
        X_train, y_train, _, _, scaler = prepare_lstm_data(dataset.dropna(), lookback=lookback,
                                                           features=["Close", "MACD", "MACD_signal"])
        model = train_lstm(X_train, y_train, lookback=lookback)
        scaled_data = scaler.transform(dataset[["Close", "MACD", "MACD_signal"]])
        X_full = [scaled_data[i - lookback:i] for i in range(lookback, len(scaled_data))]
        X_full = np.array(X_full)
        lstm_preds = (model.predict(X_full, verbose=0) > 0.5).astype(int).flatten()
        dataset = dataset.iloc[lookback:].reset_index(drop=True)
        dataset["LSTM_BUY"] = pd.Series(lstm_preds) == 1
        dataset["LSTM_SELL"] = pd.Series(lstm_preds) == 0

    # Define signals
    if strategy == 'RSI':
        dataset['BUY_SIGNAL'] = dataset['RSI'] < RSI_PARAMS['rsi_lower']
        dataset['SELL_SIGNAL'] = dataset['RSI'] > RSI_PARAMS['rsi_upper']
    elif strategy == 'BB':
        dataset['BUY_SIGNAL'] = dataset.Close < dataset['BB_LOWER']
        dataset['SELL_SIGNAL'] = dataset.Close > dataset['BB_UPPER']
    elif strategy == 'MACD':
        dataset['BUY_SIGNAL'] = dataset['MACD'] > dataset['MACD_signal']
        dataset['SELL_SIGNAL'] = dataset['MACD'] < dataset['MACD_signal']
    else:
        raise ValueError("Strategy must be 'RSI', 'BB', or 'MACD'")

    dataset = dataset.dropna()
    capital = 1000000
    com = 0.125 / 100
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

    return calculate_metrics(portfolio_value, wins, losses)


# Evaluate strategies function
def evaluate_strategies(data: pd.DataFrame):
    # Preprocess data once with all indicators
    dataset = preprocess(
        data,
        rsi_window=RSI_PARAMS['rsi_window'],
        bb_window=BB_PARAMS['bb_window'],
        bb_window_dev=BB_PARAMS['bb_window_dev'],
        macd_window_slow=MACD_PARAMS['macd_window_slow'],
        macd_window_fast=MACD_PARAMS['macd_window_fast'],
        macd_window_sign=MACD_PARAMS['macd_window_sign']
    )

    # Evaluate RSI strategy
    rsi_metrics = simulate_trading(
        dataset,
        strategy='RSI',
        stop_loss=RSI_PARAMS['stop_loss'],
        take_profit=RSI_PARAMS['take_profit'],
        n_shares=RSI_PARAMS['n_shares']
    )

    # Evaluate BB strategy
    bb_metrics = simulate_trading(
        dataset,
        strategy='BB',
        stop_loss=BB_PARAMS['stop_loss'],
        take_profit=BB_PARAMS['take_profit'],
        n_shares=BB_PARAMS['n_shares']
    )

    # Evaluate MACD strategy
    macd_metrics = simulate_trading(
        dataset,
        strategy='MACD',
        stop_loss=MACD_PARAMS['stop_loss'],
        take_profit=MACD_PARAMS['take_profit'],
        n_shares=MACD_PARAMS['n_shares'],
        lookback=MACD_PARAMS['lookback']
    )

    # Print results
    print("RSI Strategy Metrics:", rsi_metrics)
    print("BB Strategy Metrics:", bb_metrics)
    print("MACD Strategy Metrics:", macd_metrics)


# Prepare and run the backtest
# Example: Load your data (replace with your actual data source)
dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
data = pd.DataFrame({
    "Datetime": dates,
    "Close": np.random.normal(100, 10, len(dates))  # Random prices for demo
})

# Ensure data has the required columns
assert "Close" in data.columns and "Datetime" in data.columns, "DataFrame must have 'Close' and 'Datetime' columns"

# Run the evaluation
evaluate_strategies(data)