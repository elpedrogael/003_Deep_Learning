import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna
import ta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

# Parámetros para la estrategia combinada
COMBINED_PARAMS = {
    'rsi_window': 14,  # Reducido para más sensibilidad
    'rsi_lower': 30,   # Más típico para compras
    'rsi_upper': 70,   # Más típico para ventas
    'bb_window': 20,   # Estándar para BB
    'bb_window_dev': 2.0,  # Más común
    'macd_window_slow': 26,
    'macd_window_fast': 12,
    'macd_window_sign': 9,
    'stop_loss': 0.02,  # Aumentado para más realismo
    'take_profit': 0.03,  # Aumentado para más realismo
    'n_shares': 100,   # Reducido para simplificar
    'lookback': 20
}

# Supporting functions
def prepare_lstm_data(dataset, lookback=20, features=None):
    if features is None:
        features = ["Close", "RSI", "MACD", "MACD_signal"]
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

# Función calculate_metrics
def calculate_metrics(portfolio_value, trades):
    returns = np.diff(portfolio_value) / portfolio_value[:-1]
    annual_factor = np.sqrt(252 * 78)
    sharpe = np.mean(returns) / np.std(returns) * annual_factor if np.std(returns) > 0 else -float('inf')
    downside_returns = returns[returns < 0]
    sortino = np.mean(returns) / np.std(downside_returns) * annual_factor if len(downside_returns) > 0 else -float('inf')
    drawdowns = np.maximum.accumulate(portfolio_value) - portfolio_value
    max_drawdown = np.max(drawdowns)
    calmar = (portfolio_value[-1] - portfolio_value[0]) / max_drawdown if max_drawdown > 0 else -float('inf')
    wins = sum(1 for t in trades if t["PnL"] > 0)
    losses = sum(1 for t in trades if t["PnL"] <= 0)
    win_loss = (wins / (wins + losses)) * 100 if (wins + losses) > 0 else 0
    return sharpe, sortino, calmar, win_loss

# Preprocesamiento
def preprocess(data: pd.DataFrame, params: dict) -> pd.DataFrame:
    dataset = data.copy()
    rsi = ta.momentum.RSIIndicator(dataset.Close, window=params['rsi_window'])
    dataset['RSI'] = rsi.rsi()
    bb = ta.volatility.BollingerBands(dataset.Close, window=params['bb_window'], window_dev=params['bb_window_dev'])
    dataset['BB_LOWER'] = bb.bollinger_lband()
    dataset['BB_UPPER'] = bb.bollinger_hband()
    macd = ta.trend.MACD(dataset.Close, window_slow=params['macd_window_slow'],
                         window_fast=params['macd_window_fast'], window_sign=params['macd_window_sign'])
    dataset['MACD'] = macd.macd()
    dataset['MACD_signal'] = macd.macd_signal()
    dataset = dataset.dropna()
    return dataset

# Simulación de trading
def simulate_trading(dataset: pd.DataFrame, params: dict) -> tuple:
    # LSTM para MACD
    X_train, y_train, _, _, scaler = prepare_lstm_data(dataset.dropna(), lookback=params['lookback'],
                                                       features=["Close", "MACD", "MACD_signal"])
    model = train_lstm(X_train, y_train, lookback=params['lookback'])
    scaled_data = scaler.transform(dataset[["Close", "MACD", "MACD_signal"]])
    X_full = [scaled_data[i - params['lookback']:i] for i in range(params['lookback'], len(scaled_data))]
    X_full = np.array(X_full)
    lstm_preds = (model.predict(X_full, verbose=0) > 0.5).astype(int).flatten()
    dataset = dataset.iloc[params['lookback']:].reset_index(drop=True)
    dataset["LSTM_BUY"] = pd.Series(lstm_preds) == 1
    dataset["LSTM_SELL"] = pd.Series(lstm_preds) == 0

    # Señales individuales
    dataset["RSI_BUY"] = dataset["RSI"] < params['rsi_lower']
    dataset["RSI_SELL"] = dataset["RSI"] > params['rsi_upper']
    dataset["BB_BUY"] = dataset.Close < dataset["BB_LOWER"]
    dataset["BB_SELL"] = dataset.Close > dataset["BB_UPPER"]
    dataset["MACD_BUY"] = dataset["MACD"] > dataset["MACD_signal"]
    dataset["MACD_SELL"] = dataset["MACD"] < dataset["MACD_signal"]

    # Estrategia combinada: 2 o más indicadores
    dataset["BUY_SIGNAL"] = (dataset["RSI_BUY"].astype(int) + dataset["BB_BUY"].astype(int) + dataset["MACD_BUY"].astype(int)) >= 2
    dataset["SELL_SIGNAL"] = (dataset["RSI_SELL"].astype(int) + dataset["BB_SELL"].astype(int) + dataset["MACD_SELL"].astype(int)) >= 2

    capital = 1000000
    com = 0.125 / 100
    portfolio_value = [capital]
    trades = []
    active_long_pos = None
    active_short_pos = None

    for i, row in dataset.iterrows():
        if active_long_pos:
            if row.Close < active_long_pos["stop_loss"]:
                pnl = row.Close * params["n_shares"] * (1 - com) - active_long_pos["cost"]
                capital += active_long_pos["cost"] + pnl
                trades.append({"PnL": pnl})
                active_long_pos = None
            elif row.Close > active_long_pos["take_profit"]:
                pnl = row.Close * params["n_shares"] * (1 - com) - active_long_pos["cost"]
                capital += active_long_pos["cost"] + pnl
                trades.append({"PnL": pnl})
                active_long_pos = None

        if active_short_pos:
            if row.Close > active_short_pos["stop_loss"]:
                pnl = active_short_pos["revenue"] - row.Close * params["n_shares"] * (1 + com)
                capital += pnl
                trades.append({"PnL": pnl})
                active_short_pos = None
            elif row.Close < active_short_pos["take_profit"]:
                pnl = active_short_pos["revenue"] - row.Close * params["n_shares"] * (1 + com)
                capital += pnl
                trades.append({"PnL": pnl})
                active_short_pos = None

        if row["BUY_SIGNAL"] and active_long_pos is None and active_short_pos is None:
            cost = row.Close * params["n_shares"] * (1 + com)
            if capital > cost:
                capital -= cost
                active_long_pos = {
                    "datetime": row.Datetime,
                    "cost": cost,
                    "take_profit": row.Close * (1 + params["take_profit"]),
                    "stop_loss": row.Close * (1 - params["stop_loss"])
                }

        if row["SELL_SIGNAL"] and active_short_pos is None and active_long_pos is None:
            revenue = row.Close * params["n_shares"] * (1 - com)
            capital += revenue
            active_short_pos = {
                "datetime": row.Datetime,
                "revenue": revenue,
                "take_profit": row.Close * (1 - params["take_profit"]),
                "stop_loss": row.Close * (1 + params["stop_loss"])
            }

        long_value = row.Close * params["n_shares"] if active_long_pos else 0
        short_value = (active_short_pos["revenue"] - row.Close * params["n_shares"]) if active_short_pos else 0
        portfolio_value.append(capital + long_value + short_value)

    sharpe, sortino, calmar, win_loss = calculate_metrics(portfolio_value, trades)
    final_metrics = {
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": calmar,
        "Win_Loss_Percentage": win_loss
    }
    return dataset, portfolio_value, final_metrics

# Evaluar la estrategia combinada
def evaluate_combined_strategy(data: pd.DataFrame):
    dataset = preprocess(data, COMBINED_PARAMS)
    dataset_result, portfolio_value, final_metrics = simulate_trading(dataset.copy(), COMBINED_PARAMS)

    print("\n--- Combined Strategy (RSI + BB + MACD) ---")
    print("Mejor valor del portafolio:", portfolio_value[-1])
    print("Mejores parámetros:", COMBINED_PARAMS)
    print("Métricas finales:")
    print(f"Sharpe Ratio: {final_metrics['Sharpe']:.2f}")
    print(f"Sortino Ratio: {final_metrics['Sortino']:.2f}")
    print(f"Calmar Ratio: {final_metrics['Calmar']:.2f}")
    print(f"Win/Loss Percentage: {final_metrics['Win_Loss_Percentage']:.2f}%")

    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_value, label="Portfolio Value")
    plt.legend()
    plt.twinx().plot(data.Close, c="orange", label="AAPL Close")
    plt.legend(loc="upper right")
    plt.title("Combined Strategy Performance")
    plt.show()

# Datos de prueba con más volatilidad
dates = pd.date_range(start="2023-01-01", end="2023-01-02", freq="5min")
data = pd.DataFrame({
    "Datetime": dates,
    "Close": 100 + np.cumsum(np.random.normal(0, 0.5, len(dates)))  # Movimiento más realista
})

assert "Close" in data.columns and "Datetime" in data.columns, "DataFrame must have 'Close' and 'Datetime' columns"
evaluate_combined_strategy(data)