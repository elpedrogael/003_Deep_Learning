import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna
import ta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

# Evaluate strategies with visualization
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

    # Run simulations for each strategy
    strategies = [
        ('RSI', RSI_PARAMS),
        ('BB', BB_PARAMS),
        ('MACD', MACD_PARAMS)
    ]

    for strategy_name, params in strategies:
        dataset_result, portfolio_value, final_metrics = simulate_trading(
            dataset.copy(),
            strategy=strategy_name,
            stop_loss=params['stop_loss'],
            take_profit=params['take_profit'],
            n_shares=params['n_shares'],
            lookback=params['lookback'] if strategy_name == 'MACD' else None
        )

        # Mostrar resultados
        print(f"\n--- {strategy_name} Strategy ---")
        print("Mejor valor del portafolio:", portfolio_value[-1])
        print("Mejores parámetros:", params)
        print("Métricas finales:")
        print(f"Sharpe Ratio: {final_metrics['Sharpe']:.2f}")
        print(f"Sortino Ratio: {final_metrics['Sortino']:.2f}")
        print(f"Calmar Ratio: {final_metrics['Calmar']:.2f}")
        print(f"Win/Loss Percentage: {final_metrics['Win_Loss_Percentage']:.2f}%")  # Fixed from 'Win_Loss_Percent'

        # Graficar
        plt.figure(figsize=(12, 6))
        plt.plot(portfolio_value, label="Portfolio Value")
        plt.legend()
        plt.twinx().plot(data.Close, c="orange", label="AAPL Close")
        plt.legend(loc="upper right")
        plt.title(f"{strategy_name} Strategy Performance")
        plt.show()

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