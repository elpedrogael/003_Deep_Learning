import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna
import ta

data = pd.read_csv("aapl_5m_train.csv").dropna()
data.head()

rsi = ta.momentum.RSIIndicator(data.Close, window=20)
bb = ta.volatility.BollingerBands(data.Close, window =15, window_dev = 2)

dataset = data.copy()
dataset['RSI'] = rsi.rsi()
dataset['BB'] = bb.bollinger_mavg()

dataset['RSI_BUY'] = dataset['RSI'] < 25
dataset['RSI_SELL'] = dataset['RSI'] > 75

dataset['BB_BUY'] = bb.bollinger_lband_indicator().astype(bool)
dataset['BB_SELL'] = bb.bollinger_hband_indicator().astype(bool)

dataset = dataset.dropna()
dataset.head()

capital = 1_000_000
com = 0.125 / 100

portfolio_value = [capital]

stop_loss = 0.15
take_profit = 0.10
n_shares = 1000

wins = 0
losses = 0

active_long_positions = None
active_short_positions = None

for i, row in dataset.iterrows():
    # Close Long Positions
    if active_long_positions:
        # Closed By Stop Loss
        if row.Close < active_long_positions['stop_loss']:
            pnl = row.Close * n_shares * (1 - com)
            capital += pnl
            active_long_positions = None

        # Closed By Take Profit
        if row.Close > active_long_positions[take_profit]:
            pnl = row.Close * n_shares * (1 - com)
            capital += pnl
            active_long_positions = None

        # Close Short Positiones

        # Open Long Positiones
        if row.RSI_BUY and active_long_positions is None:
            cost = row.Close * n_shares * (1 + com)
            if capital > cost:
                capital -= cost
                active_long_positions = {
                    'datetime': row.Datetime,
                    'opened_at': row.Close,
                    'take_profit': row.Close * (1 + take_profit),
                    'stop_loss': row.Close * (1 - stop_loss)
                }

    # Open Short Positions

    # Calculate Long Positions Value
    long_value = 0
    if active_long_positions:
        long_value = row.CLose * n_shares

    # Calculate Short Positions Value

    # Add Portfolio Value
    portfolio_value.append(capital + long_value)

plt.figure(figsize=(12, 6))
plt.plot(portfolio_value, label= 'Portfolio Value')
plt.legend()
plt.show()


# RSI
def objective_func(trial, data):
    rsi_window = trial.suggest_int('rsi_window', 10, 100)
    rsi_lower = trial.suggest_int('rsi_lower', 5, 35)
    rsi_upper = trial.suggest_int('rsi_upper', 65, 95)

    stop_loss = trial.suggest_float('stop_loss', 0.01, 0.2)
    take_profit = trial.suggest_float('take_profit', 0.01, 0.2)
    n_shares = trial.suggest_categorical('n_shares', [100, 500, 800, 1000, 1200])

    rsi = ta.momentum.RSIIndicator(data.Close, window=20)
    bb = ta.volatility.BollingerBands(data.Close, window=15, window_dev=2)
    dataset = data.copy()
    dataset['RSI'] = rsi.rsi()
    dataset['BB'] = bb.bollinger_mavg()

    dataset['RSI_BUY'] = dataset['RSI'] < rsi_lower
    dataset['RSI_SELL'] = dataset['RSI'] > 80

    dataset['BB_BUY'] = bb.bollinger_lband_indicator().astype(bool)
    dataset['BB_SELL'] = bb.bollinger_hband_indicator().astype(bool)

    dataset = dataset.dropna()

    capital = 1_000_000
    com = 0.125 / 100

    portfolio_value = [capital]

    stop_loss = 0.15
    take_profit = 0.10
    n_shares = 1000

    wins = 0
    losses = 0

    active_long_positions = None
    active_short_positions = None

    for i, row in dataset.iterrows():
        # Close Long Positions
        if active_long_positions:
            # Closed By Stop Loss
            if row.Close < active_long_positions['stop_loss']:
                pnl = row.Close * n_shares * (1 - com)
                capital += pnl
                active_long_positions = None

            # Closed By Take Profit
            if row.Close > active_long_positions[take_profit]:
                pnl = row.Close * n_shares * (1 - com)
                capital += pnl
                active_long_positions = None

            # Close Short Positiones

            # Open Long Positiones
            if row.RSI_BUY and active_long_positions is None:
                cost = row.Close * n_shares * (1 + com)
                if capital > cost:
                    capital -= cost
                    active_long_positions = {
                        'datetime': row.Datetime,
                        'opened_at': row.Close,
                        'take_profit': row.Close * (1 + take_profit),
                        'stop_loss': row.Close * (1 - stop_loss)
                    }

        # Open Short Positions

        # Calculate Long Positions Value
        long_value = 0
        if active_long_positions:
            long_value = row.CLose * n_shares

        # Calculate Short Positions Value

        # Add Portfolio Value
        portfolio_value.append(capital + long_value)
    return portfolio_value[-1]