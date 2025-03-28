import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna
import ta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

# Running the optimization
study_rsi = optuna.create_study(direction="maximize")
study_rsi.optimize(lambda trial: objective_rsi(trial, data), n_trials=50)
print("Mejores parámetros RSI:", study_rsi.best_params)