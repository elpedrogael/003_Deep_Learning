import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna
import ta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

# Run the optimization
study_macd = optuna.create_study(direction="maximize")
study_macd.optimize(lambda trial: objective_macd(trial, data), n_trials=50)
print("Mejores parámetros MACD:", study_macd.best_params)