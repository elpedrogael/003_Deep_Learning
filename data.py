import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna
import ta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

# Carga de datos
data = pd.read_csv('aapl_5m_train.csv').dropna()
print(f"Tamaño inicial de data: {len(data)}")
