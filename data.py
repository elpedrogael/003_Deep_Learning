import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna
import ta

data = pd.read_csv("aapl_5m_train.csv").dropna()
data.head()

