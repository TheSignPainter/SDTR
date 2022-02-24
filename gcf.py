#!/usr/bin/env python
# coding: utf-8


import os, sys
import time
sys.path.insert(0, '..')
import numpy as np
import matplotlib.pyplot as plt
from lib import Dataset
from deepforest import CascadeForestRegressor
from sklearn.metrics import mean_squared_error

#dataset: [YEAR, MICROSOFT, YAHOO, SARCOS, WINE]
data = Dataset("YEAR", random_state=1234, quantile_transform=True, quantile_noise=1e-3)
in_features = data.X_train.shape[1]
# print(in_features)

# mu, std = data.y_train.mean(), data.y_train.std()
# normalize = lambda x: ((x - mu) / std).astype(np.float32)
# data.y_train, data.y_valid, data.y_test = map(normalize, [data.y_train, data.y_valid, data.y_test])

model = CascadeForestRegressor(random_state=1, n_bins=225, bin_subsample=200000, n_trees=100, delta=0.00001)
model.fit(data.X_train, data.y_train)
y_pred = model.predict(data.X_test)
mse = mean_squared_error(data.y_test, y_pred)

print("gcForest test MSE:", mse)
