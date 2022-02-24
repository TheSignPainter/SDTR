import os, sys
import time
sys.path.insert(0, '..')
import numpy as np
import matplotlib.pyplot as plt
from lib import Dataset
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.metrics import mean_squared_error

#dataset: [YEAR, MICROSOFT, YAHOO, SARCOS, WINE]
name = "YAHOO"
data = Dataset(name, random_state=1234, quantile_transform=True, quantile_noise=1e-3)
in_features = data.X_train.shape[1]
data.y_train, data.y_test, data.y_valid = map(lambda x: np.expand_dims(x, -1), (data.y_train, data.y_test, data.y_valid))
best_mse=999999
best_group=None

with open("result_tabnet_%s_lhr.log" % name, 'w', encoding='utf-8') as f:
    f.write("dataset = %s\n" % name)
    for da in [6, 8, 10]:
        for n_steps in [3, 4, 5, 6]:
            for gamma in [1.1, 1.2, 1.3, 1.4]:
                for l_sparse in [0.1, 0.01, 0.001]:
                    f.write("da: %d, n_steps: %d, gamma: %f, l_sparse: %f\n"%(da, n_steps, gamma, l_sparse))
                    f.flush()
                    model = TabNetRegressor(n_d=da, n_a=da, n_steps=n_steps, gamma=gamma, lambda_sparse=l_sparse)
                    model.fit(data.X_train, data.y_train, eval_set=[(data.X_valid, data.y_valid)],)
                    y_pred = model.predict(data.X_test)
                    mse = mean_squared_error(data.y_test, y_pred)
                    f.write("tabnet test MSE: %.4f \n"%mse)
                    f.flush()
                    if mse<best_mse:
                        best_group=(da, n_steps, gamma, l_sparse)
                        best_mse = mse
    f.write("============\n best mse: %f\n" % best_mse)
    f.write("da: %d, n_steps: %d, gamma: %f, l_sparse: %f\n"%best_group)
    f.flush()

