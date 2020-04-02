
import sys
import os
from datetime import datetime

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import numpy as np
import pickle
from data_loader.generator import load_mv_data, cons_mv_data
from utils.tools import mean_mae
from sklearn.metrics import mean_absolute_error

from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.api import VAR


EXP_DIR = '../../exp_ElectricityLoad/'
exp_config, _exp_config = get_config_from_json(EXP_DIR + 'exp_config.json')
N_VAR = exp_config.N_VAR
VARS = exp_config.VARS

MODE_LIST = ['train', 'test', 'visual']
MODE = MODE_LIST[1]


def adjust(val, length= 6): return str(val).ljust(length)


# ADF test and summary report
def adfuller_test(series, name, signif=0.05):
    r = adfuller(series, autolag='AIC')
    output = {
        'test_statistic': round(r[0], 4),
        'pvalue': round(r[1], 4),
        'n_lags': round(r[2], 4),
        'n_obs': r[3]
    }
    p_value = output['pvalue']

    # Print Summary
    print('Augmented Dickey-Fuller Test on ' + name + '\n' + '-'*47 + '\n')
    print('Null Hypothesis: Data has unit root. Non-Stationary.')
    print('Significance Level:\t'+ str(signif))
    print('Test Statistic:\t' + str(output['test_statistic']))
    print('No. Lags Chosen:\t' + str(output['n_lags']))

    for key,val in r[4].items():
        print(' Critical value' + adjust(key) + str(round(val, 3)))

    if p_value <= signif:
        print(" => P-Value = %f. Rejecting Null Hypothesis." % (p_value))
        print(" => Series is Stationary.")
    else:
        print(" => P-Value = %f. Weak evidence to reject the Null Hypothesis." % (p_value))
        print(" => Series is Non-Stationary.")
    return


def run_train_var_model(exps_dir):
    n_var = N_VAR
    cols = VARS
    # 1 load train data
    train_data = load_mv_data(
        data_file=exps_dir + 'dataset/training.csv',
        cols=cols[:n_var]
    )

    # 2 stationarity check and difference
    # ADF test on each column
    # for col in cols:
    #     adfuller_test(train_data[col], col)

    # 3 select p-order and fit model 'model.select_order(maxlags)'
    model = VAR(train_data)
    # log = model.select_order(100)
    # print(log.summary())

    model_fitted = model.fit(maxlags=96)
    # file_path = exps_dir + 'saved_models/var/training_log.txt'
    # with open(file_path, 'w+') as log_file:
    #     print(model_fitted.summary(), file=log_file)
    #     # 4 check 'Serial Correlation of Residual' with 'durbin_watson'
    #     res_check_out = durbin_watson(model_fitted.resid)
    #     for col, val in zip(cols, res_check_out):
    #         print(adjust(col), ':', round(val, 2), file=log_file)

    # 5 model saved
    file_path = exps_dir + 'saved_models/var/trained_model_' + str(N_VAR) + '.pkl'
    with open(file_path, 'wb') as trained_model_file:
        pickle.dump(model_fitted, trained_model_file)
    return


def run_test_var_model(exps_dir):
    n_var = N_VAR
    cols = VARS
    # 1 load trained model
    file_path = exps_dir + 'saved_models/var/trained_model_' + str(N_VAR) + '.pkl'
    with open(file_path, 'rb') as trained_model_file:
        model_fitted = pickle.load(trained_model_file)
    lag_order = model_fitted.k_ar
    print(lag_order)
    # 2 load testing data
    x, y = cons_mv_data(
        data_file=exps_dir + 'dataset/testing.csv',
        cols=cols[:n_var],
        look_back=96
    )
    # 3 make forecast
    y_var_pred = [
        model_fitted.forecast(x[n], steps=1)
        for n in range(x.shape[0])
    ]
    y_var_pred = np.concatenate(y_var_pred, axis=0)
    y_point_pred = x[:, -1, :]

    # mean mae
    print('mean-overall-mae:\t', mean_mae(y).mean())
    print('mean-mae:\n', mean_mae(y))
    # pre-point mae
    print('point-overall-mae:\t', mean_absolute_error(y, y_point_pred))
    print('point-mae:\n', mean_absolute_error(y, y_point_pred, multioutput='raw_values'))
    # var model mae
    print('var_model-overall-mae:\t', mean_absolute_error(y, y_var_pred))
    print('var_model-mae:\n', mean_absolute_error(y, y_var_pred, multioutput='raw_values'))

    np.savez_compressed(exps_dir+'results/y_var_pred', y=y_var_pred)
    return


if __name__ == '__main__':
    print('*** ', datetime.now(), '\t Start runing run_var.py\t', EXP_DIR)
    start = datetime.now()

    if MODE == 'train':
        run_train_var_model(EXP_DIR)
    elif MODE == 'test':
        run_test_var_model(EXP_DIR)
    else:
        print('Please choose proper mode!!!')
    pass

    print('*** ', datetime.now(), '\t Exit Successfully.')
    end = datetime.now()
    print('\tComsumed Time: ', end - start, '\n')
    pass
