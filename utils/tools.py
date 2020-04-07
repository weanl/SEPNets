
import sys
import os
from datetime import datetime

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)


from utils.configs import get_config_from_json
from scipy.signal import correlate
import numpy as np
import statsmodels.api as sm

from scipy import stats
from statsmodels.graphics.api import qqplot

from sklearn.metrics import mean_absolute_error
import json


EXP_DIRS = ['../../exp_ElectricityLoad/', 
            '../../exp_210100063/',
            '../../exp_201812/',
            '../../exp_210100112/']
EXP_DIR = EXP_DIRS[3]
exp_config, _exp_config = get_config_from_json(EXP_DIR + 'exp_config.json')
N_VAR = exp_config.N_VAR
VARS = exp_config.VARS
Max_Window = exp_config.Max_Window
Max_Epoch = exp_config.Max_Epoch


def create_json_file(data, file):
    with open(file, 'w') as f:
        json.dump(data, f)
    return


def create_json_files(models=[]):
    commom_dict = {
        "optimizer": "Adam", 
        "lag": Max_Window, 
        "batch_size": 32, 
        "epochs": Max_Epoch, 
        "option_lags": [Max_Window]
    }
    for model in models:
        [
            create_json_file(
                data=commom_dict,
                file=EXP_DIR+'saved_models/'+model+'/'+model+'_'+v+'_configs.json'
            )
            for v in VARS
        ]
    return


def slide_window_x(x, look_back=60, forecast_step=10, test_flag=False):

    assert x.ndim == 2
    # the length of x in time dimension
    span = x.shape[0]

    history = []
    step = 1
    # step = 10
    # if it is for test
    #   every another forecast_step, make a forecast
    if test_flag:
        step = forecast_step
    for i in range(0, span-look_back+1, step):
        # look_back of one window of x
        win_x = list(x[i:i+look_back])
        history.append(win_x)

    history = np.array(history)
    return history


def slide_window_y(y, look_back=60, forecast_step=10, test_flag=False):

    assert y.ndim == 2
    # the length of x in time dimension
    span = y.shape[0]

    history = []
    step = 1
    # step = 10
    # if it is for test
    #   every another forecast_step, make a forecast
    if test_flag:
        step = forecast_step
    for i in range(0, span-look_back+1, step):
        # look_back of one window of x
        win_y = list(y[i+look_back-forecast_step:i+look_back])
        history.append(win_y)

    history = np.array(history)
    return history


def _test():
    s = [1, 2, 3, 1.5, 1, 2, 3, 2, 1, 2.5, 3, 2, 1, 2, 3, 2,
         1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2,
         1, 2, 3, 2, 1, 2, 3, 1.5, 1, 2, 3, 2, 1, 2, 3, 2,
         1, 2, 3, 2.5, 1, 2, 3, 2, 1, 1.5, 3, 2, 1, 2, 3, 2]
    signal_options_p(s+s+s+s)


def test_mae():
    y = [[1,1], [2, 2]]
    y_pred = [[1, 1], [1, 2]]
    print(mean_absolute_error(y, y_pred))


def mean_mae(y):  # (num_instance, n_var)
    assert y.ndim == 2
    num_instance = y.shape[0]
    y_mean = [
        np.ones((num_instance,))*np.mean(y[:, col])
        for col in range(y.shape[-1])
    ]
    y_mean = np.array(y_mean).T
    print(y_mean.shape)

    return mean_absolute_error(y, y_mean, multioutput='raw_values')


if __name__ == '__main__':
    # test_mae()
    # res = mean_mae(np.array([[1,1], [2, 2], [2, 2], [3, 3]]))
    # print(res, res.mean())
    create_json_files(['ar', 'are', 'ear'])
    pass
