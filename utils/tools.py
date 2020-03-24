
from scipy.signal import correlate
import numpy as np
import statsmodels.api as sm

from scipy import stats
from statsmodels.graphics.api import qqplot

from sklearn.metrics import mean_absolute_error


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
    pass
