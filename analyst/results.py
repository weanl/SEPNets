
import sys
import os
from datetime import datetime

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from utils.configs import get_config_from_json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from scipy.spatial.distance import correlation


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
Skip = exp_config.Skip
Period = exp_config.Period


def get_mae(pred_file, y_truth, exp_dir):
    y_pred = np.load(exp_dir + 'results/' + pred_file + '.npz')

    test_y_pred = y_pred['test_y_pred']
    test_y_truth = y_truth['test_y_pred']
    test_mae, test_maes = get_one_mae(test_y_truth, test_y_pred)
    test_maes.append(test_mae)

    train_y_pred = y_pred['train_y_pred']
    train_y_truth = y_truth['train_y_pred']
    train_mae, train_maes = get_one_mae(train_y_truth, train_y_pred)
    train_maes.append(train_mae)

    return [train_maes, test_maes]


def run_get_mae(exp_dir, saved=False):
    y_truth = np.load(exp_dir + 'results/y_truth.npz')
    pred_files = ['y_ar_pred', 'y_are_pred', 'y_ear_pred', 'y_var_pred', 'y_rnn_pred', 
                'y_lstnets_pred', 'y_sepnets_pred_True_True']
    methods = ['ar-s-train', 'ar-s-test',
            'ar-e-train', 'ar-e-test',
            'ear-train', 'ear-test',
            'var-train', 'var-test',
            'lstm-train', 'lstm-test',
            'lstnets-train', 'lstnets-test',
            'sepnets-train', 'sepnets-test']
    vars_name = VARS
    
    data = [
        get_mae(pred_file, y_truth, exp_dir)
        for pred_file in pred_files
    ]
    data = np.array(data).reshape(-1, N_VAR+1).T
    vars_name.append('overall')
    df_data = pd.DataFrame(
        data,
        index=vars_name,
        columns=methods
    )

    if saved == True:
        df_data.to_excel(exp_dir + 'results/mae.xlsx')
    return df_data


def get_one_mae(y, y_pred):
    overall_mae =  mean_absolute_error(y, y_pred)
    mae =  mean_absolute_error(y, y_pred, multioutput='raw_values')
    return overall_mae, list(mae)


def get_corr(y_truth, pred_file, exp_dir):
    y_pred = np.load(exp_dir + 'results/' + pred_file + '.npz')

    test_y_pred = y_pred['test_y_pred']
    test_y_truth = y_truth['test_y_pred']
    test_corr = get_one_corr(test_y_pred, test_y_truth)

    train_y_pred = y_pred['train_y_pred']
    train_y_truth = y_truth['train_y_pred']
    train_corr = get_one_corr(train_y_pred, train_y_truth)

    return [train_corr, test_corr]


def run_get_corr(exp_dir, saved=False):
    y_truth = np.load(exp_dir + 'results/y_truth.npz')
    pred_files = ['y_ar_pred', 'y_are_pred', 'y_ear_pred', 'y_var_pred', 'y_rnn_pred', 
                'y_lstnets_pred', 'y_sepnets_pred_True_True']
    methods = ['ar-s-train', 'ar-s-test',
            'ar-e-train', 'ar-e-test',
            'ear-train', 'ear-test',
            'var-train', 'var-test',
            'lstm-train', 'lstm-test',
            'lstnets-train', 'lstnets-test',
            'sepnets-train', 'sepnets-test']
    
    data = [
        get_corr(y_truth, pred_file, exp_dir)
        for pred_file in pred_files
    ]
    data = np.array(data).reshape(1, -1)
    df_data = pd.DataFrame(
        data,
        index=['Corr'],
        columns=methods
    )
    if saved == True:
        df_data.to_excel(exp_dir + 'results/corr.xlsx')
    return df_data

def get_one_corr(y_pred, y):
    num_instance = y.shape[0]
    corrs = [
        correlation(y_pred[n], y[n])
        for n in range(num_instance)
    ]
    corrs = np.array(corrs)
    corr = np.mean(corrs)
    return corr


def run_get_mae_condition(exp_dir):
    df_mae = pd.read_excel(exp_dir + 'results/mae.xlsx')
    cols = df_mae.columns
    [
        df_mae[col].astype(float)
        for col in cols[1:]
    ]

    # electricity
    # base_method = 'ar-e-test'
    # methods = ['var-train', 'lstm-train', 'lstnets-test', 'sepnets-test']

    # 210100063
    # base_method = 'ar-s-test'
    # methods = ['ar-e-test', 'lstnets-test', 'sepnets-test', 'ear-test']

    # 201812
    base_method = 'ar-e-test'
    methods = ['ear-train', 'lstnets-train', 'lstnets-test', 'sepnets-test']

    base_method = 'ar-s-test'
    methods = ['ar-e-test', 'lstnets-test', 'var-train', 'sepnets-test']

    names = ['VAR', 'LSTM', 'LSTNets', 'SEPNets']

    data = [
        ((df_mae[base_method]-df_mae[method])/(1+df_mae[base_method])).values
        for method in methods
    ]
    data = np.array(data).T
    
    cond = data < 0
    cond = cond.astype(int).sum(axis=0)
    print('cond', cond)
    a_cond = N_VAR - cond
    print('a_cond', a_cond)
    mean_data = np.mean(data, axis=0)
    print('mean_data', mean_data)

    df_data = pd.DataFrame(
        data=[cond, a_cond, mean_data],
        index=['差', '优', '误差降低率'],
        columns=names
    )
    df_data.to_excel(exp_dir + 'results/condition.xlsx')

    return


if __name__ == '__main__':
    # run_get_mae(EXP_DIR)
    # run_get_corr(EXP_DIR, True)
    run_get_mae_condition(EXP_DIR)
    pass
