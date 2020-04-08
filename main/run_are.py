
import sys
import os
from datetime import datetime

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import numpy as np
import json

from models.ear_model import ARE
from utils.configs import get_config_from_json
from sklearn.metrics import mean_absolute_error
from utils.tools import mean_mae

from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import EarlyStopping

from data_loader.generator import cons_ur_data, cons_mv_data


EXP_DIRS = ['../../exp_ElectricityLoad/', 
            '../../exp_210100063/',
            '../../exp_201812/',
            '../../exp_210100112/']
EXP_DIR = EXP_DIRS[0]
exp_config, _exp_config = get_config_from_json(EXP_DIR + 'exp_config.json')
N_VAR = exp_config.N_VAR
VARS = exp_config.VARS
Max_Window = exp_config.Max_Window
Max_Epoch = exp_config.Max_Epoch

MODE_LIST = ['train', 'test', 'visual']
MODE = MODE_LIST[1]


def run_train_ARE(exps_dir):
    return


def run_train_AREs(exps_dir):
    n_var = N_VAR
    cols = VARS
    for n in range(n_var):
        # load json configure file
        config_file = exps_dir + 'saved_models/are/are_' + cols[n] + '_configs.json'
        configs, configs_dict = get_config_from_json(config_file)
        option_lags = configs.option_lags

        best_params = configs_dict  # need grid-search
        args = {}
        args['look_back'] = best_params['lag']
        # make model
        model = ARE(args).make_model()
        model.summary()

        # training
        x, y = cons_ur_data(
            data_file=exps_dir + 'dataset/training.csv',
            col=cols[n],
            look_back=best_params['lag']
        )
        model.fit(
            x, y, batch_size=32, epochs=Max_Epoch,
            callbacks=[EarlyStopping(monitor='loss', patience=8, mode='min')],
            validation_split=0,
            verbose=2
        )
        # save model
        saved_models_file = exps_dir + 'saved_models/are/are_' + cols[n] + '_weights'
        model.save_weights(saved_models_file)
        print('Model saved ... ', saved_models_file)

    return


def run_test_AREs(exps_dir):
    n_var = N_VAR
    cols = VARS
    p_list = []
    are_weights_files = []
    for n in range(n_var):
        # load json configure file
        config_file = exps_dir + 'saved_models/are/are_' + cols[n] + '_configs.json'
        configs, _ = get_config_from_json(config_file)
        p_list.append(configs.lag)
        are_weights_files.append(exps_dir+'saved_models/are/are_'+cols[n]+'_weights')
    # make model
    are_models = [
        ARE({'look_back': p}).make_model()
        for p in p_list
    ]
    # load weights
    [
        are_models[idx].load_weights(are_weights_files[idx])
        for idx in range(n_var)
    ]

    # make forecast
    test_x, test_y = cons_mv_data(
        data_file=exps_dir + 'dataset/testing.csv',
        cols=cols,
        look_back=Max_Window
    )
    test_y_are_pred = [
        are_models[idx].predict(
            test_x[:, -p_list[idx]:, idx]
        )
        for idx in range(n_var)
    ]  # make auto-regression[ARE] prediction with shape of (n_var, batch_size)
    test_y_are_pred = np.concatenate(test_y_are_pred, axis=-1)  # (batch_size, n_var)
    test_y_point_pred = test_x[:, -1, :]
    # make forecast for training
    train_x, train_y = cons_mv_data(
        data_file=exps_dir + 'dataset/training.csv',
        cols=cols,
        look_back=Max_Window
    )
    train_y_are_pred = [
        are_models[idx].predict(
            train_x[:, -p_list[idx]:, idx]
        )
        for idx in range(n_var)
    ]
    train_y_are_pred = np.concatenate(train_y_are_pred, axis=-1)
    train_y_point_pred = train_x[:, -1, :]

    # mean mae
    print('mean-overall-mae:\t', mean_mae(test_y).mean())
    print('mean-mae:\n', mean_mae(test_y))
    # pre-point mae
    print('point-overall-mae:\t', mean_absolute_error(test_y, test_y_point_pred))
    print('point-mae:\n', mean_absolute_error(test_y, test_y_point_pred, multioutput='raw_values'))
    # are model mae
    print('are_model-overall-mae:\t', mean_absolute_error(test_y, test_y_are_pred))
    print('are_model-mae:\n', mean_absolute_error(test_y, test_y_are_pred, multioutput='raw_values'))

    np.savez_compressed(
        exps_dir+'results/y_are_pred',
        train_y_pred=train_y_are_pred, 
        test_y_pred=test_y_are_pred
    )
    return


def run_visual_AREs(exps_dir):
    n_var = N_VAR
    cols = VARS
    p_list = []
    ar_weights_files = []
    for n in range(n_var):
        # load json configure file
        config_file = exps_dir + 'saved_models/are/are_' + cols[n] + '_configs.json'
        configs, _ = get_config_from_json(config_file)
        p_list.append(configs.lag)
        ar_weights_files.append(exps_dir+'saved_models/are/are_'+cols[n]+'_weights')
    # make model
    ar_models = [
        ARE({'look_back': p}).make_model()
        for p in p_list
    ]
    # load weights
    [
        ar_models[idx].load_weights(ar_weights_files[idx])
        for idx in range(n_var)
    ]
    # print
    print(ar_models[0].summary())
    [
        print(ar_models[idx].get_weights()[-1])
        for idx in range(n_var)
    ]
    return


if __name__ == '__main__':
    print('*** ', datetime.now(), '\t Start runing run_are.py\t', EXP_DIR)
    start = datetime.now()

    if MODE == 'train':
        run_train_AREs(EXP_DIR)
    elif MODE == 'test':
        run_test_AREs(EXP_DIR)
    elif MODE == 'visual':
        run_visual_AREs(EXP_DIR)
    else:
        print('Please choose proper mode!!!')

    pass

    print('*** ', datetime.now(), '\t Exit Successfully.')
    end = datetime.now()
    print('\tComsumed Time: ', end - start, '\n')
    pass
