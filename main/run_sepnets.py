
import sys
import os
from datetime import datetime

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import numpy as np
import json

from models.sepnets_model import SEPNets
from utils.configs import get_config_from_json
from sklearn.metrics import mean_absolute_error
from utils.tools import mean_mae

from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import EarlyStopping

from data_loader.generator import cons_ur_data, cons_mv_data

N_VAR = 32
VARS = ['MT_001', 'MT_002', 'MT_003', 'MT_004', 'MT_005', 'MT_006', 'MT_007', 'MT_008', 'MT_009', 'MT_010',
        'MT_011', 'MT_012', 'MT_013', 'MT_014', 'MT_015', 'MT_016', 'MT_017', 'MT_018', 'MT_019', 'MT_020',
        'MT_021', 'MT_022', 'MT_023', 'MT_024', 'MT_025', 'MT_026', 'MT_027', 'MT_028', 'MT_029', 'MT_030',
        'MT_031', 'MT_032']


def run_train_SEPNets(exps_dir, pretrain, trainable):
    print('----pretrain: ', pretrain, '\t----trainable: ', trainable)
    n_var = N_VAR
    cols = VARS
    p_list = []
    ear_weights_files = []
    for n in range(n_var):
        # load json configure file
        config_file = exps_dir + 'saved_models/ear/ear_' + cols[n] + '_configs.json'
        configs, configs_dict = get_config_from_json(config_file)
        p_list.append(configs.lag)
        ear_weights_files.append(exps_dir+'saved_models/ear/ear_'+cols[n]+'_weights')

    args = {}
    args['look_back'] = 96
    args['n_var'] = n_var
    args['p_list'] = p_list
    args['se_weights_files'] = ear_weights_files
    args['pretrain'] = pretrain
    args['trainable'] = trainable

    # make model
    model = SEPNets(args).make_model()
    model.summary()

    # training
    x, y = cons_mv_data(
        data_file=exps_dir + 'dataset/training.csv',
        cols=cols[:N_VAR],
        look_back=96
    )
    model.fit(
        x, y, batch_size=32, epochs=400,
        callbacks=[EarlyStopping(monitor='loss', patience=8, mode='min')],
        validation_split=0,
        verbose=2
    )
    # save model
    saved_models_file = exps_dir + 'saved_models/sepnets/model_weights_' + \
                        str(pretrain) + '_' + str(trainable) + '_' + str(N_VAR)
    model.save_weights(saved_models_file)
    print('Model saved ... ', saved_models_file)

    return


def run_test_SEPNets(exps_dir, pretrain, trainable):
    print('----pretrain: ', pretrain, '\t----trainable: ', trainable)
    n_var = N_VAR
    cols = VARS
    p_list = []
    ear_weights_files = []
    for n in range(n_var):
        # load json configure file
        config_file = exps_dir + 'saved_models/ear/ear_' + cols[n] + '_configs.json'
        configs, _ = get_config_from_json(config_file)
        p_list.append(configs.lag)
        ear_weights_files.append(exps_dir+'saved_models/ear/ear_'+cols[n]+'_weights')
    # make model
    args = {}
    args['look_back'] = 96
    args['n_var'] = n_var
    args['p_list'] = p_list
    args['se_weights_files'] = ear_weights_files
    args['pretrain'] = pretrain
    args['trainable'] = trainable

    # make model
    model = SEPNets(args).make_model()
    print(model.summary())
    # load weights
    saved_models_file = exps_dir + 'saved_models/sepnets/model_weights_' + \
                        str(pretrain) + '_' + str(trainable) + '_' + str(N_VAR)
    model.load_weights(saved_models_file)
    # make forecast
    x, y = cons_mv_data(
        data_file=exps_dir + 'dataset/testing.csv',
        cols=cols[:N_VAR],
        look_back=96
    )
    y_sepnets_pred = model.predict(x)
    y_point_pred = x[:, -1, :]

    # mean mae
    print('mean-overall-mae:\t', mean_mae(y).mean())
    print('mean-mae:\n', mean_mae(y))
    # pre-point mae
    print('point-overall-mae:\t', mean_absolute_error(y, y_point_pred))
    print('point-mae:\n', mean_absolute_error(y, y_point_pred, multioutput='raw_values'))
    # sepnets model mae
    print('sepnets_model-overall-mae:\t', mean_absolute_error(y, y_sepnets_pred))
    print('sepnets_model-mae:\n', mean_absolute_error(y, y_sepnets_pred, multioutput='raw_values'))

    saved_results = exps_dir+'results/y_sepnets_pred_'+ \
                    str(pretrain) + '_' + str(trainable) + '_' + str(N_VAR)
    np.savez_compressed(saved_results, y=y_sepnets_pred)
    return


# network structure and contribution weights
def run_visual_SEPNets(exps_dir, pretrain, trainable):
    print('----pretrain: ', pretrain, '\t----trainable: ', trainable)
    n_var = N_VAR
    cols = VARS
    p_list = []
    ear_weights_files = []
    for n in range(n_var):
        # load json configure file
        config_file = exps_dir + 'saved_models/ear/ear_' + cols[n] + '_configs.json'
        configs, configs_dict = get_config_from_json(config_file)
        p_list.append(configs.lag)
        ear_weights_files.append(exps_dir + 'saved_models/ear/ear_' + cols[n] + '_weights')

    args = {}
    args['look_back'] = 96
    args['n_var'] = n_var
    args['p_list'] = p_list
    args['se_weights_files'] = ear_weights_files
    args['pretrain'] = pretrain
    args['trainable'] = trainable

    # make model
    model = SEPNets(args).make_model()
    print(model.summary())
    print(model.get_weights()[-2:])
    return


MODE_LIST = ['train', 'test', 'visual']
MODE = MODE_LIST[1]
PRETRAIN_TRANABLE = [(True, True)]# [(False, True), (True, True), (True, False)]


if __name__ == '__main__':
    print('*** ', datetime.now(), '\t Start runing run_sepnets.py')
    start = datetime.now()

    if MODE == 'train':
        [
            run_train_SEPNets('../../exp_ElectricityLoad/', pretrain, trainable)
            for pretrain, trainable in PRETRAIN_TRANABLE
        ]
    elif MODE == 'test':
        for pretrain, trainable in PRETRAIN_TRANABLE:
            run_test_SEPNets('../../exp_ElectricityLoad/', pretrain, trainable)
    elif MODE == 'visual':
        for pretrain, trainable in PRETRAIN_TRANABLE:
            run_visual_SEPNets('../../exp_ElectricityLoad/', pretrain, trainable)
    else:
        print('Please choose proper mode!!!')
    pass

    print('*** ', datetime.now(), '\t Exit Successfully.')
    end = datetime.now()
    print('\tComsumed Time: ', end - start, '\n')
    pass