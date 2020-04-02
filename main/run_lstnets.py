
import sys
import os
from datetime import datetime

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from utils.configs import get_config_from_json
import numpy as np
from models.lstnets import LSTNet_multi_inputs

from data_loader.generator import cons_mv_data
from sklearn.metrics import mean_absolute_error
from utils.tools import mean_mae
from bunch import Bunch

from keras.callbacks import EarlyStopping


EXP_DIR = '../../exp_ElectricityLoad/'
exp_config, _exp_config = get_config_from_json(EXP_DIR + 'exp_config.json')
N_VAR = exp_config.N_VAR
VARS = exp_config.VARS

MODE_LIST = ['train', 'test', 'visual']
MODE = MODE_LIST[1]


ARGS = {
    "window": 4*24,
    'hidRNN': 16,
    'hidCNN': 3,
    'hidSkip': 4,
    'CNN_kernel': 4,
    'skip': 4,
    'ps': 24,
    'highway_window': 4*24,
    'dropout': 0.2,
    'output_fun': 'linear',
    'lr': 0.001,
    'loss':'mae',
    'clip': 10
}
ARGS = Bunch(ARGS)


def run_lstnets_model(exps_dir, args, n_dim=N_VAR, mode='train'):
    model = LSTNet_multi_inputs(args, n_dim).make_model()
    model.summary()

    if mode == 'train':
        # load training data
        x, y = cons_mv_data(
            data_file=exps_dir + 'dataset/training.csv',
            cols=VARS[:N_VAR],
            look_back=4*24
        )
        # train
        model.fit(
            [x, x],
            y,
            batch_size=32,
            epochs=400,  # 200
            callbacks=[EarlyStopping(monitor='loss', patience=3, mode='min')],
            validation_split=0,
            verbose=2
        )
        #
        saved_models_file = exps_dir + 'saved_models/lstnets/model_weights_' + str(N_VAR)
        print('Model saved ... ', saved_models_file)
        model.save_weights(saved_models_file)
    elif mode == 'test':
        saved_models_file = exps_dir + 'saved_models/lstnets/model_weights_' + str(N_VAR)
        model.load_weights(saved_models_file)
    else:
        raise ValueError('choose running mode in [train, test]')

    # load testing data
    x, y = cons_mv_data(
        data_file=exps_dir + 'dataset/testing.csv',
        cols=VARS[:N_VAR],
        look_back=4*24
    )
    y_lstnet_pred = model.predict([x, x])
    y_point_pred = x[:, -1, :]

    # mean mae
    print('mean-overall-mae:\t', mean_mae(y).mean())
    print('mean-mae:\n', mean_mae(y))
    # pre-point mae
    print('point-overall-mae:\t', mean_absolute_error(y, y_point_pred))
    print('point-mae:\n', mean_absolute_error(y, y_point_pred, multioutput='raw_values'))
    # lstnet model mae
    print('lstnet_model-overall-mae:\t', mean_absolute_error(y, y_lstnet_pred))
    print('lstnet_model-mae:\n', mean_absolute_error(y, y_lstnet_pred, multioutput='raw_values'))

    if mode == 'test':
        np.savez_compressed(exps_dir + 'results/' + 'y_lstnets_pred_' + str(N_VAR), y=y_lstnet_pred)
        np.savez_compressed(exps_dir + 'results/' + 'y_truth_32', y=y)
    return


if __name__ == '__main__':
    print('*** ', datetime.now(), '\t Start runing run_lstnets.py\t', EXP_DIR)
    start = datetime.now()

    run_lstnets_model(EXP_DIR, ARGS, mode=MODE)

    print('*** ', datetime.now(), '\t Exit Successfully.')
    end = datetime.now()
    print('\tComsumed Time: ', end - start, '\n')
    pass
