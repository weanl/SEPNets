
import sys
import os
from datetime import datetime

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from utils.configs import get_config_from_json
import numpy as np

from data_loader.generator import cons_mv_data
from sklearn.metrics import mean_absolute_error
from utils.tools import mean_mae

from keras import Sequential, Model
from keras.layers import GRU, Dense, LSTM, Input, Dropout
from keras.callbacks import EarlyStopping


N_VAR = 32
VARS = ['MT_001', 'MT_002', 'MT_003', 'MT_004', 'MT_005', 'MT_006', 'MT_007', 'MT_008', 'MT_009', 'MT_010',
        'MT_011', 'MT_012', 'MT_013', 'MT_014', 'MT_015', 'MT_016', 'MT_017', 'MT_018', 'MT_019', 'MT_020',
        'MT_021', 'MT_022', 'MT_023', 'MT_024', 'MT_025', 'MT_026', 'MT_027', 'MT_028', 'MT_029', 'MT_030',
        'MT_031', 'MT_032']


def run_rnn(exps_dir, mode='train'):
    num_var = N_VAR

    inputs = Input(shape=(96, num_var))
    rnn_outputs = LSTM(units=32)(inputs)
    rnn_outputs = Dropout(0.1)(rnn_outputs)
    outputs = Dense(units=num_var)(rnn_outputs)
    model = Model(inputs, outputs)

    model.compile(optimizer='Adam', loss='mse')
    model.summary()
    #
    if mode == 'train':
        x, y = cons_mv_data(
            data_file=exps_dir + 'dataset/training.csv',
            cols=VARS[:N_VAR],
            look_back=96
        )
        # UR4ML_residual_fit(model, X, y)
        model.fit(
            x,
            y,
            batch_size=32,
            epochs=400,  # 200
            callbacks=[EarlyStopping(monitor='loss', patience=3, mode='min')],
            validation_split=0.0,
            verbose=2
        )
        #
        saved_models_file = exps_dir + 'saved_models/rnn/model_weights_' + str(N_VAR)
        print('Model saved ... ', saved_models_file)
        model.save_weights(saved_models_file)
    elif mode == 'test':
        saved_models_file = exps_dir + 'saved_models/rnn/model_weights_' + str(N_VAR)
        model.load_weights(saved_models_file)
    else:
        raise ValueError('choose running mode in [train, test]')

    x, y = cons_mv_data(
        data_file=exps_dir + 'dataset/testing.csv',
        cols=VARS[:N_VAR],
        look_back=96
    )
    y_pred = model.predict(x)
    y_point_pred = x[:, -1, :]
    # print(y[0], '\n', y_pred[0], '\n', ar_pred[0], '\n', y_res_pred[0])

    # mean mae
    print('mean-overall-mae:\t', mean_mae(y).mean())
    print('mean-mae:\n', mean_mae(y))
    # pre-point mae
    print('point-overall-mae:\t', mean_absolute_error(y, y_point_pred))
    print('point-mae:\n', mean_absolute_error(y, y_point_pred, multioutput='raw_values'))
    # rnn model mae
    print('rnn_model-overall-mae:\t', mean_absolute_error(y, y_pred))
    print('rnn_model-mae:\n', mean_absolute_error(y, y_pred, multioutput='raw_values'))

    if mode == 'test':
        y_rnn_pred = y_pred
        np.savez_compressed(exps_dir + 'results/' + 'y_rnn_pred_' + str(N_VAR), y=y_rnn_pred)
        pass


MODE_LIST = ['train', 'test', 'visual']
MODE = MODE_LIST[1]


if __name__ == '__main__':
    print('*** ', datetime.now(), '\t Start runing run_rnn.py')
    start = datetime.now()

    run_rnn('../../exp_ElectricityLoad/', mode=MODE)

    print('*** ', datetime.now(), '\t Exit Successfully.')
    end = datetime.now()
    print('\tComsumed Time: ', end - start, '\n')
    pass