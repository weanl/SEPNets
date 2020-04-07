
import sys
import os
from datetime import datetime

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import pandas as pd
from utils.tools import slide_window_x, slide_window_y
from utils.configs import get_config_from_json


EXP_DIRS = ['../../exp_ElectricityLoad/', 
            '../../exp_210100063/',
            '../../exp_201812/',
            '../../exp_210100112/']
EXP_DIR = EXP_DIRS[3]
exp_config, _exp_config = get_config_from_json(EXP_DIR + 'exp_config.json')
N_VAR = exp_config.N_VAR
VARS = exp_config.VARS


# generate training and testing files
class DataGenerator:

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_file = data_dir + 'LD2011_2014.txt'

        self.load_file()
        print(self.data.shape)
        print(self.data.columns)
        print(self.data.index.shape)

    def load_file(self):
        self.data = pd.read_csv(self.data_file, sep=';', dtype=str)
        all_columns = self.data.columns
        self.record_time = pd.to_datetime(self.data[all_columns[0]])
        self.data = self.data.drop(all_columns[0], axis=1)
        self.data.index = self.record_time
        self.n_var_names = all_columns[1:]

    def pick_train_test(self):
        sub_start = pd.datetime(2013, 1, 1, 0, 15, 00)
        sub_train_split = pd.datetime(2014, 1, 1, 0, 15, 00)
        sub_end = pd.datetime(2015, 1, 1, 0, 15, 00)
        self.sub_data = self.data[self.data.index >= sub_start]
        self.sub_data = self.sub_data[self.sub_data.index < sub_end]
        print(self.sub_data.shape)

        # convert '5,0761421319797' to 50761421319797
        self.sub_data = self.sub_data.apply(
            lambda col: col.apply(lambda x: float(x.replace(',', '')))
        )
        print(self.sub_data.dtypes)
        # modify some points

        # normalization
        self.sub_data = self.sub_data.apply(
            lambda col: (col-col.mean())/(col.std()+1)
        )
        print(self.sub_data)

        # generate train and test files
        train_file = self.data_dir + 'training.csv'
        test_file = self.data_dir + 'testing.csv'
        self.sub_data[self.sub_data.index < sub_train_split].to_csv(train_file)
        self.sub_data[self.sub_data.index >= sub_train_split].to_csv(test_file)
        pass


class DataGenerator2(object):

    def __init__(self, exp_dir):
        self.exp_dir = exp_dir
        self.data_file = exp_dir + 'dataset/pearsonr.csv'
        self.load_file()
        print(self.data.shape)
        # print(self.data.columns)
        print(self.data.index.shape)
    
    # load file of dataset (sep=',')
    def load_file(self):
        self.data = pd.read_csv(self.data_file, sep=',', dtype=str)
        all_columns = self.data.columns
        self.record_time = pd.to_datetime(self.data[all_columns[0]])
        self.data = self.data.drop(all_columns[0], axis=1)
        self.data.index = self.record_time
        self.n_var_names = all_columns[1:]
        return

    #
    def pick_train_test(self):
        # training split
        train_rate = 0.8
        train_split = int(self.data.shape[0]*train_rate)
        train_split_index = self.data.index[train_split]
        print('training split timestamp: ', train_split_index)
        self.data = self.data.apply(
            lambda col: col.apply(lambda x: float(x.replace(',', '')))
        )
        # normalization
        self.data = self.data.apply(
            lambda col: (col-col.mean())/(col.std()+1)
        )
        # generate train and test file
        train_file = self.exp_dir + 'dataset/training.csv'
        test_file = self.exp_dir + 'dataset/testing.csv'
        self.data[self.data.index < train_split_index].to_csv(train_file)
        self.data[self.data.index >= train_split_index].to_csv(test_file)        
        return


def generator_test(data_file):
    data = DataGenerator(data_file)
    data.pick_train_test()


def cons_ur_data(data_file, col, look_back, forecast_step=1):
    data = pd.read_csv(data_file)
    ur = data[col].values.reshape(-1, 1)

    x = ur[:-forecast_step]
    y = ur[forecast_step:]
    x_seqs = slide_window_x(x, look_back, forecast_step)
    y_seqs = slide_window_y(y, look_back, forecast_step)

    x_seqs = x_seqs.reshape(-1, look_back)
    y_seqs = y_seqs.reshape(-1,)
    print(x_seqs.shape)
    print(y_seqs.shape)

    return x_seqs, y_seqs


def cons_mv_data(data_file, cols, look_back, forecast_step=1):
    data = pd.read_csv(data_file)
    mr = data[cols].values

    x = mr[:-forecast_step]
    y = mr[forecast_step:]
    x_seqs = slide_window_x(x, look_back, forecast_step)
    y_seqs = slide_window_y(y, look_back, forecast_step)

    y_seqs = y_seqs.reshape(-1, len(cols))
    print(x_seqs.shape)
    print(y_seqs.shape)

    return x_seqs, y_seqs


def load_mv_data(data_file, cols):
    data = pd.read_csv(data_file)
    all_cols = data.columns
    record_time = pd.to_datetime(data[all_cols[0]])
    data = data.drop(all_cols[0], axis=1)
    data.index = record_time
    data.index.name = 'record_time'
    return data[cols]


if __name__ == '__main__':
    # generator_test('../exps/dataset/')
    # cons_ur_data('../exps/dataset/training.csv', col='MT_001', look_back=96)
    # cons_mv_data('../exps/dataset/training.csv', cols=VARS, look_back=96)
    data = DataGenerator2(EXP_DIR)
    data.pick_train_test()
    pass
