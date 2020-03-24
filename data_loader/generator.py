
import pandas as pd
from utils.tools import slide_window_x, slide_window_y

N_VAR = 10
VARS = ['MT_001', 'MT_002', 'MT_003', 'MT_004', 'MT_005', 'MT_006', 'MT_007', 'MT_008', 'MT_009', 'MT_010']


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
    cons_mv_data('../exps/dataset/training.csv', cols=VARS, look_back=96)
    pass
