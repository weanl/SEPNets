
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


N_VAR = 32
VARS = ['MT_001', 'MT_002', 'MT_003', 'MT_004', 'MT_005', 'MT_006', 'MT_007', 'MT_008', 'MT_009', 'MT_010',
        'MT_011', 'MT_012', 'MT_013', 'MT_014', 'MT_015', 'MT_016', 'MT_017', 'MT_018', 'MT_019', 'MT_020',
        'MT_021', 'MT_022', 'MT_023', 'MT_024', 'MT_025', 'MT_026', 'MT_027', 'MT_028', 'MT_029', 'MT_030',
        'MT_031', 'MT_032']


def run_get_mae(exp_dir):
    exp_dir += 'results/'
    cols = ['ar-s', 'ar-e', 'var', 'lstm', 'lstnets', 'sepnets_11', 'sepnets10', 'ear']
    y_truth = np.load(exp_dir + 'y_truth_32.npz')['y'][:, :N_VAR]
    y_ar = np.load(exp_dir + 'y_ar_pred.npz')['y'][:, :N_VAR]
    y_are = np.load(exp_dir + 'y_are_pred.npz')['y'][:, :N_VAR]
    y_var = np.load(exp_dir + 'y_var_pred.npz')['y'][:, :N_VAR]
    y_lstm = np.load(exp_dir + 'y_rnn_pred_32.npz')['y'][:, :N_VAR]
    y_lstnets = np.load(exp_dir + 'y_lstnets_pred_32.npz')['y'][:, :N_VAR]
    y_sepnets11 = np.load(exp_dir + 'y_sepnets_pred_True_True_32.npz')['y'][:, :N_VAR]
    y_sepnets10 = np.load(exp_dir + 'y_sepnets_pred_True_False_32.npz')['y'][:, :N_VAR]
    y_ear = np.load(exp_dir + 'y_ear_pred_32.npz')['y'][:, :N_VAR]

    y_preds = [y_ar, y_are, y_var, y_lstm, y_lstnets, y_sepnets11, y_sepnets10, y_ear]
    data = []
    for y_pred in y_preds:
        overall_mae, mae = get_one_mae(y_truth, y_pred)
        mae.append(overall_mae)
        data.append(mae)
    data = np.array(data).T
    VARS.append('overall')
    df_data = pd.DataFrame(
        data,
        index=VARS,
        columns=cols
    )
    df_data.to_csv(exp_dir + 'mae_32.csv')
    return


def get_one_mae(y, y_pred):
    overall_mae =  mean_absolute_error(y, y_pred)
    mae =  mean_absolute_error(y, y_pred, multioutput='raw_values')
    return overall_mae, list(mae)


if __name__ == '__main__':
    run_get_mae('../../exp_ElectricityLoad/')
    pass
