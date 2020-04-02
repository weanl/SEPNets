
import sys
import os
from datetime import datetime

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import tensorflow as tf
from utils.configs import get_config_from_json



# nohup python test.py 1>>train_log.out 2>&1 &
if __name__ == "__main__":
    print('*** ', datetime.now(), '\t Start runing gpu_test.py')
    start = datetime.now()

    print('Is GPU available: ', tf.test.is_gpu_available())
    exp_config, _exp_config = get_config_from_json('../../exp_ElectricityLoad/' + 'exp_config.json')
    print(exp_config.exp_name)
    print(exp_config.N_VAR)
    print(exp_config.VARS)

    print('*** ', datetime.now(), '\t Exit Successfully.')
    end = datetime.now()
    print('\tComsumed Time: ', end - start, '\n')
    pass

