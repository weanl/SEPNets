
from datetime import datetime
import tensorflow as tf


# nohup python test.py 1>>train_log.out 2>&1 &
if __name__ == "__main__":
    print('*** ', datetime.now(), '\t Start runing gpu_test.py')
    start = datetime.now()

    print('Is GPU available: ', tf.test.is_gpu_available())

    print('*** ', datetime.now(), '\t Exit Successfully.')
    end = datetime.now()
    print('\tComsumed Time: ', end - start, '\n')
    pass

