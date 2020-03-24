

import tensorflow as tf
from keras.layers import Input, Dense, Lambda, concatenate, Conv1D, GRU, add, Add, LSTM
from keras.constraints import non_neg, unit_norm
from keras.models import Model
from keras.layers import Layer
from keras.initializers import random_uniform

from models.ear_model import EAR


RND_UNI = random_uniform(minval=0.05, maxval=1.05)


def set_non_trainable(model):
    for layer in model.layers:
        layer.trainable = False
    return


class Multiply(Layer):

    def __init__(self, unit, **kwargs):
        super(Multiply, self).__init__(**kwargs)
        self.build(unit)

    def build(self, unit, input_shape=None):
        self.prop_weights = self.add_weight(
            name='proportion-weights',
            shape=(unit,),
            initializer=RND_UNI,
            trainable=True,
            constraint=non_neg()
        )
        super(Multiply, self).build(input_shape)

    def call(self, input):
        return tf.multiply(
         name='ele-wise-product',
         x=input,
         y=self.prop_weights
        )


class SEPNets(object):

    def __init__(self, args):
        super(SEPNets, self).__init__()
        self.look_back = args['look_back']
        self.n_var = args['n_var']
        self.p_list = args['p_list']
        self.se_weights_files = args['se_weights_files']
        self.pretrain = args['pretrain']
        self.trainable = args['trainable']

    def make_model(self):
        x = Input(shape=(self.look_back, self.n_var))

        # make self-evolution
        se_models = self.make_se_model(self.pretrain, self.trainable)
        se_outputs = [
            se_models[idx](
                inputs=Lambda(
                    lambda k: k[:, -self.p_list[idx]:, idx]
                )(x)
            )
            for idx in range(self.n_var)
        ]
        se_pred = concatenate(se_outputs)

        # make res
        c1 = Conv1D(
            filters=1, kernel_size=1, name='Conv1D-1'
        )(x)
        c3 = Conv1D(
            filters=1, kernel_size=3, name='Conv1D-3'
        )(x)
        c5 = Conv1D(
            filters=1, kernel_size=5, name='Conv1D-5'
        )(x)

        r1 = LSTM(
            units=16, name='LSTM-1'
        )(c1)
        r3 = LSTM(
            units=16, name='LSTM-3'
        )(c3)
        r5 = LSTM(
            units=16, name='LSTM-5'
        )(c5)
        r135 = add([r1, r3, r5])
        res_pred = Dense(
            units=self.n_var,
            kernel_initializer='uniform', kernel_constraint=unit_norm()
        )(r135)

        # make final
        se = Multiply(
            unit=self.n_var, name='se-weights'
        )(se_pred)
        res = Multiply(
            unit=self.n_var, name='res-weights'
        )(res_pred)
        y_pred = Add()([se, res])

        model = Model(inputs=x, outputs=y_pred)
        model.compile('Adam', 'mae')
        return model

    def make_se_model(self, pretrain=True, trainable=False):
        se_models = [
            EAR({'look_back': self.p_list[idx]}).make_model()
            for idx in range(self.n_var)
        ]
        if pretrain:
            [
                se_models[idx].load_weights(self.se_weights_files[idx])
                for idx in range(self.n_var)
            ]
        if pretrain and not trainable:
            [
                set_non_trainable(se_models[idx])
                for idx in range(self.n_var)
            ]
        return se_models
