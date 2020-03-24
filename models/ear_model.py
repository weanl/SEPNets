
from keras.layers import Input, Dense, Lambda, concatenate
from keras.constraints import non_neg, unit_norm
from keras.models import Model
from keras.initializers import random_uniform


RND_UNI = random_uniform(minval=0.05, maxval=1.05)


class EAR(object):

    def __init__(self, args):
        super(EAR, self).__init__()
        self.look_back = args['look_back']

    def make_model(self):
        x = Input(shape=(self.look_back,))

        ar_output = Dense(
            units=1, kernel_initializer='uniform',
            kernel_constraint=unit_norm(), name='ar-weights'
        )(x)

        pre_point = Lambda(
            lambda k: k[:, -1:]
        )(x)

        merged_output = concatenate(
            [ar_output, pre_point]
        )

        outputs = Dense(
            units=1, kernel_initializer=RND_UNI, use_bias=False,
            kernel_constraint=non_neg(), name='contrib-weights'
        )(merged_output)

        model = Model(inputs=x, outputs=outputs)
        model.compile('Adam', 'mae')
        return model


class AR(object):

    def __init__(self, args):
        super(AR, self).__init__()
        self.look_back = args['look_back']

    def make_model(self):
        x = Input(shape=(self.look_back,))

        ar_output = Dense(
            units=1, kernel_initializer='uniform',
            kernel_constraint=unit_norm(), name='ar-weights'
        )(x)

        model = Model(inputs=x, outputs=ar_output)
        model.compile('Adam', 'mse')
        return model


class ARE(object):

    def __init__(self, args):
        super(ARE, self).__init__()
        self.look_back = args['look_back']

    def make_model(self):
        x = Input(shape=(self.look_back,))

        ar_output = Dense(
            units=1, kernel_initializer='uniform',
            kernel_constraint=unit_norm(), name='ar-weights'
        )(x)

        model = Model(inputs=x, outputs=ar_output)
        model.compile('Adam', 'mae')
        return model
