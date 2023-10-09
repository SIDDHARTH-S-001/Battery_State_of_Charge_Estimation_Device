#    This file was created by
#    MATLAB Deep Learning Toolbox Converter for TensorFlow Models.
#    23-Mar-2023 09:54:54

import tensorflow as tf
from tensorflow import keras
from keras import layers

def create_model():
    sequenceinput_unnormalized = keras.Input(shape=(None,5))
    sequenceinput = SubtractConstantLayer((5,), name="sequenceinput_")(sequenceinput_unnormalized)
    fc_1 = layers.Dense(55, name="fc_1_")(sequenceinput)
    layer = layers.Activation('tanh')(fc_1)
    fc_2 = layers.Dense(55, name="fc_2_")(layer)
    leakyrelu = layers.LeakyReLU(alpha=0.300000)(fc_2)
    fc_3 = layers.Dense(1, name="fc_3_")(leakyrelu)
    clippedrelu = layers.ReLU(max_value=1.000000)(fc_3)
    regressionoutput = clippedrelu

    model = keras.Model(inputs=[sequenceinput_unnormalized], outputs=[regressionoutput])
    return model

## Helper layers:

class SubtractConstantLayer(tf.keras.layers.Layer):
    def __init__(self, shape, name=None):
        super(SubtractConstantLayer, self).__init__(name=name)
        self.const = tf.Variable(initial_value=tf.zeros(shape), trainable=False)

    def call(self, input):
        return input - self.const

