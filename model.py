import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
from gym.spaces import Box, Discrete

##### Model construction #####
class MLP(tf.keras.Model):       # def mlp in def create_ppo_model
    def __init__(self, odim=24, adim=8, hdims=[256,256], actv='relu',
                 output_actv='relu'):
        super(MLP, self).__init__()
        self.hdims = hdims
        self.actv = actv
        self.ouput_actv = output_actv
        self.layers_ = tf.keras.Sequential()
        ki = tf.keras.initializers.truncated_normal(stddev=0.1)
        self.layers_.add(tf.keras.layers.InputLayer(input_shape=(odim,)))
        for hdim in self.hdims:
            linear = tf.keras.layers.Dense(hdim, kernel_initializer=ki, activation=actv)
            self.layers_.add(linear)
        linear_out = tf.keras.layers.Dense(adim, kernel_initializer=ki, activation=output_actv)
        self.layers_.add(linear_out)

    @tf.function
    def call(self, obs):
        x = obs
        mu = self.layers_(x)
        return mu

# weight와 동일하게 dictionary 형태로 noise 만들고 main에서 더하는 걸로.
def get_noises_from_weights(weights, nu=0.01):
    noises = []
    for weight in weights:
        noise = nu * np.random.randn(*weight.shape)
        noises.append(noise)
    return noises # dictionary