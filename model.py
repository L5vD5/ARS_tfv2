import tensorflow as tf
import numpy as np
from gym.spaces import Box, Discrete

##### Model construction #####
class MLP(tf.keras.Model):       # def mlp in def create_ppo_model
    def __init__(self, odim=24, adim=8, hdims=[256,256], actv='relu',
                 out_actv='relu'):
        super(MLP, self).__init__()
        self.hdims = hdims
        self.layers_ = tf.keras.Sequential()
        ki = tf.keras.initializers.truncated_normal(stddev=0.1)
        self.layers_.add(tf.keras.layers.InputLayer(input_shape=(odim,)))
        for hdim in self.hdims:
            linear = tf.keras.layers.Dense(hdim, kernel_initializer=ki, activation=actv)
            self.layers_.add(linear)
        linear_out = tf.keras.layers.Dense(adim, kernel_initializer=ki, activation=out_actv)
        self.layers_.add(linear_out)

    @tf.function
    def call(self, obs):
        x = obs
        mu = self.layers_(x)
        return mu

def get_noises_from_weights(weights, nu=0.01):
    noises = []
    for weight in weights:
        noise = nu * np.random.randn(*weight.shape)
        noises.append(noise)
    return noises