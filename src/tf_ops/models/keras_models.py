from src.tf_ops.layers.dense import DenseLayers, ResBlock
import tensorflow as tf


class SimpleDenseClassifier(tf.keras.Model):
    def __init__(self, params):
        super(SimpleDenseClassifier, self).__init__()
        self.dense_layers = DenseLayers(
            **params
        )

    def call(self, input_tensor):
        x = self.dense_layers(input_tensor)
       
        return x
    

class SimpleResNetClassifier(tf.keras.Model):
    def __init__(self, params):
        super().__init__()
        self.res_layers = ResBlock(
            **params
        )

        self.out_layer = tf.keras.layers.Dense(1, activation='softmax')

    def call(self, input_tensor):
        x = self.res_layers(input_tensor)
        x = self.out_layer(x)
       
        return x
    