from src.tf_ops.layers.dense import DenseLayers
import tensorflow as tf


class SimpleDenseClassifier(tf.keras.Model):
    def __init__(self, params):
        super(SimpleDenseClassifier, self).__init__()
        self.dense_layers = DenseLayers(
            **params
        )


    def call(self, input_tensor):
        print(input_tensor.shape)
        x = self.dense_layers(input_tensor)
       
        
        return x
    