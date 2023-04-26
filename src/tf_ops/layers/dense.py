import tensorflow as tf
from typing import List, Any

class DenseLayers(tf.keras.layers.Layer):
    def __init__(self, layers: List[Any], activation, batch_norm, layer_norm, target_dim, out_activation):
        super().__init__()
        self.layers = layers
        self.activation = activation
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.target_dim = target_dim
        self.out_activation = out_activation

    def build(self, input_shape):
        layers = []
        for layer in self.layers:
            layers.append(
                tf.keras.layers.Dense(
                    units=layer,
                    activation=self.activation,
                )
            )
            if self.batch_norm:
                layers.append(tf.keras.layers.BatchNormalization())
            if self.layer_norm:
                layers.append(tf.keras.layers.LayerNormalization())

        layers.append(tf.keras.layers.Dense(
                units=self.target_dim,
                activation=self.out_activation,
            ))
        
        self.sequential = tf.keras.Sequential(layers)


    def call(self, input_tensor):
        return self.sequential(input_tensor)
        

class ResBlock(tf.keras.layers.Layer):
    def __init__(self, units_1, units_2, drop_rate):
        super().__init__()

        self.units_1 = units_1
        self.units_2 = units_2
        self.drop_rate = drop_rate

    def build(self, input_shape):
        self.layer1 = tf.keras.layers.Dense(
                units=self.units_1,
                activation='relu'
            )
        
        self.layer2 = tf.keras.layers.Dense(
                units=self.units_2,
                activation='relu'
            )
        self.dropout_layer = tf.keras.layers.Dropout(
                self.drop_rate
            )



    def call(self, input_tensor):
        x = self.layer1(input_tensor)
        x = self.layer2(x)
        x = self.dropout_layer(x)

        self.add = tf.keras.layers.Add()
        out = self.add([input_tensor, x])
        return tf.keras.layers.LayerNormalization()(out)
        



