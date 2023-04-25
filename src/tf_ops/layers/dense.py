import tensorflow as tf

class DenseLayers(tf.keras.layers.Layer):
    def __init__(self, layers, activation, batch_norm, layer_norm, target_dim):
        super().__init__()
        self.layers = layers
        self.activation = activation
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.target_dim = target_dim

    def call(self, input_tensor):
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

            if self.target_dim:
                layers.append(
                    tf.keras.layers.Dense(
                        units=self.target_dim,
                        activation='linear',
                    )
                )

            self.sequntial = tf.keras.Sequential(layers)
            return self.sequntial(input_tensor)
        



