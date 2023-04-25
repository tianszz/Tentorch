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
        

class ResBlock(tf.keras.layers.Layer):
    def __init__(self, units_1, units_2, drop_rate):
        super().__init__()

        self.units_1 = units_1
        self.units_2 = units_2
        self.drop_rate = drop_rate

    def call(self, input_tensor):

        layers = []
        layers.append(
            tf.keras.layers.Dense(
                units=self.units_1,
                activation='relu'
            )
        )
        layers.append(
            tf.keras.layers.Dense(
                units=self.units_2,
                activation='linear'
            )
        )
        layers.append(
            tf.keras.layers.Dropout(
                self.drop_rate
            )
        )

        self.denses = tf.keras.Sequential(layers)
        self.output = self.denses(input_tensor)
        self.add = tf.keras.layers.Add()
        output = self.add([input_tensor, self.output])
        return tf.keras.layers.LayerNormalization(output)
        



