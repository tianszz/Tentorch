from src.tf_ops.layers.dense import DenseLayers, ResBlock
import tensorflow as tf
import numpy as np


class TestDenseLayers(tf.test.TestCase):
    def setUp(self):
        self.data = X = np.random.random(((10, 4)))
    def test_dense_layers(self):
        self.d_layer = DenseLayers(
            layers=[4, 4],
            activation='relu',
            batch_norm=False,
            layer_norm=False,
            target_dim=1
        )
        self.assertEqual(self.d_layer(self.data).shape, [10, 1])

class TestResBlock(tf.test.TestCase):
    def setUp(self):
        self.data = X = np.random.random(((10, 4)))
    
    def test_res_block(self):
        self.d_layer = ResBlock(
            units_1=4,
            units_2=4,
            drop_rate=0.1
        )
        self.assertEqual(self.d_layer(self.data).shape, [10, 4])