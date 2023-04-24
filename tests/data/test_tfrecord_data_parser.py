import unittest
from src.utils.feature_util import _bytes_feature, _float_feature
import tensorflow as tf
from src.data.tfrecords_data import TFRecordDataParser
import tempfile
import shutil
from tempfile import mkdtemp

class TestTFRecordDataParser(tf.test.TestCase):
    def serialize_example(self):
        """
        Creates a tf.Example message ready to be written to a file.
        """

        # Create a dictionary mapping the feature name to the tf.Example-compatible
        # data type.
        feature = {
            'feature0': _float_feature([0.0, 0.1, 0.2]),
            'feature1': _bytes_feature([b'a',b'', b'', b'c', b'd']),
            'feature2': _bytes_feature([b'b']),
        }
        

        # Create a Features message using tf.train.Example.
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()
    
    
    def setUp(self):
        self.temp_dir = mkdtemp()
        self.feature_spec = {
            "feature0": tf.io.VarLenFeature(tf.float32),
            "feature1": tf.io.VarLenFeature(tf.string),
            "feature2": tf.io.VarLenFeature(tf.string)

        }

        with tf.io.TFRecordWriter(self.temp_dir + "/test.tfrecord") as writer:
            example = self.serialize_example()
            writer.write(example)
        
        
        self.tfrdp = TFRecordDataParser(
            path=self.temp_dir + "/test.tfrecord",
            feature_spec=self.feature_spec,
            buffer_size=100,
            batch_size=32,
        )

    def test_read_data(self):
        ret = self.tfrdp.read_data()

        for data in ret:
            self.assertAllEqual(tf.sparse.to_dense(data['feature0']), tf.constant([[0., 0.1, 0.2]]))

    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        print("deleted")
