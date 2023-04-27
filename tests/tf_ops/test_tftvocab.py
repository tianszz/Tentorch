from src.tf_ops.preprocessing import preprocessing_fn
from src.tf_ops.tft_tokenizer import TFTVocab
import tensorflow as tf
from tempfile import mkdtemp


class TestTFTVocab(tf.test.TestCase):
    def setUp(self):
        self.temp_dir = mkdtemp()
        self.params = {
            
        }
        

    def test_tftvocab(self):
        TFTVocab(
            preprocessing_fn=preprocessing_fn,
            filepattern_train=self.train,
            filepattern_eval=self.eval,
            output_uri=self.temp_dir,
            params=self.params,
            direct_runner=True
        ).run()

    def tearDown(self):
        pass
