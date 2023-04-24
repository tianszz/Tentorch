import tensorflow as tf


class TFRecordDataParser():
    def __init__(self, path, feature_spec, buffer_size, batch_size, label=None, sample_weight_key=None):
        self.path = path
        self.feature_spec = feature_spec
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.label = label
        self.sample_weight_key = sample_weight_key
    

    def parse_examples(self, bytes_examples):
        examples = tf.io.parse_example(bytes_examples, self.feature_spec)
        if self.label is None:
            return examples
        else:
            label = examples.pop(self.label)
            if self.sample_weight_key is None:
                return examples, label
            else:
                sample_weight = examples.pop(self.sample_weight_key)
                return examples, label, sample_weight
    
    def read_data(self) -> tf.data.Dataset:

        tf_datasets = tf.data.TFRecordDataset(
            self.path,
            buffer_size=self.buffer_size,
            num_parallel_reads=tf.data.AUTOTUNE
        )
        dataset = (
            tf_datasets
            .batch(
                self.batch_size, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False
                )
            .map(
                lambda x: self.parse_examples(x),
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=False
                )
            )
        return dataset