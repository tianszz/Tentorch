import tensorflow as tf


class TFRecordDataParser():
    def __init__(self, path, feature_spec, buffer_size):
        self.path = path
        self.feature_spec = feature_spec
        self.buffer_size = buffer_size

    def parse_examples(self, bytes_examples, label_key, sample_key):
        examples = tf.io.parse_example(bytes_examples, self.feature_spec)
        if label_key is None:
            return examples
        else:
            label = examples.pop(label_key)
            if sample_key is None:
                return examples, label
            else:
                sample_weight = examples.pop(sample_key)
                return examples, label, sample_weight
    
    def read_data(self):
        data_files = tf.data.Dataset.list_files(self.path)

        tf_datasets = tf.data.TFRecordDataset(
            data_files,
            compression_types="gz",
            buffer_size=self.buffer_size,
            num_parallel_reads=tf.data.AUTOTUNE
        )

        dataset = tf_datasets.batch(self.batch_size, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False, drop_reminder=True)
        return dataset