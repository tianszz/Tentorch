import tensorflow as tf
import tensorflow_transform as tft

def preprocessing_fn(inputs, params):
    outputs = inputs
    for feature in params['vocab_features']:
        outputs[feature] = tft.compute_and_apply_vocabulary(
            inputs[feature],
            top_k=params['top_k'],
            frequency_threshold=params['frequency_threshold'],
            vocab_filename=feature,
            num_oov_buckets=params['num_oov_buckets']
        )
    
    return outputs