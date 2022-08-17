import tensorflow as tf

def build_dataset(data, batch_size, arch='dm'):
    """Generates tensor dict mapping from tensor names to tensors.
    """
    def generator_fn():
        for item in data:
            doc_id, context_window, target_ids = item
            yield (doc_id, context_window), target_ids

    dataset = tf.data.Dataset.from_generator(generator_fn,
            output_signature=((
                tf.TensorSpec(shape=(), dtype=tf.int64),
                tf.TensorSpec(shape=([None]), dtype=tf.int64)),
                tf.TensorSpec(shape=([None]), dtype=tf.float32)))
    # batch the dataset
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset

