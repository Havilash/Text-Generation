import os
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras

from model import Model
from one_step import OneStep


def load_text_file(file_path):
    with open(file_path, "rb") as f:
        text = f.read().decode(encoding="utf-8")
    return text


def create_vocab(text):
    return sorted(set(text))


def create_string_lookup(vocab):
    ids_from_chars = tf.keras.layers.StringLookup(
        vocabulary=list(vocab), mask_token=None
    )

    chars_from_ids = tf.keras.layers.StringLookup(
        vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None
    )

    return ids_from_chars, chars_from_ids


def text_from_ids(ids, chars_from_ids):
    return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)


def create_dataset(text, ids_from_chars, seq_length):
    all_ids = ids_from_chars(tf.strings.unicode_split(text, "UTF-8"))
    ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
    sequences = ids_dataset.batch(seq_length + 1, drop_remainder=True)

    def split_input_target(sequence):
        input_text = sequence[:-1]
        target_text = sequence[1:]
        return input_text, target_text

    dataset = sequences.map(split_input_target)
    return dataset


def prepare_dataset(dataset, batch_size, buffer_size):
    dataset = (
        dataset.shuffle(buffer_size)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    return dataset


def main():
    path_to_file = tf.keras.utils.get_file(
        "shakespeare.txt",
        "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt",
    )

    text = load_text_file(path_to_file)
    vocab = create_vocab(text)
    ids_from_chars, chars_from_ids = create_string_lookup(vocab)

    seq_length = 100
    dataset = create_dataset(text, ids_from_chars, seq_length)

    batch_size = 64
    buffer_size = 10000
    dataset = prepare_dataset(dataset, batch_size, buffer_size)

    vocab_size = len(ids_from_chars.get_vocabulary())
    embedding_dim = 256
    rnn_units = 1024
    '''model = Model(vocab_size, embedding_dim, rnn_units)

    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer="adam", loss=loss)

    checkpoint_dir = "./training_checkpoints"
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)

    if latest_checkpoint is not None:
        model.load_weights(latest_checkpoint)

    checkpoint_dir = "./training_checkpoints"
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix, save_weights_only=True
    )

    EPOCHS = 1
    history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

    one_step_model = OneStep(model, chars_from_ids, ids_from_chars)

    tf.saved_model.save(one_step_model, 'one_step')'''
    one_step_model = tf.saved_model.load('one_step')

    start = time.time()
    states = None
    next_char = tf.constant(['ROMEO:'])
    result = [next_char]

    for n in range(1000):
        next_char, states = one_step_model.generate_one_step(next_char, states=states)
        result.append(next_char)

    result = tf.strings.join(result)
    end = time.time()
    print(result[0].numpy().decode('utf-8'), '\n\n' + '_'*80)
    print('\nRun time:', end - start)


if __name__ == "__main__":
    main()
