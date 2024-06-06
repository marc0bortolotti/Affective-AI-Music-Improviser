import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from pretty_midi_tokenization import create_sequences, midi_to_notes, KEY_ORDER, extract_bars_from_notes
import os

DIRECTORY_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(DIRECTORY_PATH, 'dataset')
CHECKPOINTS_PATH = os.path.join(DIRECTORY_PATH, 'training_checkpoints')

KEY_ORDER = KEY_ORDER
BATCH_SIZE = 64
SEQ_LENGTH = 25
VOCAB_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 0.0001
BPM = 100
BEATS_PER_BAR = 4


def build_model(input_shape, learning_rate):

    def mse_with_positive_pressure(y_true: tf.Tensor, y_pred: tf.Tensor):
        mse = (y_true - y_pred) ** 2
        positive_pressure = 10 * tf.maximum(-y_pred, 0.0)
        return tf.reduce_mean(mse + positive_pressure)
    

    inputs = tf.keras.Input(input_shape)
    x = tf.keras.layers.LSTM(128)(inputs)

    outputs = {
    'pitch': tf.keras.layers.Dense(128, name='pitch')(x),
    'step': tf.keras.layers.Dense(1, name='step')(x),
    'duration': tf.keras.layers.Dense(1, name='duration')(x),
    }

    model = tf.keras.Model(inputs, outputs)

    loss = {
        'pitch': tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True),
        'step': mse_with_positive_pressure,
        'duration': mse_with_positive_pressure,
    }

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(loss=loss,
                  loss_weights={'pitch': 0.05,
                                'step': 1.0,
                                'duration':1.0},
                  optimizer=optimizer)

    model.summary()

    return model




def generate_samples_from_recordings(seq_length, vocab_size, batch_size):
    '''
    Assumptions:
    Sequences described as input_#.mid and output_#.mid in the corresponding folders
    '''

    input_filenames = glob.glob(os.path.join(DATASET_PATH, 'input/*.MID'))
    print('Number of input files:', len(input_filenames))

    output_filenames = glob.glob(os.path.join(DATASET_PATH, 'output/*.MID'))
    print('Number of output files:', len(output_filenames))

    for i, (in_file, out_file) in enumerate(zip(input_filenames, output_filenames)):

        in_file_name = os.path.basename(in_file)
        out_file_name = os.path.basename(out_file)

        print(f'{i + 1}: {in_file_name} -> {out_file_name}')
        in_notes = midi_to_notes(in_file)
        in_bars = extract_bars_from_notes(in_notes)
        print(f'Number of notes in input: {len(in_notes)}')
        print(f'Number of bars in input: {len(in_bars)}')

        out_notes = midi_to_notes(out_file)
        out_bars = extract_bars_from_notes(out_notes)
        print(f'Number of notes in output: {len(out_notes)}')
        print(f'Number of bars in output: {len(out_bars)}')

        # for i, bar in enumerate(out_bars):
        #     print(f'BAR_{i}')
        #     print(bar)
        #     print()

if __name__ == '__main__':
    
    generate_samples_from_recordings(SEQ_LENGTH, VOCAB_SIZE, BATCH_SIZE)


#     train_notes = np.stack([all_notes[key] for key in KEY_ORDER], axis=1)

#     notes_ds = tf.data.Dataset.from_tensor_slices(train_notes)
#     notes_ds.element_spec


#     seq_ds = create_sequences(notes_ds, seq_length, vocab_size)
#     seq_ds.element_spec

#     for seq, target in seq_ds.take(1):
#         print('sequence shape:', seq.shape)
#         print('sequence elements (first 10):', seq[0: 10])
#         print()
#         print('target:', target)


#     buffer_size = n_notes - seq_length  # the number of items in the dataset
#     train_ds = (seq_ds
#                 .shuffle(buffer_size)
#                 .batch(batch_size, drop_remainder=True)
#                 .cache()
#                 .prefetch(tf.data.experimental.AUTOTUNE))
    
#     return train_ds



# def train_midi_gererator(seq_length = SEQ_LENGTH, 
#                          vocab_size = VOCAB_SIZE, 
#                          batch_size = BATCH_SIZE,
#                          learning_rate = LEARNING_RATE, 
#                          epochs = EPOCHS):
    
#     train_dataset = generate_samples_from_recordings(seq_length, vocab_size, batch_size)
    
#     callbacks = [
#         tf.keras.callbacks.ModelCheckpoint(
#             filepath = os.path.join(CHECKPOINTS_PATH, 'ckpt_{epoch}'),
#             save_weights_only=True),
#         tf.keras.callbacks.EarlyStopping(
#             monitor='loss',
#             patience=5,
#             verbose=1,
#             restore_best_weights=True),
#     ]

#     model = build_model()
    
#     history = model.fit(
#         train_dataset,
#         epochs=epochs,
#         callbacks=callbacks,
#     )

#     plt.plot(history.epoch, history.history['loss'], label='total loss')
#     plt.show()