import os
import numpy as np
from utils.loader import load_data, unicorn_eeg_channels, unicorn_fs


def load_dataset(path_dataset, labels):
    dataset = {}
    for label in labels:
        path_folder = os.path.join(path_dataset, label)
        files = [f for f in os.listdir(path_folder) if f.endswith('.csv')]
        path_file = os.path.join(path_folder, files[0])
        print(f'Loading file: {path_file}')
        eeg = load_data(path_file, header=False, fs=unicorn_fs, names = unicorn_eeg_channels) [0]
        dataset[label] = eeg
    return dataset



def generate_samples(eeg, window_size, window_overlap):
    samples = []
    step_size = int(window_size * (1 - window_overlap))
    for i in range(0, eeg.shape[0]-window_size, step_size):
        # eeg = (signal, channels)
        sample = eeg[i:i+window_size]
        samples.append(sample)

    print('Sample dimension:', samples[0].shape)
    print('Number of samples:', len(samples))
    return samples