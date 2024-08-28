import pandas as pd
import numpy as np
import os

synth_eeg_channels = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3",
                     "C3", "Cz", "C4", "T4", "T5", "P3", "Pz", "P4"]

unicorn_eeg_channels = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
unicorn_fs = 250

enophone_eeg_channels = ["A1", "A2", "C1", "C2"]
enophone_fs = 250


def load_data(path, header, fs, names, skiprows=5):
    if header:
        df = pd.read_csv(path,
                         names=names + ["trigger", "id", "target", "nontarget", "trial", "islast"],
                         skiprows=skiprows * fs)
        trigger = np.array(df.id)
    else:
        df = pd.read_csv(path, names=names + ["STI"], skiprows=skiprows * fs)
        trigger = np.array(df.STI)
    eeg = df.iloc[:, 0:len(unicorn_eeg_channels)].to_numpy()
    return eeg, trigger, df


def load_dataset(path_dataset, labels):
    dataset = {}
    labels_copy = labels.copy()
    labels_copy.append('baseline')
    for label in labels_copy:
        path_folder = os.path.join(path_dataset, label)
        files = [f for f in os.listdir(path_folder) if f.endswith('.csv')]
        eeg_list = []
        for file in files:
            path_file = os.path.join(path_folder, file)
            print(f'Loading file: {path_file}')
            eeg = load_data(path_file, header=False, fs=unicorn_fs, names = unicorn_eeg_channels) [0]
            eeg_list.append(eeg)
        eeg = np.concatenate(eeg_list, axis=0)
        dataset[label] = eeg
    return dataset