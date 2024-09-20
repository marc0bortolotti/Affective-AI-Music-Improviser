import pandas as pd
import numpy as np
import os


def load_data(path, header, fs, ch_names, skiprows=5):
    if header:
        df = pd.read_csv(path,
                         names=ch_names + ["trigger", "id", "target", "nontarget", "trial", "islast"],
                         skiprows=skiprows * fs)
        trigger = np.array(df.id)
    else:
        df = pd.read_csv(path, names=ch_names + ["STI"], skiprows=skiprows * fs)
        trigger = np.array(df.STI)
    eeg = df.iloc[:, 0:len(ch_names)].to_numpy()
    return eeg, trigger, df


def load_dataset(path_dataset, labels, fs, ch_names):
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
            eeg = load_data(path_file, header=False, fs=fs, ch_names=ch_names) [0]
            eeg_list.append(eeg)
        eeg = np.concatenate(eeg_list, axis=0)
        dataset[label] = eeg
    return dataset