import pandas as pd
import numpy as np
from mne.io import RawArray
from mne import create_info
from mne.channels import make_standard_montage
import os


unicorn_eeg_channels = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
unicorn_fs = 250


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


def convert_to_mne(eeg, trigger, fs, chs, rescale=1e6, recompute=False):
    """
    Convert the data to MNE format
    :param eeg: numpy array of shape (n_samples, n_channels)
    :param trigger: numpy array of shape (n_samples, )
    :param fs: sampling frequency
    :param chs: list of channels names
    :param rescale: rescaling factor to the right units
    :param recompute: whether if changing trigger numerical values or not to avoid Event "0"
    :return: MNE RawArray object
    """

    this_rec = RawArray(eeg.T / rescale, create_info(chs, fs, ch_types='eeg'))

    # Get event indexes where value is not 0, i.e. -1 or 1
    pos = np.nonzero(trigger)[0]

    # Filter 0 values from the trigger array
    y = trigger[trigger != 0]

    # Create the stimuli channel
    stim_data = np.ones((1, this_rec.n_times)) if recompute else np.zeros((1, this_rec.n_times))

    # MNE works with absolute values of labels so -1 and +1 would result in only one kind of event
    # that's why we add 1 and obtain 1 and 2 as label values
    stim_data[0, pos] = (y + 1) if recompute else y

    stim_raw = RawArray(stim_data, create_info(['STI'], this_rec.info['sfreq'], ch_types=['stim']))

    # adding the stimuli channel (as a Raw object) to our EEG Raw object
    this_rec.add_channels([stim_raw])

    # Set the standard 10-20 montage
    montage = make_standard_montage('standard_1020')
    this_rec.set_montage(montage)
    return this_rec


def load_dataset(path_dataset, labels):
    dataset = {}
    labels_copy = labels.copy()
    labels_copy.append('baseline')
    for label in labels_copy:
        path_folder = os.path.join(path_dataset, label)
        path_folder = path_dataset+'/'+label
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