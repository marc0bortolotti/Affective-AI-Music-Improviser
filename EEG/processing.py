from mne.io import RawArray
from mne import create_info
from mne.channels import make_standard_montage
import numpy as np


def generate_samples(eeg, window_size, window_overlap, parse=False):
    samples = []
    step_size = int(window_size * (1 - window_overlap))
    for i in range(0, eeg.shape[0]-window_size, step_size):
        # eeg = (signal, channels)
        sample = eeg[i:i+window_size]
        samples.append(sample)

    if parse:
        print('EEG dimension:', eeg.shape)
        print('Sample dimension:', samples[0].shape)
        print('Number of samples:', len(samples))
    return samples


def convert_to_mne(eeg, trigger, fs, chs, rescale=1e6, recompute=False, transpose=True):
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

    eeg = eeg.T if transpose else eeg
    this_rec = RawArray(eeg / rescale, create_info(chs, fs, ch_types='eeg'))

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


def extract_features(eeg_samples, fs, ch_names, parse=False):
    eeg_features = []

    for i, sample in enumerate(eeg_samples):
        if parse:
            print(f'Processing sample: {i+1}/{len(eeg_samples)}', end='\r')
        filtered_sample = apply_filters(sample, fs, ch_names)
        log_var_sample = log_var_transform(filtered_sample)
        eeg_features.append(log_var_sample)


    eeg_features = np.array(eeg_features)
    if parse:
        print(f'\nFeatures dimension: {eeg_features.shape}')

    return eeg_features


def calculate_baseline(eeg_samples, fs, ch_names, parse=False):
    eeg_features = extract_features(eeg_samples, fs, ch_names, parse=parse)
    baseline = np.mean(eeg_features, axis=0)
    return np.array(baseline)


def apply_filters(eeg, fs, chs):

    # extract bands from the EEG samples: theta, alpha, low beta, high beta, gamma

    trigger = np.zeros(len(eeg))
    raw_data = convert_to_mne(eeg, trigger, fs=fs, chs=chs, recompute=False) 
    
    filtered_theta = raw_data.copy() 
    filtered_theta.filter(4, 7)

    filtered_alpha = raw_data.copy()
    filtered_alpha.filter(8, 13)

    filtered_low_beta = raw_data.copy()
    filtered_low_beta.filter(14, 17)

    filtered_high_beta = raw_data.copy()
    filtered_high_beta.filter(22, 29)

    filtered_gamma = raw_data.copy()
    filtered_gamma.filter(30, 47)

    filtered_sample = [filtered_theta.get_data()[0:-1,:]*1e6, # without the STI channel and rescaled to microvolts
                       filtered_alpha.get_data()[0:-1,:]*1e6, 
                       filtered_low_beta.get_data()[0:-1,:]*1e6, 
                       filtered_high_beta.get_data()[0:-1,:]*1e6, 
                       filtered_gamma.get_data()[0:-1,:]*1e6]
    
    return np.array(filtered_sample)


def log_var_transform(sample):
    # (bands, channel, samples)
    log_var_sample = np.log(np.var(sample, axis=2))
    # flatten (bands, channel) to one dimension
    log_var_sample = log_var_sample.flatten()
    return np.array(log_var_sample)


def baseline_correction(features, baseline):
    corrected_features = features - baseline
    return np.array(corrected_features)