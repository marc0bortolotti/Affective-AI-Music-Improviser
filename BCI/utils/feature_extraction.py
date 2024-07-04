import numpy as np
from BCI.utils.loader import convert_to_mne, unicorn_eeg_channels, unicorn_fs, enophone_eeg_channels, enophone_fs


def extract_features(eeg_samples, fs = unicorn_fs, chs = unicorn_eeg_channels):
    eeg_features = []

    for i, sample in enumerate(eeg_samples):
        print(f'Processing sample: {i+1}/{len(eeg_samples)}', end='\r')
        filtered_sample = apply_filters(sample, fs, chs)
        log_var_sample = log_var_transform(filtered_sample)
        eeg_features.append(log_var_sample)


    eeg_features = np.array(eeg_features)
    print(f'\nFeatures dimension: {eeg_features.shape}')

    return eeg_features


def calculate_baseline(eeg_samples, fs = unicorn_fs, chs = unicorn_eeg_channels):
    eeg_features = extract_features(eeg_samples, fs, chs)
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

