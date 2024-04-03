import numpy as np
from utils.loader import convert_to_mne, unicorn_eeg_channels, unicorn_fs

def extract_features(baseline_samples, eeg_samples, baseline = None):

    if baseline is None:
        baseline_samples = extract_bands(baseline_samples)
        baseline_features = apply_log_variance(baseline_samples)
        baseline = np.mean(baseline_features, axis=0)
        
    eeg_samples = extract_bands(eeg_samples)
    eeg_features = apply_log_variance(eeg_samples)
    eeg_features_cleared = baseline_correction(eeg_features, baseline)

    return np.array(eeg_features_cleared), np.array(baseline)



def extract_bands(eeg_samples):
    filtered_eeg_samples = []
    for i in range(len(eeg_samples)):
        print(f'Extracting bands from sample: {i+1}/{len(eeg_samples)}', end='\r')
        sample = eeg_samples[i]
        filtered_eeg_samples.append(apply_filters(sample))
    print()
    return np.array(filtered_eeg_samples)


def apply_filters(eeg):

    # extract bands from the EEG samples: theta, alpha, low beta, high beta, gamma
    # apply notch filters to remove 50Hz and 60Hz noise

    trigger = np.zeros(len(eeg))
    raw_data = convert_to_mne(eeg, trigger, fs=unicorn_fs, chs=unicorn_eeg_channels, recompute=False) 
    raw_data.notch_filter(50) 
    raw_data.notch_filter(60) 
    
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

    filtered_sample = [filtered_theta.get_data()[0:8,:]*1e6, # without the STI channel and rescaled to microvolts
                       filtered_alpha.get_data()[0:8,:]*1e6, 
                       filtered_low_beta.get_data()[0:8,:]*1e6, 
                       filtered_high_beta.get_data()[0:8,:]*1e6, 
                       filtered_gamma.get_data()[0:8,:]*1e6]
    
    return np.array(filtered_sample)


def log_var_transform(sample):
    # reshape to (bands, channels, signal) 
    sample = np.transpose(sample, (0, 1, 2))
    # calculate the log-variance
    log_var_sample = np.log(np.var(sample, axis=2))
    # flatten to one dimension
    log_var_sample = log_var_sample.flatten()
    return np.array(log_var_sample)


def apply_log_variance(eeg_samples):
    log_var_samples = []
    for i in range(len(eeg_samples)):
        sample = eeg_samples[i]
        log_var_sample = log_var_transform(sample)
        log_var_samples.append(log_var_sample)
    return np.array(log_var_samples)


def baseline_correction(samples, baseline):
    corrected_samples = samples - baseline
    return np.array(corrected_samples)

