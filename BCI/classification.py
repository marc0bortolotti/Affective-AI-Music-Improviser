import numpy as np
import mne
import os
from utils.loader import unicorn_fs, load_dataset, generate_samples
from utils.feature_extraction import extract_features, calculate_baseline, baseline_correction
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from utils.validation import plot_confusion_matrix, plot_points_scatter
from sklearn.metrics import f1_score, accuracy_score
from sklearn import svm
import pickle
import datetime

mne.set_log_level(verbose='ERROR', return_old_level=False, add_frames=None)


# Global variables

dataset_name = 'dataset_4'
# dataset_type = 'listening'
dataset_type = 'listening'
path_dataset = os.path.join('BCI/data/dataset', dataset_name, dataset_type)
# labels = ['relax', 'excited']
labels = ['bpm_60', 'bpm_90', 'bpm_120', 'bpm_150']

save_data = True
path_save = 'BCI/results/' + dataset_name + '_' + dataset_type

window_duration = 4 # seconds
window_size = window_duration * unicorn_fs # samples
window_overlap = 0.875 # percentage



# Read dataset
print()
dataset = load_dataset(path_dataset, labels)

# Calculate baseline
print('\n----baseline----')
eeg_raw_baseline = dataset['baseline']
eeg_samples_baseline = generate_samples(eeg_raw_baseline, window_size, window_overlap)
baseline = calculate_baseline(eeg_samples_baseline)

# Extract features and apply baseline correction
eeg_raw_list = [eeg_raw_baseline]
eeg_features_list = []
for label in labels:
    print(f'\n\n----{label}----')
    # Generate samples
    eeg_raw = dataset[label]
    eeg_samples = generate_samples(eeg_raw, window_size, window_overlap)
    # Extract features
    eeg_features = extract_features(eeg_samples)
    # Apply baseline correction
    eeg_features_corrected = baseline_correction(eeg_features, baseline)
    eeg_features_list.append(eeg_features_corrected)

# Prepare the data for classification
X = np.concatenate(eeg_features_list)
y = np.concatenate([np.ones(eeg_features.shape[0]) * i for i, eeg_features in enumerate(eeg_features_list)])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33)
print(f'\n\nX_train shape: {X_train.shape}, y_train shape: {y_train.shape}, X_test shape: {X_test.shape}, y_test shape: {y_test.shape}')

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test) 

print()

# LDA
lda_model = LinearDiscriminantAnalysis()
lda_model.fit(X_train, y_train)
y_lda_pred = lda_model.predict(X_test)
y_lda_pred_proba = lda_model.predict_proba(X_test)
# Calculate accuracy
accuracy_lda = accuracy_score(y_test, y_lda_pred)
f1_lda = f1_score(y_test, y_lda_pred, average='micro')
print(f'LDA\tAccuracy: {accuracy_lda:.2f} F1 Score: {f1_lda:.2f}')

# SVM
svm_model = svm.SVC()
svm_model.fit(X_train, y_train)
y_svm_pred = svm_model.predict(X_test)
# Calculate accuracy
accuracy_svm = accuracy_score(y_test, y_svm_pred)
f1_svm = f1_score(y_test, y_svm_pred, average='micro')
print(f'SVM\tAccuracy: {accuracy_svm:.2f} F1 Score: {f1_svm:.2f}')

print()

if save_data:
    if not os.path.exists(path_save):
        os.makedirs(path_save)

    # Plot confusion matrix and scatter plot
    X = np.vstack([X_train, X_test])
    y = np.concatenate([y_train, y_test])


    plot_confusion_matrix(y_test, y_lda_pred, classes=labels, normalize=True, save_data=save_data, save_path=path_save, classifier='LDA')
    X_lda = lda_model.transform(X) # return (samples, components)
    plot_points_scatter(X_lda, y, labels, save_data=save_data, save_path=path_save, classifier='LDA')

    plot_confusion_matrix(y_test, y_svm_pred, classes=labels, normalize=True, save_data=save_data, save_path=path_save, classifier='SVM')
    X_svm = svm_model.decision_function(X)
    plot_points_scatter(X_svm, y, labels, save_data=save_data, save_path=path_save, classifier='SVM')

    # Save parameters and results
    with open(os.path.join(path_save, 'param_and_results.txt'), 'w') as f:
        f.write(f'Date: {datetime.datetime.now()}\n')
        f.write('\nPARAMETERS\n')
        f.write(f'Dataset: {dataset_name}\n')
        f.write(f'Dataset Type: {dataset_type}\n')
        f.write(f'Labels: {labels}\n')
        f.write(f'Window Duration: {window_duration} seconds\n')
        f.write(f'Window Size: {window_size} samples\n')
        f.write(f'Window Overlap: {window_overlap:.2f}\n')
        f.write(f'\nRESULTS\n')
        f.write(f'X_train shape: {X_train.shape}\n')
        f.write(f'X_test shape: {X_test.shape}\n')
        f.write(f'y_train shape: {y_train.shape}\n')
        f.write(f'y_test shape: {y_test.shape}\n')
        f.write(f'LDA\tAccuracy: {accuracy_lda:.2f} F1 Score: {f1_lda:.2f}\n')
        f.write(f'SVM\tAccuracy: {accuracy_svm:.2f} F1 Score: {f1_svm:.2f}\n')
        f.write(f'\n')

    # Save the LDA model
    outfile = os.path.join(path_save, 'LDA_model.pkl')
    with open(outfile, 'wb') as pickle_file:
        pickle.dump(lda_model, pickle_file)

    # Save the SVM model
    outfile = os.path.join(path_save, 'SVM_model.pkl')
    with open(outfile, 'wb') as pickle_file:
        pickle.dump(svm_model, pickle_file)
