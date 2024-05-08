import numpy as np
import mne
import os
from utils.loader import unicorn_fs, load_dataset, generate_samples
from utils.feature_extraction import extract_features, calculate_baseline, baseline_correction
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from utils.validation import plot_confusion_matrix, plot_points_scatter, plot_cross_validated_confusion_matrix
from sklearn.metrics import f1_score, accuracy_score
from sklearn import svm
import pickle
from sklearn.model_selection import StratifiedKFold
import datetime
from sklearn.model_selection import cross_val_score

mne.set_log_level(verbose='ERROR', return_old_level=False, add_frames=None)


# Global variables

dataset_1 = {'dataset_name': 'dataset_1', 'dataset_type': ['listening', 'playing'], 'labels': ['relax', 'excited']}
dataset_2 = {'dataset_name': 'dataset_2', 'dataset_type': ['listening', 'playing'], 'labels': ['relax', 'excited']}
dataset_3 = {'dataset_name': 'dataset_3', 'dataset_type': ['listening', 'playing'], 'labels': ['bpm_60', 'bpm_90', 'bpm_120', 'bpm_150']}
dataset_4 = {'dataset_name': 'dataset_4', 'dataset_type': ['listening', 'playing'], 'labels': ['bpm_60', 'bpm_90', 'bpm_120', 'bpm_150']}
dataset_5 = {'dataset_name': 'dataset_5', 'dataset_type': ['listening', 'playing'], 'labels': ['relax', 'excited']}
dataset_6_1 = {'dataset_name': 'dataset_6-1', 'dataset_type': ['listening', 'playing'], 'labels': ['relax', 'excited']}
dataset_6_2 = {'dataset_name': 'dataset_6-2', 'dataset_type': ['listening', 'playing'], 'labels': ['bpm_60', 'bpm_120']}
dataset_7_1 = {'dataset_name': 'dataset_7-1', 'dataset_type': ['listening', 'playing'], 'labels': ['relax', 'excited']}
dataset_7_2 = {'dataset_name': 'dataset_7-2', 'dataset_type': ['listening', 'playing'], 'labels': ['bpm_60', 'bpm_120']}
dataset_8_1 = {'dataset_name': 'dataset_8-1', 'dataset_type': ['listening', 'playing'], 'labels': ['relax', 'excited']}
dataset_8_2 = {'dataset_name': 'dataset_8-2', 'dataset_type': ['listening', 'playing'], 'labels': ['bpm_60', 'bpm_120']}
dataset_9_1 = {'dataset_name': 'dataset_9-1', 'dataset_type': ['listening', 'playing'], 'labels': ['relax', 'excited']}
dataset_9_2 = {'dataset_name': 'dataset_9-2', 'dataset_type': ['listening', 'playing'], 'labels': ['bpm_60', 'bpm_120']}
dataset_prova = {'dataset_name': 'dataset_prova', 'dataset_type': ['playing'], 'labels': ['relax', 'excited']}
dataset_prova_2 = {'dataset_name': 'dataset_prova_2', 'dataset_type': ['listening', 'playing'], 'labels': ['baseline_1', 'baseline_2', 'baseline_3', 'baseline_4']}

def classification(dataset):

    dataset_name = dataset['dataset_name']
    dataset_type = dataset['dataset_type']
    labels = dataset['labels']

    save_data = True

    window_duration = 4 # seconds
    window_size = window_duration * unicorn_fs # samples
    window_overlap = 0.875 # percentage


    for data_type in dataset_type:
        print(f'\nProcessing {dataset_name} - {data_type} dataset\n')

        dataset_path_name = dataset_name
        if '-' in dataset_name:
            dataset_path_name = dataset_name.split('-')[0]

        path_dataset = os.path.join('BCI/data/dataset', dataset_path_name, data_type)
        # Read dataset
        dataset = load_dataset(path_dataset, labels)

        # Calculate baseline
        print('\n----baseline----')
        eeg_raw_baseline = dataset['baseline']
        eeg_samples_baseline = generate_samples(eeg_raw_baseline, window_size, window_overlap)
        baseline = calculate_baseline(eeg_samples_baseline)

        # Extract features and apply baseline correction
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
        print(f'\n\nX shape: {X.shape}, y shape: {y.shape}\n\
X_train shape: {X_train.shape}, y_train shape: {y_train.shape}\n\
X_test shape: {X_test.shape}, y_test shape: {y_test.shape}')
        
        X_shuffled = np.vstack([X_train, X_test])
        y_shuffled = np.concatenate([y_train, y_test])

        # Models
        scaler = StandardScaler()
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        lda_model = LinearDiscriminantAnalysis()
        svm_model = svm.SVC()

        # # Load pretrained models
        # lda_model = pickle.load(open('BCI/results/dataset_7-1/playing/LDA_model.pkl', 'rb'))
        # svm_model = pickle.load(open('BCI/results/dataset_7-1/playing/SVM_model.pkl', 'rb'))

        # Normalization
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test) 

        print()

        # LDA
        lda_model.fit(X_train, y_train)
        y_lda_pred = lda_model.predict(X_test)
        y_lda_pred_proba = lda_model.predict_proba(X_test)
        # Calculate accuracy
        accuracy_lda = accuracy_score(y_test, y_lda_pred)
        f1_lda = f1_score(y_test, y_lda_pred, average='macro')
        print(f'LDA\tAccuracy: {accuracy_lda:.2f} F1 Score: {f1_lda:.2f}')
        # Cross-validation
        lda_cross_scores = cross_val_score(lda_model, X_shuffled, y_shuffled, cv=cv)
        lda_cross_f1 = cross_val_score(lda_model, X_shuffled, y_shuffled, cv=cv, scoring='f1_macro', error_score=0)
        print(f'LDA\tCross-validated accuracy: {np.mean(lda_cross_scores):.2f} (+/- {np.std(lda_cross_scores):.2f}) F1-Score: {np.mean(lda_cross_f1):.2f}')



        # SVM
        svm_model.fit(X_train, y_train)
        y_svm_pred = svm_model.predict(X_test)
        # Calculate accuracy
        accuracy_svm = accuracy_score(y_test, y_svm_pred)
        f1_svm = f1_score(y_test, y_svm_pred, average='macro')
        print(f'SVM\tAccuracy: {accuracy_svm:.2f} F1 Score: {f1_svm:.2f}')
        # Cross-validation
        svm_cross_scores = cross_val_score(svm_model, X_shuffled, y_shuffled, cv=cv)
        svm_cross_f1 = cross_val_score(svm_model, X_shuffled, y_shuffled, cv=cv, scoring='f1_macro', error_score=0)
        print(f'SVM\tCross-validated accuracy: {np.mean(svm_cross_scores):.2f} (+/- {np.std(svm_cross_scores):.2f}) F1-Score: {np.mean(svm_cross_f1):.2f}')


        print()

        if save_data:
            path_save = 'BCI/results/' + dataset_name + '/' + data_type

            if not os.path.exists(path_save):
                os.makedirs(path_save)

            # Plot confusion matrix and scatter plot
            plot_confusion_matrix(y_test, y_lda_pred, classes=labels, normalize=True, save_data=save_data, save_path=path_save, classifier='LDA')
            X_lda = lda_model.transform(X_shuffled) # return (samples, components)
            plot_points_scatter(X_lda, y_shuffled, labels, save_data=save_data, save_path=path_save, classifier='LDA')
            plot_cross_validated_confusion_matrix(X_shuffled, y_shuffled, lda_model, cv=cv, classes=labels, normalize=True, classifier='LDA', save_data=save_data, save_path=path_save)

            plot_confusion_matrix(y_test, y_svm_pred, classes=labels, normalize=True, save_data=save_data, save_path=path_save, classifier='SVM')
            X_svm = svm_model.decision_function(X_shuffled)
            plot_points_scatter(X_svm, y_shuffled, labels, save_data=save_data, save_path=path_save, classifier='SVM')
            plot_cross_validated_confusion_matrix(X_shuffled, y_shuffled, svm_model, cv=cv, classes=labels, normalize=True, classifier='SVM', save_data=save_data, save_path=path_save)

            # Save parameters and results
            with open(os.path.join(path_save, 'param_and_results.txt'), 'w') as f:
                f.write(f'Date: {datetime.datetime.now()}\n')
                f.write('\nPARAMETERS\n')
                f.write(f'Dataset: {dataset_name}\n')
                f.write(f'Dataset Type: {data_type}\n')
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
                f.write(f'LDA\tCross-validated accuracy: {np.mean(lda_cross_scores):.2f} (+/- {np.std(lda_cross_scores):.2f}) F1-Score: {np.mean(lda_cross_f1):.2f}\n')
                f.write(f'SVM\tAccuracy: {accuracy_svm:.2f} F1 Score: {f1_svm:.2f}\n')
                f.write(f'SVM\tCross-validated accuracy: {np.mean(svm_cross_scores):.2f} (+/- {np.std(svm_cross_scores):.2f}) F1-Score: {np.mean(svm_cross_f1):.2f}\n')
                f.write(f'\n')

            # Save the LDA model
            outfile = os.path.join(path_save, 'LDA_model.pkl')
            with open(outfile, 'wb') as pickle_file:
                pickle.dump(lda_model, pickle_file)

            # Save the SVM model
            outfile = os.path.join(path_save, 'SVM_model.pkl')
            with open(outfile, 'wb') as pickle_file:
                pickle.dump(svm_model, pickle_file)

if __name__ == '__main__':
    classification(dataset_prova_2)