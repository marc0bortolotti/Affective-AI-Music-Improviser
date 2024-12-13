from EEG.processing import extract_features, baseline_correction, calculate_baseline
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
import pickle
import os
import numpy as np
import logging


def fit_eeg_classifier(eeg_samples_baseline, eeg_samples_classes, sample_frequency, ch_names, asr=None):
        # Preprocessing (Feature extraction and Baseline correction)
        baseline = calculate_baseline(eeg_samples_baseline, sample_frequency, ch_names, asr, parse=True)
        eeg_features_list = []
        for eeg_samples in eeg_samples_classes:
            eeg_features = extract_features(eeg_samples, sample_frequency, ch_names, asr, parse=True)
            # Apply baseline correction
            eeg_features_corrected = baseline_correction(eeg_features, baseline)
            eeg_features_list.append(eeg_features_corrected)

        # Prepare the data for classification
        X = np.concatenate(eeg_features_list)
        y = np.concatenate([np.ones(eeg_features.shape[0]) * i for i, eeg_features in enumerate(eeg_features_list)])

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33)

        # Normalization
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # LDA
        lda_model = LinearDiscriminantAnalysis()
        lda_model.fit(X_train, y_train)
        y_lda_pred = lda_model.predict(X_test)
        y_lda_pred_proba = lda_model.predict_proba(X_test)
        accuracy_lda = accuracy_score(y_test, y_lda_pred)
        f1_lda = f1_score(y_test, y_lda_pred, average='macro')
        logging.info(f'LDA\tAccuracy: {accuracy_lda:.2f} F1 Score: {f1_lda:.2f}')

        # SVM
        svm_model = svm.SVC(kernel='linear')
        svm_model.fit(X_train, y_train)
        y_svm_pred = svm_model.predict(X_test)
        accuracy_svm = accuracy_score(y_test, y_svm_pred)
        f1_svm = f1_score(y_test, y_svm_pred, average='macro')
        logging.info(f'SVM\tAccuracy: {accuracy_svm:.2f} F1 Score: {f1_svm:.2f}')

        return scaler, svm_model, lda_model, baseline, accuracy_lda, f1_lda, accuracy_svm, f1_svm


def save_eeg_classifier(path, scaler, lda_model, svm_model, baseline):
        # Save the scaler 
        with open(os.path.join(path, 'scaler.pkl'), 'wb') as pickle_file:
            pickle.dump(scaler, pickle_file)
        # Save the LDA model
        with open(os.path.join(path, 'LDA_model.pkl'), 'wb') as pickle_file:
            pickle.dump(lda_model, pickle_file)
        # Save the SVM model
        with open(os.path.join(path, 'SVM_model.pkl'), 'wb') as pickle_file:
            pickle.dump(svm_model, pickle_file)
        # Save the baseline
        with open(os.path.join(path, 'baseline.pkl'), 'wb') as pickle_file:
            pickle.dump(baseline, pickle_file)


def load_eeg_classifier(path):
        lda_model = pickle.load(open(os.path.join(path, 'LDA_model.pkl'), 'rb'))
        svm_model = pickle.load(open(os.path.join(path, 'SVM_model.pkl'), 'rb'))
        scaler = pickle.load(open(os.path.join(path, 'scaler.pkl'), 'rb'))
        baseline = pickle.load(open(os.path.join(path, 'baseline.pkl'), 'rb'))
        return scaler, lda_model, svm_model, baseline


def get_eeg_prediction(eeg, sample_frequency, ch_names, classifier, scaler, baseline, asr=None):

    eeg_features = extract_features([eeg], sample_frequency, ch_names, asr)
    eeg_features_corrected = baseline_correction(eeg_features, baseline)

    # Prediction
    try:
        sample = scaler.transform(eeg_features_corrected)
        prediction = classifier.predict(sample)
        prediction = int(prediction[0])
    except:
        logging.info('Unicorn: no classifier set')
        prediction = None

    return prediction


def get_eeg_metrics(eeg_samples_classes, sample_frequency, ch_names, classifier, scaler, baseline):

    eeg_features_classes = []
    for eeg_samples in eeg_samples_classes:
        eeg_features = extract_features(eeg_samples, sample_frequency, ch_names, parse=True)
        # Apply baseline correction
        eeg_features_corrected = baseline_correction(eeg_features, baseline)
        eeg_features_classes.append(eeg_features_corrected)

    # Prepare the data for classification
    X = np.concatenate(eeg_features_classes)
    y = np.concatenate([np.ones(samples.shape[0]) * i for i, samples in enumerate(eeg_features_classes)])

    # Normalization
    X = scaler.transform(X)

    # LDA
    y_pred = classifier.predict(X)
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average='macro')

    return accuracy, f1