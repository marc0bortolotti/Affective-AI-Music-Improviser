import  numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score
from eeg.processing import calculate_baseline, extract_features, baseline_correction


def get_eeg_classifiers(self, eeg_samples_baseline, eeg_samples_classes):
    # Preprocessing (Feature extraction and Baseline correction)
    baseline = calculate_baseline(eeg_samples_baseline)
    eeg_features_list = []
    for eeg_samples in eeg_samples_classes:
        eeg_features = extract_features(eeg_samples, fs=self.sample_frequency, chs=self.eeg_channels)
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
    svm_model = svm.SVC(kernel = 'linear')
    svm_model.fit(X_train, y_train) 
    y_svm_pred = svm_model.predict(X_test)
    accuracy_svm = accuracy_score(y_test, y_svm_pred)
    f1_svm = f1_score(y_test, y_svm_pred, average='macro')
    logging.info(f'SVM\tAccuracy: {accuracy_svm:.2f} F1 Score: {f1_svm:.2f}')

    return scaler, svm_model, lda_model, baseline




 