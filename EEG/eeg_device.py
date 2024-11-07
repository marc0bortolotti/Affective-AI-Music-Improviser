import re
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import numpy as np
import time
import logging
from EEG.processing import extract_features, baseline_correction, calculate_baseline
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from pylsl import resolve_stream, StreamInlet
import pylsl
import pickle
import os
from brainflow.data_filter import DataFilter


def retrieve_board_id(device_name):
    if re.search(r'UN-\d{4}.\d{2}.\d{2}', device_name):
        return BoardIds.UNICORN_BOARD
    elif re.search(r'(?i)enophone', device_name):
        return BoardIds.ENOPHONE_BOARD
    elif re.search(r'(?i)ANT.NEURO.225', device_name):
        return BoardIds.ANT_NEURO_EE_225_BOARD
    elif re.search(r'(?i)ANT.NEURO.411', device_name):
        return BoardIds.ANT_NEURO_EE_411_BOARD
    else:
        return BoardIds.SYNTHETIC_BOARD


markers_dict = {
    'RS': 1,  #resting state
    'PTR': 2, #pretraining  relaxed
    'PTE': 3, #pretraining  excited
    'WN': 4,  #white noise
    'P': 5,   #prediction
    'WD': 6,  #window
    'VR': 7,  #validation  relaxed
    'VE': 8,  #validation  excited
    'V': 97,  #validation mode
    'T': 98,  #training mode
    'A': 99,  #application mode
}


class EEG_Device:
    def __init__(self, serial_number):

        self.recording_data = None  # recordings raw data
        self.streams = None

        self.classifier = None
        self.scaler = None
        self.baseline = None

        self.params = BrainFlowInputParams()
        self.params.serial_number = serial_number
        self.params.board_id = retrieve_board_id(self.params.serial_number)
        self.ch_names = BoardShim.get_eeg_names(self.params.board_id)
        self.ch_names = self.ch_names if len(self.ch_names) <= 8 else self.ch_names[:8]
        self.board = BoardShim(self.params.board_id, self.params)
        self.sample_frequency = self.board.get_sampling_rate(self.params.board_id)
        logging.info(f"EEG Device: connected to {self.params.serial_number}")

        self.prepare_session()

    def stop_recording(self):
        self.recording_data = self.board.get_board_data()
        self.board.stop_stream()
        logging.info('EEG Device: stop recording')

    def prepare_session(self):
        try:
            self.board.prepare_session()
        except Exception as e:
            logging.error(f"EEG Device: {e}")
            self.board.release_session()

    def start_recording(self):
        if self.params.serial_number == 'LSL':
            logging.info('LSLDevice: looking for a stream')
            while not self.streams:
                self.streams = resolve_stream('name', 'Cortex EEG')
                time.sleep(1)
            logging.info("LSL stream found: {}".format(self.streams[0].name()))
            self.inlet = StreamInlet(self.streams[0], pylsl.proc_threadsafe)
        else:
            try:
                self.board.prepare_session()
            except Exception as e:
                logging.error(f"EEG Device: {e}")
                self.board.release_session()
            self.board.start_stream()
        logging.info('EEG Device: start recording')

    def insert_marker(self, marker):
        try:
            self.board.insert_marker(markers_dict[marker])
        except Exception as e:
            logging.error(f"EEG Device: Could not insert marker! {e}")

    def get_eeg_data(self, recording_time=4):
        num_samples = int(self.sample_frequency * recording_time)
        data = self.board.get_current_board_data(num_samples=num_samples)
        data = np.array(data).T
        data = data[:, 0:len(self.ch_names)]
        data = data[- num_samples:]
        return data

    def close(self):
        if self.params.serial_number == 'LSL':
            self.inlet.close_stream()
        else:
            self.recording_data = self.board.get_board_data()
            self.board.release_session()
        logging.info('EEG Device: disconnected')

    def get_classifier(self):
        return self.classifier, self.scaler, self.baseline

    def save_classifier(self, path, scaler, lda_model, svm_model, baseline):
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

    def load_classifier(self, path):
        lda_model = pickle.load(open(os.path.join(path, 'LDA_model.pkl'), 'rb'))
        svm_model = pickle.load(open(os.path.join(path, 'SVM_model.pkl'), 'rb'))
        scaler = pickle.load(open(os.path.join(path, 'scaler.pkl'), 'rb'))
        baseline = pickle.load(open(os.path.join(path, 'baseline.pkl'), 'rb'))
        return scaler, lda_model, svm_model, baseline

    def set_classifier(self, baseline, scaler, classifier):
        self.baseline = baseline
        self.scaler = scaler
        self.classifier = classifier

    def get_prediction(self, eeg):

        eeg_features = extract_features([eeg], self.sample_frequency, self.ch_names)
        eeg_features_corrected = baseline_correction(eeg_features, self.baseline)

        # Prediction
        try:
            sample = self.scaler.transform(eeg_features_corrected)
            prediction = self.classifier.predict(sample)
            prediction = int(prediction[0])
        except:
            logging.info('Unicorn: no classifier set')
            prediction = None

        return prediction

    def get_metrics(self, eeg_samples_classes):

        eeg_features_classes = []
        for eeg_samples in eeg_samples_classes:
            eeg_features = extract_features(eeg_samples, self.sample_frequency, self.ch_names, parse=True)
            # Apply baseline correction
            eeg_features_corrected = baseline_correction(eeg_features, self.baseline)
            eeg_features_classes.append(eeg_features_corrected)

        # Prepare the data for classification
        X = np.concatenate(eeg_features_classes)
        y = np.concatenate([np.ones(samples.shape[0]) * i for i, samples in enumerate(eeg_features_classes)])

        # Normalization
        X = self.scaler.transform(X)

        # LDA
        y_pred = self.classifier.predict(X)
        accuracy = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average='macro')

        return accuracy, f1

    def fit_classifier(self, eeg_samples_baseline, eeg_samples_classes):
        # Preprocessing (Feature extraction and Baseline correction)
        baseline = calculate_baseline(eeg_samples_baseline, self.sample_frequency, self.ch_names, parse=True)
        eeg_features_list = []
        for eeg_samples in eeg_samples_classes:
            eeg_features = extract_features(eeg_samples, self.sample_frequency, self.ch_names, parse=True)
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

        return scaler, svm_model, lda_model, baseline

    def save_session(self, path):
        if self.recording_data is not None:
            DataFilter.write_file(self.recording_data, path, "w")
