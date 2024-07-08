import re
import bluetooth 
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import numpy as np
import logging
from eeg.processing import calculate_baseline, extract_features, baseline_correction
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score
from sklearn import svm

def retrieve_eeg_devices():
    saved_devices = bluetooth.discover_devices(duration=1, lookup_names=True, lookup_class=True)
    unicorn_devices = list(filter(lambda x: re.search(r'UN-\d{4}.\d{2}.\d{2}', x[1]), saved_devices))
    enophone_devices = list(filter(lambda x: re.search(r'enophone', x[1]), saved_devices))
    return unicorn_devices, enophone_devices

class Unicorn:

    def __init__(self):
    
        self.params = BrainFlowInputParams()
        logging.info("Unicorn: searching for devices...")
        
        unicorn_devices, enophone_devices = retrieve_eeg_devices()

        if len(unicorn_devices) > 0:
            self.params.serial_number = unicorn_devices[0][1]
            self.params.board_id = BoardIds.UNICORN_BOARD.value
            self.eeg_channels = BoardShim.get_eeg_channels(BoardIds.UNICORN_BOARD.value)
            self.board = BoardShim(BoardIds.UNICORN_BOARD.value, self.params)
        elif len(enophone_devices) > 0:
            self.params.serial_number = enophone_devices[0][1]
            self.params.board_id = BoardIds.ENOPHONE_BOARD.value
            self.eeg_channels = BoardShim.get_eeg_channels(BoardIds.ENOPHONE_BOARD.value)
            self.board = BoardShim(BoardIds.ENOPHONE_BOARD.value, self.params)
        else:
            logging.info("Unicorn: no devices found")
            return 

        logging.info(f"Unicorn: connected to {self.params.serial_number}")
        
        
        self.board.prepare_session()
        self.sample_frequency = 250

        self.exit = False

        # recordings raw data
        self.recording_data = None

    def stop_unicorn_recording(self):
        self.recording_data = self.board.get_board_data()
        self.board.stop_stream()
        logging.info('Unicorn: stop recording')

    def start_unicorn_recording(self):
        self.board.start_stream()
        logging.info('Unicorn: start recording')

    def get_eeg_data(self, recording_time=4):
        data = self.board.get_board_data()
        data = np.array(data).T
        data = data[:, self.eeg_channels]
        data = data[- recording_time * self.sample_frequency : ]
        return data
    
    def close(self):
        self.board.release_session()
        self.exit = True
        logging.info('Unicorn: disconnected')

    def preprocessing(self, eeg_samples_baseline, eeg_samples_list):
        # Calculate baseline
        baseline = calculate_baseline(eeg_samples_baseline, fs=self.sample_frequency, chs=self.eeg_channels)

        eeg_features_list = []
        for eeg_samples in eeg_samples_list:
            eeg_features = extract_features(eeg_samples, fs=self.sample_frequency, chs=self.eeg_channels)
            # Apply baseline correction
            eeg_features_corrected = baseline_correction(eeg_features, baseline)
            eeg_features_list.append(eeg_features_corrected)

        return eeg_features_list, baseline
        
    def eeg_classification(self, eeg_samples_baseline, eeg_samples_list):

        # Preprocessing (Feature extraction and Baseline correction)
        eeg_features_list, baseline = self.preprocessing(eeg_samples_baseline, eeg_samples_list)

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

        return scaler, svm_model, lda_model , baseline
    







