import re
import bluetooth 
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import numpy as np
import time
import logging
from eeg.processing import extract_features, baseline_correction, calculate_baseline
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

EEG_DEVICE_BOARD_TYPES = {'SYNTHETIC': BoardIds.SYNTHETIC_BOARD.value,
                          'UNICORN':   BoardIds.UNICORN_BOARD.value , 
                          'ENOPHONE':  BoardIds.ENOPHONE_BOARD.value , 
                          'LSL':       'LSL'}

def retrieve_eeg_devices():
    saved_devices = bluetooth.discover_devices(duration=2, flush_cache=True, lookup_names=True, lookup_class=True)
    unicorn_devices = list(filter(lambda x: re.search(r'UN-\d{4}.\d{2}.\d{2}', x[1]), saved_devices))
    enophone_devices = list(filter(lambda x: re.search(r'enophone', x[1]), saved_devices))
    return unicorn_devices, enophone_devices

class EEG_Device:
    def __init__(self, device_board_type):

        self.device_board_type = device_board_type

        self.recording_data = None # recordings raw data
        self.streams = None

        self.classifier = None
        self.scaler = None
        self.baseline = None
    
        self.params = BrainFlowInputParams()

        if not self.device_board_type == EEG_DEVICE_BOARD_TYPES['LSL']: 

            logging.info("EEG Device: searching for devices...")
            unicorn_devices, enophone_devices = retrieve_eeg_devices()
            self.params.board_id = self.device_board_type

            if self.device_board_type == EEG_DEVICE_BOARD_TYPES['UNICORN']:
                if len(unicorn_devices) > 1:
                    print("Available Unicorn devices:")
                    for i, device in enumerate(unicorn_devices):
                        print(f"\t{i}: {device[1]}")
                    id = input('Select the Unicorn device:(0/1/2/...): ')
                    self.params.serial_number = unicorn_devices[int(id)][1]
                elif unicorn_devices:
                    self.params.serial_number = unicorn_devices[0][1]
                else:
                    logging.error("EEG Device: Unicorn device not found")
            elif self.device_board_type == EEG_DEVICE_BOARD_TYPES['ENOPHONE']:
                if enophone_devices:
                    self.params.serial_number = enophone_devices[0][1]
                else:
                    logging.error("EEG Device: Enophone device not found")
            else:
                self.params.serial_number = 'SYNTHETIC_BOARD'

            self.ch_names = BoardShim.get_eeg_names(self.params.board_id)
            self.board = BoardShim(self.params.board_id, self.params)
            self.sample_frequency = self.board.get_sampling_rate(self.params.board_id)
            logging.info(f"EEG Device: connected to {self.params.serial_number}")

            try:
                self.board.prepare_session()
            except Exception as e:
                logging.error(f"EEG Device: {e}")
                self.board.release_session()
        
    def stop_recording(self):
        self.recording_data = self.board.get_board_data()
        self.board.stop_stream()
        logging.info('EEG Device: stop recording')

    def start_recording(self):
        if self.device_board_type == EEG_DEVICE_BOARD_TYPES['LSL']:
            logging.info('LSLDevice: looking for a stream')
            while not self.streams:
                self.streams = resolve_stream()
                time.sleep(1)
            logging.info("LSL stream found: {}".format(self.streams[0].name()))
            self.inlet = StreamInlet(self.streams[0], pylsl.proc_threadsafe)
        else:
            self.board.start_stream()
        logging.info('EEG Device: start recording')

    def get_eeg_data(self, recording_time=4, chunk = False):
        if self.device_board_type == EEG_DEVICE_BOARD_TYPES['LSL']: 
            try:
                if chunk:
                    data, timestamps = self.inlet.pull_chunk(timeout=recording_time, max_samples=int(recording_time * self.sr))
                else:
                    data, timestamps = self.inlet.pull_sample()
                data = np.array(data)
                data = data[:, 0:self.n_eeg_channels]
            except Exception as e:
                logging.error(f"LSLDevice: {e}")
        else:
            data = self.board.get_current_board_data(num_samples=self.sample_frequency * recording_time)
            data = np.array(data).T
            data = data[:, 0:len(self.ch_names)]
            data = data[- recording_time * self.sample_frequency : ]
        return data
    
    def close(self):
        if self.device_board_type == EEG_DEVICE_BOARD_TYPES['LSL']:
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
            print(prediction)
            prediction = int(prediction[0])
        except:
            logging.info('Unicorn: no classifier set')
            prediction = None

        return prediction

    def get_metrics(self, eeg_samples_classes):

        eeg_features_classes = []
        for eeg_samples in eeg_samples_classes:
            eeg_features = extract_features(eeg_samples, self.sample_frequency, self.ch_names)
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
        baseline = calculate_baseline(eeg_samples_baseline, self.sample_frequency, self.ch_names)
        eeg_features_list = []
        for eeg_samples in eeg_samples_classes:
            eeg_features = extract_features(eeg_samples, self.sample_frequency, self.ch_names)
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

    def save_session(self, path):
        if self.recording_data is not None:
            DataFilter.write_file(self.recording_data, path, "w")
        
