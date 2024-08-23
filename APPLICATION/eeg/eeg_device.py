import re
import bluetooth 
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import numpy as np
import logging
from processing import extract_features, baseline_correction
from sklearn.metrics import accuracy_score, f1_score


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

        self.classifier = None
        self.scaler = None
        self.baseline = None

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
    

    def set_classifier(self, baseline, scaler, classifier):
        self.baseline = baseline
        self.scaler = scaler
        self.classifier = classifier

    def get_prediction(self, eeg):

        eeg_features = extract_features([eeg])
        eeg_features_corrected = baseline_correction(eeg_features, self.baseline)

        # Prediction
        try:
            sample = self.scaler.transform(eeg_features_corrected)
            prediction = self.classifier.predict(sample)
        except:
            logging.info('Unicorn: no classifier set')
            prediction = 0
        
        return prediction

    def get_metrics(self, eeg_samples_classes):

        eeg_features_classes = []
        for eeg_samples in eeg_samples_classes:
            eeg_features = extract_features(eeg_samples, fs=self.sample_frequency, chs=self.eeg_channels)
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








