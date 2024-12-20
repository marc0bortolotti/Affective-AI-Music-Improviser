import re
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import numpy as np
import time
import logging
from EEG.processing import extract_features, baseline_correction
from sklearn.metrics import accuracy_score, f1_score
from pylsl import resolve_stream, StreamInlet
import pylsl
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

        if serial_number == 'None':
            serial_number = 'Synthetic Board'

        self.recording_data = None  # recordings raw data
        self.streams = None
        self.asr = None # artifact removal 

        self.classifier = None
        self.scaler = None
        self.baseline = None

        self.params = BrainFlowInputParams()
        self.params.serial_number = serial_number
        self.params.board_id = retrieve_board_id(self.params.serial_number)
        self.ch_names = BoardShim.get_eeg_names(self.params.board_id)
        self.board = BoardShim(self.params.board_id, self.params)
        self.sample_frequency = self.board.get_sampling_rate(self.params.board_id)
        logging.info(f"EEG Device: connected to {self.params.serial_number}")

        self.prepare_session()

        self.is_recording = False

    def stop_recording(self):
        self.is_recording = False
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
        if not self.is_recording:
            if self.params.serial_number == 'LSL':
                logging.info('LSLDevice: looking for a stream')
                while not self.streams:
                    self.streams = resolve_stream('name', 'Cortex EEG')
                    time.sleep(1)
                logging.info("LSL stream found: {}".format(self.streams[0].name()))
                self.inlet = StreamInlet(self.streams[0], pylsl.proc_threadsafe)
            else:
                self.board.start_stream()
            self.is_recording = True
            logging.info('EEG Device: start recording')

    def insert_marker(self, marker):
        try:
            self.board.insert_marker(markers_dict[marker])
        except Exception as e:
            logging.error(f"EEG Device: Could not insert marker! {e}")
    
    def set_asr(self, asr):
        self.asr = asr

    def get_eeg_data(self, recording_time=4):
        num_samples = int(self.sample_frequency * recording_time)
        data = self.board.get_current_board_data(num_samples=num_samples)
        data = np.array(data).T
        if self.params.serial_number == 'Synthetic Board':
            data = data[:, 1:(len(self.ch_names)+1)]
        else:
            data = data[:, 0:len(self.ch_names)]
        data = data[- num_samples:]
        return data

    def close(self):
        if self.params.serial_number == 'LSL':
            self.inlet.close_stream()
        else:
            self.board.release_session()
        logging.info('EEG Device: disconnected')

    def get_classifier(self):
        return self.classifier, self.scaler, self.baseline

    def set_classifier(self, baseline, scaler, classifier):
        self.baseline = baseline
        self.scaler = scaler
        self.classifier = classifier

    def get_prediction(self, eeg):

        eeg_features = extract_features([eeg], self.sample_frequency, self.ch_names, self.asr)
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

    def save_session(self, path):
        logging.info(f'Saving EEG session to {path}') 
        if self.recording_data is not None:
            DataFilter.write_file(self.recording_data, path, "w")
