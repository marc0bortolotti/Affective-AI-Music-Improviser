import re
import bluetooth 
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import numpy as np
import logging
from eeg.processing import extract_features, baseline_correction
from sklearn.metrics import accuracy_score, f1_score
from pylsl import resolve_stream, StreamInlet
import pylsl 
import time


# set StreamInlet logging level to ERROR
logging.getLogger('pylsl').setLevel(logging.ERROR)

def retrieve_eeg_devices():
    saved_devices = bluetooth.discover_devices(duration=1, lookup_names=True, lookup_class=True)
    unicorn_devices = list(filter(lambda x: re.search(r'UN-\d{4}.\d{2}.\d{2}', x[1]), saved_devices))
    enophone_devices = list(filter(lambda x: re.search(r'enophone', x[1]), saved_devices))
    return unicorn_devices, enophone_devices

class EEG_Device:

    def __init__(self):

        self.exit = False

        # recordings raw data
        self.recording_data = None

        self.classifier = None
        self.scaler = None
        self.baseline = None
    
        self.params = BrainFlowInputParams()
        logging.info("EEG Device: searching for devices...")
        
        unicorn_devices, enophone_devices = retrieve_eeg_devices()

        if len(unicorn_devices) > 0:
            self.params.serial_number = unicorn_devices[0][1]
            self.params.board_id = BoardIds.UNICORN_BOARD.value
            
        elif len(enophone_devices) > 0:
            self.params.serial_number = enophone_devices[0][1]
            self.params.board_id = BoardIds.ENOPHONE_BOARD.value

        else:
            logging.info("EEG Device: no devices found, run a synthetic device")
            self.params.serial_number = 'synthetic'
            return 

        self.eeg_channels = BoardShim.get_eeg_channels(self.params.board_id.value)
        self.board = BoardShim(self.params.board_id, self.params)
        self.sample_frequency = self.board.get_sampling_rate(self.params.board_id.value)
        logging.info(f"EEG Device: connected to {self.params.serial_number}")
        
        self.board.prepare_session()
        
    def stop_recording(self):
        self.recording_data = self.board.get_board_data()
        self.board.stop_stream()
        logging.info('EEG Device: stop recording')

    def start_recording(self):
        self.board.start_stream()
        logging.info('EEG Device: start recording')

    def get_eeg_data(self, recording_time=4):
        data = self.board.get_current_board_data(num_samples=self.sample_frequency * recording_time)
        data = np.array(data).T
        data = data[:, self.eeg_channels]
        data = data[- recording_time * self.sample_frequency : ]
        return data
    
    def close(self):
        self.board.release_session()
        self.exit = True
        logging.info('EEG Device: disconnected')
    

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






class LSLDevice(EEG_Device):
    def __init__(self, n_eeg_channels=8, name="Cortex EEG", type="EEG", sampling_rate=250):
        self.sr = sampling_rate
        self.name = name
        self.type = type
        self.streams = []
        self.inlet = None
        self.n_eeg_channels = n_eeg_channels

    def __str__(self):
        return f"LSLDevice(name={self.name}, type={self.type}, stream={self.streams})"

    def __repr__(self):
        return str(self)

    def stop_recording(self):
        self.inlet.close_stream()
        logging.info('LSLDevice: stop recording')

    def start_recording(self):
        logging.info('LSLDevice: looking for a stream')
        while not self.streams:
            self.streams = resolve_stream('name', self.name)
            time.sleep(1)
        logging.info("LSL stream found: {}".format(self.streams[0].name()))
        self.inlet = StreamInlet(self.streams[0], pylsl.proc_threadsafe)

    def get_eeg_data(self, recording_time=4, chunk=True):
        try:
            if chunk:
                data, timestamps = self.inlet.pull_chunk(timeout=recording_time, max_samples=int(recording_time * self.sr))
            else:
                data, timestamps = self.inlet.pull_sample()
            data = np.array(data)
            data = data[:, 0:self.n_eeg_channels]
            
            logging.info(f"LSLDevice: {data.shape}")
            return data
        except Exception as e:
            logging.error(f"LSLDevice: {e}")

