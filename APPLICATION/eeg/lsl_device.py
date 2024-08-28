import time

import numpy as np
import pylsl
from pylsl import resolve_stream, StreamInlet
import logging

from sklearn.metrics import accuracy_score, f1_score

from eeg.loader import synth_eeg_channels
from eeg.processing import extract_features, baseline_correction


class LSLDevice:
    def __init__(self, n_eeg_channels=8, name="Cortex EEG", type="EEG", sampling_rate=250):
        self.sr = sampling_rate
        self.name = name
        self.type = type
        self.streams = []
        self.inlet = None

        self.exit = False

        self.n_eeg_channels = n_eeg_channels
        self.classifier = None
        self.scaler = None
        self.baseline = None

    def __str__(self):
        return f"LSLDevice(name={self.name}, type={self.type}, stream={self.streams})"

    def __repr__(self):
        return str(self)

    def stop_unicorn_recording(self):
        self.inlet.close_stream()
        logging.info('LSLDevice: stop recording')

    def start_unicorn_recording(self):
        logging.info('LSLDevice: looking for a stream')
        while not self.streams:
            self.streams = resolve_stream('name', self.name)
            time.sleep(1)
        logging.info("LSL stream found: {}".format(self.streams[0].name()))
        self.inlet = StreamInlet(self.streams[0], pylsl.proc_threadsafe)

    def get_eeg_data(self, recording_time=4, chunk=True):
        try:
            if chunk:
                data, timestamps = self.inlet.pull_chunk(timeout=recording_time)
            else:
                data, timestamps = self.inlet.pull_sample()
            data = np.array(data)
            logging.info(f"LSLDevice: {data.shape}")
            #return data[:, 0:self.n_eeg_channels]
            return data
        except Exception as e:
            logging.error(f"LSLDevice: {e}")

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
