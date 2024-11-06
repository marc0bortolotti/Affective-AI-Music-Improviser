import time
import numpy as np
import pylsl
from pylsl import resolve_stream, StreamInlet, StreamOutlet
import logging
from sklearn.metrics import accuracy_score, f1_score
from EEG.processing import extract_features, baseline_correction

markers_dict = {
    'RS': 1,  #resting state
    'PTR': 2,  #pretraining  relaxed
    'PTE': 3,  #pretraining  excited
    'WN': 4,  #white noise
    'P': 5,  #prediction
    'T': 98,  #training mode
    'A': 99,  #application mode
}


class LSLDevice():
    def __init__(self, n_eeg_channels=8, sampling_rate=250, name="Cortex EEG", type="EEG"):
        self.sample_frequency = sampling_rate
        self.eeg_channels = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
        self.name = name
        self.type = type
        self.streams = []
        self.inlet = None

        self.exit = False
        marker_info = pylsl.StreamInfo('AAIMIMarkers', 'Markers', 1, 0, 'string', 'myuidw435')
        self.marker_stream = StreamOutlet(marker_info)

        self.n_eeg_channels = n_eeg_channels
        self.classifier = None
        self.scaler = None
        self.baseline = None

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

    def insert_marker(self, marker):
        try:
            self.marker_stream.push_sample([markers_dict[marker]])
        except Exception as e:
            logging.error(f"LSLDevice: Could not insert marker! {e}")

    def get_eeg_data(self, recording_time=4, chunk=True):
        try:
            data = []
            if chunk:
                data, timestamps = self.inlet.pull_chunk(timeout=recording_time,
                                                         max_samples=int(recording_time * self.sample_frequency))
                data = np.array(data)
                data = data[:, 0:self.n_eeg_channels]
            else:
                for i in range(int(recording_time * self.sample_frequency)):
                    sample, timestamps = self.inlet.pull_sample()
                    data.append(sample)
                data = np.array(data)
                data = data[:, 0:self.n_eeg_channels]
            logging.info(f"LSLDevice: {data.shape}")
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
            eeg_features = extract_features(eeg_samples, fs=self.sample_frequency, ch_names=self.eeg_channels)
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
