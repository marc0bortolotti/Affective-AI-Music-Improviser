import re
import bluetooth 
import matplotlib
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import numpy as np
import logging
import threading
from BCI.utils.loader import generate_samples
from BCI.utils.feature_extraction import calculate_baseline, extract_features, baseline_correction
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score
from sklearn import svm
import time


def retrieve_unicorn_devices():
        saved_devices = bluetooth.discover_devices(duration=1, lookup_names=True, lookup_class=True)
        unicorn_devices = filter(lambda x: re.search(r'UN-\d{4}.\d{2}.\d{2}', x[1]), saved_devices)
        return list(unicorn_devices)


class Unicorn:

    def __init__(self):
    
        self.params = BrainFlowInputParams()
        logging.info("Unicorn: searching for devices...")
        if retrieve_unicorn_devices():
            self.params.serial_number = retrieve_unicorn_devices()[0][1]
            logging.info(f"Unicorn: connected to {self.params.serial_number}")
        
        self.eeg_channels = BoardShim.get_eeg_channels(BoardIds.UNICORN_BOARD.value)
        self.board = BoardShim(BoardIds.UNICORN_BOARD.value, self.params)
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
    







