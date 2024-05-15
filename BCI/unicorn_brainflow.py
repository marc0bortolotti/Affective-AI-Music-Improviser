import re
# pip install pybluez2
import bluetooth 
import argparse
import time
import mne
import matplotlib.pyplot as plt
import matplotlib
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import numpy as np

matplotlib.use("Qt5Agg")


def retrieve_unicorn_devices():
    saved_devices = bluetooth.discover_devices(duration=1, lookup_names=True, lookup_class=True)
    unicorn_devices = filter(lambda x: re.search(r'UN-\d{4}.\d{2}.\d{2}', x[1]), saved_devices)
    return list(unicorn_devices)


def collect_unicorn_data(recording_time=10):
    BoardShim.enable_dev_board_logger()
    # use synthetic board for demo
    params = BrainFlowInputParams()

    # Get bluetooth devices that match the UN-XXXX.XX.XX pattern
    print(retrieve_unicorn_devices())
    params.serial_number = retrieve_unicorn_devices()[0][1]

    # Create a board object and prepare the session
    board = BoardShim(BoardIds.UNICORN_BOARD.value, params)
    board.prepare_session()
    print('Session started')
    board.start_stream()

    # Get data from the board, 10 seconds in this example, then close the session
    time.sleep(recording_time)
    print('Session stopped')
    data = board.get_board_data()
    board.stop_stream()
    board.release_session()
    eeg_channels = BoardShim.get_eeg_channels(BoardIds.UNICORN_BOARD.value)
    eeg_data = data[eeg_channels, :]

    # convert to numpy array and transpose
    eeg_data = np.array(eeg_data).T

    return eeg_data


    # # Creating MNE objects from brainflow data arrays
    # eeg_data = eeg_data / 1e6  # BrainFlow returns uV, convert to V for MNE
    # ch_types = ['eeg'] * len(eeg_channels)
    # ch_names = BoardShim.get_eeg_names(BoardIds.UNICORN_BOARD.value)
    # sfreq = BoardShim.get_sampling_rate(BoardIds.UNICORN_BOARD.value)
    # info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    # raw = mne.io.RawArray(eeg_data, info)

    # # Plot the data using MNE
    # raw.plot()
    # raw.compute_psd().plot(average=True)
    # plt.show()
    # plt.savefig('psd.png')


if __name__ == '__main__':
    eeg = collect_unicorn_data()
