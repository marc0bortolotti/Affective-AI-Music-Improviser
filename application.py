import threading
import MIDI.metronome as metronome
import MIDI.midi_in as midi_input
import logging
import threading
import time
import numpy as np
import BCI.unicorn_brainflow as unicorn_brainflow
from pretraining import pretraining
from BCI.utils.feature_extraction import extract_features, baseline_correction
import mne

mne.set_log_level(verbose='ERROR', return_old_level=False, add_frames=None)

format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO,datefmt="%H:%M:%S")

BPM = 120
SERVER_IP = '127.0.0.1'
SERVER_PORT = 1001
UNICORN_FS = 250
EEG_CLASSES = ['relax', 'excited']
WINDOW_DURATION = 4 # seconds
WINDOW_SIZE = WINDOW_DURATION * UNICORN_FS # samples
WINDOW_OVERLAP = 0.875 # percentage



def thread_function_metronome(name, click):
    logging.info("Thread %s: starting", name)
    click.start()


def thread_function_unicorn(name, unicorn):
    logging.info("Thread %s: starting", name)
    unicorn.start_unicorn_recording()
    unicorn.stop_unicorn_recording()
    eeg = unicorn.get_eeg_data()
    logging.info(f"EEG data shape: {eeg.shape}")


def thread_function_midi(name, midi):
    logging.info("Thread %s: starting", name)
    midi.run()


def countdown(duration):
    for i in range(0, duration):
        print('Countdown: {:2d} seconds left'.format(duration - i), end='\r')
        time.sleep(1)
    print('Countdown:  0 seconds left\n')


if __name__ == "__main__":

    unicorn = unicorn_brainflow.Unicorn()
    click = metronome.Metronome(SERVER_IP, SERVER_PORT, BPM)

    # PRETRAINING
    scaler, svm_model, lda_model, baseline = pretraining(unicorn, click, WINDOW_SIZE, WINDOW_OVERLAP)
    
    # REAL TIME CLASSIFICATION
    thread_click = threading.Thread(target=thread_function_metronome, args=('Click', click))
    thread_click.start()   
    logging.info("Main: REAL TIME CLASSIFICATION")
    time.sleep(5)

    for i in range (20):
        countdown(4)
        eeg = unicorn.get_eeg_data(recording_time = 4)
        print(f"EEG data shape: {eeg.shape}")
        eeg_features = extract_features([eeg])
        eeg_features_corrected = baseline_correction(eeg_features, baseline)

        # Prediction
        sample = scaler.transform(eeg_features_corrected)
        prediction = svm_model.predict(sample)
        logging.info(f'Prediction: {EEG_CLASSES[int(prediction)]}')
        prediction_lda = lda_model.predict(sample)
        logging.info(f'Prediction LDA: {EEG_CLASSES[int(prediction_lda)]}')
        # prediction_lda_proba = lda_model.predict_proba(sample)
        # logging.info(f'Prediction LDA Probability: {prediction_lda_proba}')
    
    unicorn.stop_unicorn_recording()
    click.stop()
    thread_click.join()
    logging.info("Thread Click: finishing")








    # midi = midi_input.MIDI_Input(server_ip = SERVER_IP, server_port = SERVER_PORT)
    # thread_midi = threading.Thread(target=thread_function_midi, args=('MIDI Input', midi))
    # thread_unicorn = threading.Thread(target=thread_function_unicorn, args=('Unicorn', unicorn))
    # # thread_midi.start()
    # thread_unicorn.start()
    # logging.info("Main: wait for the thread to finish")

    # # time.sleep(60)
    # # click.stop()
    # # midi.close()

    # # thread_click.join()
    # # thread_midi.join()
    # thread_unicorn.join()
    # logging.info("Thread MIDI Input: finishing")
    # logging.info("Thread Click: finishing")
    # logging.info("Thread unicorn: finishing")
    # logging.info("Main: all done")


