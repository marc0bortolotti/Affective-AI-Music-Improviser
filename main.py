import logging
import time
import mne
import rtmidi
import os
from app.application import AI_AffectiveMusicImproviser
from eeg.pretraining import pretraining, validation
from eeg.eeg_device import EEG_DEVICE_BOARD_TYPES
import threading
import brainflow

# Set log levels
mne.set_log_level(verbose='WARNING', return_old_level=False, add_frames=None)
brainflow.BoardShim.set_log_level(3)

format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO,datefmt="%H:%M:%S")


# EEG PARAMETERS
WINDOW_DURATION = 4 # seconds
WINDOW_OVERLAP = 0.875 # percentage
EEG_DEVICE_TYPE = EEG_DEVICE_BOARD_TYPES['UNICORN'] # 'SYNTHETIC', 'UNICORN', 'ENOPHONE', 'LSL'

# PATHS
PROJECT_PATH = os.path.dirname(__file__)
MODEL_DICT = os.path.join(PROJECT_PATH, 'generative_model/trained_models/model')
SAVE_PATH = os.path.join(PROJECT_PATH, 'output')

# APPLICATION PARAMETERS
SIMULATE_MIDI = False
USE_EEG = True
EEG_TRAINING = False
TRAINING_SESSIONS = 3

if __name__ == "__main__":

    print('\nAvailable INPUT - MIDI ports:')
    for port in rtmidi.MidiIn().get_ports():
        print('\t',port)

    print('\nAvailable OUTPUT - MIDI ports:')
    for port in rtmidi.MidiOut().get_ports():
        print('\t',port)
    print('\n')

    DRUM_IN_PORT_NAME = 'Drum In Port 3'
    DRUM_OUT_PORT_NAME = 'Drum Out Port 1'
    BASS_PLAY_PORT_NAME = 'Bass Out Port Playing 3'
    BASS_RECORD_PORT_NAME = 'Bass Out Port Recording 2'
    SIMULATION_PORT_NAME = 'Simulation Port 5' # output port for MIDI simulation

    app = AI_AffectiveMusicImproviser(DRUM_IN_PORT_NAME, 
                                      DRUM_OUT_PORT_NAME, 
                                      BASS_PLAY_PORT_NAME, 
                                      BASS_RECORD_PORT_NAME, 
                                      EEG_DEVICE_TYPE,
                                      WINDOW_DURATION, 
                                      MODEL_DICT)
    
    if SIMULATE_MIDI:
        app.set_application_status('SIMULATE_MIDI', True)
        app.midi_in.set_midi_simulation_port(SIMULATION_PORT_NAME)

    if USE_EEG:
        app.set_application_status('USE_EEG', True)

    if EEG_TRAINING:
        # train the EEG classification model
        scaler, svm_model, lda_model, baseline = pretraining(app.eeg_device, app.WINDOW_SIZE, WINDOW_OVERLAP, steps=TRAINING_SESSIONS)

        # Save the EEG classifier and the EEG raw data
        app.eeg_device.save_session(os.path.join(SAVE_PATH, 'pretraining.csv'))
        app.eeg_device.save_classifier(SAVE_PATH, scaler=scaler, svm_model=svm_model, lda_model=lda_model, baseline=baseline)
    else:
        scaler, lda_model, svm_model, baseline = app.eeg_device.load_classifier(SAVE_PATH)
    
    command = input('Set EEG model (lda/svm): ')
    if command == 'lda':
        classifier = lda_model
    else:
        classifier = svm_model
    app.eeg_device.set_classifier(baseline=baseline, classifier=classifier, scaler=scaler)

    # Validate the EEG classifier
    validation(app.eeg_device, app.WINDOW_SIZE, WINDOW_OVERLAP)
    app.eeg_device.save_session(os.path.join(SAVE_PATH, 'validation.csv'))
    
    command = input('Do you want to start the application? (y/n): ')

    if command == 'y':
        thread_app = threading.Thread(target=app.run, args=())
        thread_app.start()

        time.sleep(2*60)

        app.close()
        app.eeg_device.save_session(os.path.join(SAVE_PATH, 'session.csv'))

        thread_app.join()

    
  




