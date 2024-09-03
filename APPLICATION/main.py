import logging
import time
import mne
import rtmidi
import mido
import os
from app.application import run_application, close_application, initialize_application, set_application_status, get_eeg_device
from app.pretraining import pretraining, validation
import threading


mne.set_log_level(verbose='ERROR', return_old_level=False, add_frames=None)
format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO,datefmt="%H:%M:%S")

# PATHS
PROJECT_PATH = os.path.dirname(__file__)
MODEL_DICT = os.path.join(PROJECT_PATH, 'model/trained_models/model')

# EEG PARAMETERS
UNICORN_FS = 250
WINDOW_DURATION = 4 # seconds
WINDOW_SIZE = WINDOW_DURATION * UNICORN_FS # samples
WINDOW_OVERLAP = 0.875 # percentage


if __name__ == "__main__":

    print('Available Input MIDI ports:', rtmidi.MidiIn().get_ports())
    print('Available Output MIDI ports:', rtmidi.MidiOut().get_ports())

    drum_in_port_name = 'Drum In Port 3'
    drum_out_port_name = 'Drum Out Port 1'
    bass_play_port_name = 'Bass Out Port Playing 3'
    bass_record_port_name = 'Bass Out Port Recording 2'

    initialize_application(drum_in_port_name, drum_out_port_name, bass_play_port_name, bass_record_port_name, WINDOW_DURATION, MODEL_DICT)

    # train the EEG classification model
    eeg_device = get_eeg_device()
    scaler, svm_model, lda_model, baseline = pretraining(eeg_device, WINDOW_SIZE, WINDOW_OVERLAP)
    command = input('Set EEG model (lda/svm): ')
    if command == 'lda':
        classifier = lda_model
    else:
        classifier = svm_model
    eeg_device.set_classifier(scaler = scaler, classifier = classifier, baseline = baseline)
    # Validate the classifier
    validation(eeg_device, WINDOW_DURATION)
    
    command = input('Do you want to start the application? (y/n): ')

    if command == 'y':
        thread_app = threading.Thread(target=run_application, args=())
        thread_app.start()

        time.sleep(2*60)
        set_application_status('STOPPED', True)
        thread_app.join()

        close_application()

    
  




