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


    ''' MIDI DRUM'''
    drum_in_port = rtmidi.MidiIn()
    available_in_ports = drum_in_port.get_ports()  
    # inport_idx = int(input(f'\nEnter the idx of the MIDI <<INPUT>> port: {available_ports}\n'))
    idx = 4
    drum_in_port.open_port(idx)
    logging.info(f'MIDI DRUM IN: Connected to port {available_in_ports[idx]}') 

    drum_out_port = rtmidi.MidiOut()
    available_out_ports = drum_out_port.get_ports()
    # outport_idx = int(input(f'\nEnter the idx of the MIDI <<INPUT>> port: {available_ports}\n'))
    idx = 1
    drum_out_port.open_port(idx)
    logging.info(f'MIDI DRUM OUT: Connected to port {available_out_ports[idx]}')
    '''---------------------------------------------'''

    ''' MIDI BASS'''
    # logging.info(mido.get_output_names())
    available_out_port_ = mido.get_output_names()
    idx = 3
    # outport_idx = int(input(f'\nEnter the idx of the MIDI <<OUTPUT>> port: {available_ports}\n'))
    bass_play_port = mido.open_output(available_out_ports[idx])
    logging.info(f'MIDI BASS OUT Playing: Connected to port {available_out_ports[idx]}') 

    # logging.info(mido.get_output_names())
    idx = 2
    # outport_idx = int(input(f'\nEnter the idx of the MIDI <<OUTPUT>> port: {available_ports}\n'))
    bass_record_port = mido.open_output(available_out_ports[idx])
    logging.info(f'MIDI BASS OUT Recording: Connected to port {available_out_ports[idx]}') 
    '''---------------------------------------------'''

    initialize_application(drum_in_port, drum_out_port, bass_play_port, bass_record_port, WINDOW_DURATION, MODEL_DICT)

    # # train the EEG classification model
    # unicorn = get_eeg_device()
    # scaler, svm_model, lda_model, baseline = pretraining(unicorn, WINDOW_SIZE, WINDOW_OVERLAP)
    # command = input('Set EEG model (lda/svm): ')
    # if command == 'lda':
    #     classifier = lda_model
    # else:
    #     classifier = svm_model
    # unicorn.set_classifier(scaler = scaler, classifier = classifier, baseline = baseline)
    # validation(unicorn, WINDOW_DURATION)
    
    command = input('Do you want to start the application? (y/n): ')

    if command == 'y':
        thread_app = threading.Thread(target=run_application, args=())
        thread_app.start()

        time.sleep(5*60)
        set_application_status('STOPPED', True)
        thread_app.join()

        close_application()

    
  




