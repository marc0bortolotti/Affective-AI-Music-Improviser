import logging
import time
import mne
import rtmidi
import os
from app.application import AI_AffectiveMusicImproviser
from eeg.pretraining import pretraining, validation
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

# PATHS
PROJECT_PATH = os.path.dirname(__file__)
MODEL_DICT = os.path.join(PROJECT_PATH, 'model/trained_models/model')

# APPLICATION PARAMETERS
SIMULATION = True
USE_EEG = False

if __name__ == "__main__":

    print('\nAvailable INPUT - MIDI ports:')
    for port in rtmidi.MidiIn().get_ports():
        print('\t',port)

    print('\nAvailable OUTPUT - MIDI ports:')
    for port in rtmidi.MidiOut().get_ports():
        print('\t',port)
    print('\n')

    drum_in_port_name = 'Drum In Port 3'
    drum_out_port_name = 'Drum Out Port 1'
    bass_play_port_name = 'Bass Out Port Playing 3'
    bass_record_port_name = 'Bass Out Port Recording 2'
    simulation_port_name = 'Simulation Port 5' # output port for MIDI simulation

    app = AI_AffectiveMusicImproviser(drum_in_port_name, 
                                      drum_out_port_name, 
                                      bass_play_port_name, 
                                      bass_record_port_name, 
                                      WINDOW_DURATION, 
                                      MODEL_DICT)
    
    if SIMULATION:
        app.set_application_status('SIMULATION', SIMULATION)
        app.midi_in.set_midi_simulation_port(simulation_port_name)

    if USE_EEG:
        app.set_application_status('EEG', True)
        # train the EEG classification model
        scaler, svm_model, lda_model, baseline = pretraining(app.eeg_device, app.WINDOW_SIZE, WINDOW_OVERLAP)
        command = input('Set EEG model (lda/svm): ')
        if command == 'lda':
            classifier = lda_model
        else:
            classifier = svm_model
        app.eeg_device.set_classifier(scaler = scaler, classifier = classifier, baseline = baseline)

        # Validate the EEG classifier
        validation(app.eeg_device, app.WINDOW_SIZE, WINDOW_OVERLAP)
    
    command = input('Do you want to start the application? (y/n): ')

    if command == 'y':
        thread_app = threading.Thread(target=app.run, args=())
        thread_app.start()

        time.sleep(2*60)
        app.close()
        thread_app.join()

    
  




