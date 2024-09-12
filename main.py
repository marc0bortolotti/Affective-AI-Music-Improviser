import logging
import time
import mne
import os
from app.application import AI_AffectiveMusicImproviser
from eeg.pretraining import pretraining, validation
import threading
import brainflow
from PyQt5 import QtWidgets
from gui.dialog_window import SetupDialog, CustomDialog
import sys


# Set log levels
mne.set_log_level(verbose='WARNING', return_old_level=False, add_frames=None)
brainflow.BoardShim.set_log_level(3)

format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO,datefmt="%H:%M:%S")


# EEG PARAMETERS
WINDOW_OVERLAP = 0.875 # percentage

# PATHS
PROJECT_PATH = os.path.dirname(__file__)
MODEL_DICT = os.path.join(PROJECT_PATH, 'generative_model/trained_models/model')
SAVE_PATH = os.path.join(PROJECT_PATH, 'output', time.strftime("%Y%m%d-%H%M%S"))
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

# APPLICATION PARAMETERS
SIMULATE_MIDI = False
USE_EEG = True
TRAINING_SESSIONS = 3

if __name__ == "__main__":

    win = QtWidgets.QApplication([])
    dialog = SetupDialog()
    if dialog.exec_() == QtWidgets.QDialog.Accepted:
        setup_parameters = dialog.get_data()

        app = AI_AffectiveMusicImproviser(  setup_parameters['INSTRUMENT_MIDI_IN_PORT_NAME'], 
                                            setup_parameters['INSTRUMENT_MIDI_OUT_PORT_NAME'], 
                                            setup_parameters['MELODY_MIDI_PLAY_PORT_NAME'], 
                                            setup_parameters['MELODY_MIDI_REC_PORT_NAME'], 
                                            setup_parameters['EEG_DEVICE_SERIAL_NUMBER'],
                                            setup_parameters['WINDOW_DURATION'], 
                                            MODEL_DICT)

        if SIMULATE_MIDI:
            app.set_application_status('SIMULATE_MIDI', True)
            app.midi_in.set_midi_simulation_port(setup_parameters['SIMULATION_MIDI_OUT_PORT_NAME'])

        if USE_EEG:
            app.set_application_status('USE_EEG', True)

        dialog = CustomDialog('Do you want to train the EEG classifier?')
        if dialog.exec_() == 0:
            # train the EEG classification model
            scaler, svm_model, lda_model, baseline = pretraining(app.eeg_device, app.WINDOW_SIZE, WINDOW_OVERLAP, steps=TRAINING_SESSIONS)

            # Save the EEG classifier and the EEG raw data
            app.eeg_device.save_session(os.path.join(SAVE_PATH, 'pretraining.csv'))
            app.eeg_device.save_classifier(SAVE_PATH, scaler=scaler, svm_model=svm_model, lda_model=lda_model, baseline=baseline)
        else:
            try:
                scaler, lda_model, svm_model, baseline = app.eeg_device.load_classifier(os.path.join(PROJECT_PATH, 'output'))
            except:
                logging.error("No classifier found. Please, train the classifier first.")
                sys.exit()
        
        # Validate the EEG classifier with both LDA and SVM
        dialog = CustomDialog('Do you want to VALIDATE the EEG classifier?')
        if dialog.exec_() == 0:
            app.eeg_device.set_classifier(baseline=baseline, classifier=lda_model, scaler=scaler)
            print('\nLDA')
            validation(app.eeg_device, app.WINDOW_SIZE, WINDOW_OVERLAP)
            app.eeg_device.save_session(os.path.join(SAVE_PATH, 'lda_validation.csv'))

            app.eeg_device.set_classifier(baseline=baseline, classifier=svm_model, scaler=scaler)
            print('\nSVM')
            validation(app.eeg_device, app.WINDOW_SIZE, WINDOW_OVERLAP)
            app.eeg_device.save_session(os.path.join(SAVE_PATH, 'svm_validation.csv'))

        # Set the classifier
        dialog = CustomDialog('Validation finished! Which classifier do you want to use?', buttons=['LDA', 'SVM'])
        if dialog.exec_() == 0:
            app.eeg_device.set_classifier(baseline=baseline, classifier=lda_model, scaler=scaler)
        else:
            app.eeg_device.set_classifier(baseline=baseline, classifier=svm_model, scaler=scaler)

        
        start_dialog = CustomDialog('Do you want to start the application?')
        if start_dialog.exec_() == 0:
            thread_app = threading.Thread(target=app.run, args=())
            thread_app.start()
            time.sleep(2*60)
            app.close()
            app.eeg_device.save_session(os.path.join(SAVE_PATH, 'session.csv'))
            thread_app.join()
        else:
            sys.exit()







