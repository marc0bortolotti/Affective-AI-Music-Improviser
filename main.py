import logging
import time
import mne
import os
from app.application import AI_AffectiveMusicImproviser
from eeg.pretraining import pretraining
from eeg.validation import validation
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
WINDOW_DURATION = 4 # seconds

# TRAINING AND VALIDATION PARAMETERS
TRAINING_SESSIONS = 1
TRAINING_TIME = 10 # must be larger than 2*WINDOW_DURATION (>8sec)
VALIDATION_TIME = 5

# APPLICATION PARAMETERS
SIMULATE_MIDI = True
USE_EEG = True
SAVE_SESSION = False

# PATHS
PROJECT_PATH = os.path.dirname(__file__)
MODEL_DICT = os.path.join(PROJECT_PATH, 'generative_model/trained_models/model')
SAVE_PATH = os.path.join(PROJECT_PATH, 'output', time.strftime("%Y%m%d-%H%M%S"))


if __name__ == "__main__":

    win = QtWidgets.QApplication([])
    dialog = SetupDialog()
    if dialog.exec_() == QtWidgets.QDialog.Accepted:
        setup_parameters = dialog.get_data()

        if SAVE_SESSION == True:
            if not os.path.exists(SAVE_PATH):
                os.makedirs(SAVE_PATH)

        app = AI_AffectiveMusicImproviser(  setup_parameters['INSTRUMENT_MIDI_IN_PORT_NAME'], 
                                            setup_parameters['INSTRUMENT_MIDI_OUT_PORT_NAME'], 
                                            setup_parameters['MELODY_MIDI_PLAY_PORT_NAME'], 
                                            setup_parameters['MELODY_MIDI_REC_PORT_NAME'], 
                                            setup_parameters['EEG_DEVICE_SERIAL_NUMBER'],
                                            WINDOW_DURATION, 
                                            MODEL_DICT,
                                            parse_message=True)

        if SIMULATE_MIDI:
            app.set_application_status('SIMULATE_MIDI', True)
            app.midi_in.set_midi_simulation_port(setup_parameters['SIMULATION_MIDI_OUT_PORT_NAME'])

        if USE_EEG:
            app.set_application_status('USE_EEG', True)

        dialog = CustomDialog('Do you want to TRAIN EEG classifiers?', buttons=['Yes', 'No'])
        if dialog.exec_() == 0:
            # train the EEG classification model
            scaler, svm_model, lda_model, baseline = pretraining(app.eeg_device, app.WINDOW_SIZE, WINDOW_OVERLAP, steps=TRAINING_SESSIONS, rec_time=TRAINING_TIME)

            # Save the EEG classifier and the EEG raw data
            if SAVE_SESSION:
                app.eeg_device.save_session(os.path.join(SAVE_PATH, 'pretraining.csv'))
                app.eeg_device.save_classifier(SAVE_PATH, scaler=scaler, svm_model=svm_model, lda_model=lda_model, baseline=baseline)
        else:
            try:
                scaler, lda_model, svm_model, baseline = app.eeg_device.load_classifier(os.path.join(PROJECT_PATH, 'eeg/pretrained_classifier'))
            except:
                logging.error("No classifier found. Please, train the classifier first.")
                sys.exit()
        
        # Validate the EEG classifier with both LDA and SVM
        dialog = CustomDialog('Do you want to VALIDATE classifiers?')
        f1_lda = None
        if dialog.exec_() == 0:
            app.eeg_device.set_classifier(baseline=baseline, classifier=lda_model, scaler=scaler)
            print('\nLDA')
            accuracy_lda, f1_lda = validation(app.eeg_device, app.WINDOW_SIZE, WINDOW_OVERLAP, rec_time=VALIDATION_TIME)
            if SAVE_SESSION:
                app.eeg_device.save_session(os.path.join(SAVE_PATH, 'lda_validation.csv'))

            app.eeg_device.set_classifier(baseline=baseline, classifier=svm_model, scaler=scaler)
            print('\nSVM')
            accuracy_svm, f1_svm = validation(app.eeg_device, app.WINDOW_SIZE, WINDOW_OVERLAP, rec_time=VALIDATION_TIME)
            if SAVE_SESSION:
                app.eeg_device.save_session(os.path.join(SAVE_PATH, 'svm_validation.csv'))

        dialog = CustomDialog('Which classifier do you want to use?', buttons=['LDA', 'SVM'])
        if dialog.exec_() == 0:
            app.eeg_device.set_classifier(baseline=baseline, classifier=lda_model, scaler=scaler)
        else:
            app.eeg_device.set_classifier(baseline=baseline, classifier=svm_model, scaler=scaler)

        
        start_dialog = CustomDialog('Do you want to START the application?')
        close_dialog = CustomDialog('Do you want to CLOSE the application?')
        if start_dialog.exec_() == 0:
            thread_app = threading.Thread(target=app.run, args=())
            thread_app.start()
            if close_dialog.exec_() == 0:
                app.close()
                if SAVE_SESSION:
                    app.eeg_device.save_session(os.path.join(SAVE_PATH, 'session.csv'))
                    app.save_hystory(os.path.join(SAVE_PATH, 'history.csv'))
                thread_app.join()
        else:
            sys.exit()







