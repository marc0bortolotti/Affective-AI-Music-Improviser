import logging
import time
import mne
import os
from app.application import AI_AffectiveMusicImproviser
from EEG.pretraining import pretraining
from EEG.validation import validation
import threading
import brainflow
from PyQt5 import QtWidgets
from gui.dialog_window import SetupDialog, CustomDialog, SIMULATE_INSTRUMENT
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
USE_EEG = True
SAVE_SESSION = False

# PATHS
PROJECT_PATH = os.path.dirname(__file__)
MODEL_PARAM_PATH = os.path.join(PROJECT_PATH, 'generative_model/results/model_20240921-120237')
MODEL_MODULE_PATH = os.path.join(PROJECT_PATH, 'architectures/transformer.py')
SAVE_PATH = os.path.join(PROJECT_PATH, 'output', time.strftime("%Y%m%d-%H%M%S"))


if __name__ == "__main__":

    # Setup the application
    win = QtWidgets.QApplication([])
    dialog = SetupDialog()
    
    if dialog.exec_() == QtWidgets.QDialog.Accepted:
    
        # Get the setup parameters from the dialog window
        setup_parameters = dialog.get_data()

        # Check if the session should be saved and create the folder
        if SAVE_SESSION == True:
            if not os.path.exists(SAVE_PATH):
                os.makedirs(SAVE_PATH)

        # Initialize the application
        app = AI_AffectiveMusicImproviser(  setup_parameters['INSTRUMENT_MIDI_IN_PORT_NAME'], 
                                            setup_parameters['INSTRUMENT_MIDI_OUT_PORT_NAME'], 
                                            setup_parameters['MELODY_MIDI_PLAY_PORT_NAME'], 
                                            # setup_parameters['MELODY_MIDI_REC_PORT_NAME'], 
                                            setup_parameters['EEG_DEVICE_SERIAL_NUMBER'],
                                            WINDOW_DURATION, 
                                            MODEL_DICT,
                                            parse_message=True)

        # Set the simulation mode
        if setup_parameters['INSTRUMENT_MIDI_IN_PORT_NAME'] == SIMULATE_INSTRUMENT:
            app.set_application_status('SIMULATE_MIDI', True)
            app.midi_in.set_midi_simulation_port(setup_parameters['INSTRUMENT_MIDI_OUT_PORT_NAME'])

        # Set if the EEG device should be used in the application    
        if USE_EEG:
            app.set_application_status('USE_EEG', True)

        # Train the EEG classifier
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
                # Load the EEG classifier from the file
                scaler, lda_model, svm_model, baseline = app.eeg_device.load_classifier(os.path.join(PROJECT_PATH, 'eeg/pretrained_classifier'))
            except:
                logging.error("No classifier found. Please, train the classifier first.")
                sys.exit()
        
        # Validate the EEG classifier with both LDA and SVM
        dialog = CustomDialog('Do you want to VALIDATE classifiers?')
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

        # Set the classifier to be used in the application
        dialog = CustomDialog('Which classifier do you want to use?', buttons=['LDA', 'SVM'])
        if dialog.exec_() == 0:
            app.eeg_device.set_classifier(baseline=baseline, classifier=lda_model, scaler=scaler)
        else:
            app.eeg_device.set_classifier(baseline=baseline, classifier=svm_model, scaler=scaler)

        # Start the application
        start_dialog = CustomDialog('Do you want to START the application?')
        if start_dialog.exec_() == 0:

            # Start the application in a separate thread
            thread_app = threading.Thread(target=app.run, args=())
            thread_app.start()

            # Close the application
            close_dialog = CustomDialog('Do you want to CLOSE the application?')
            if close_dialog.exec_() == 0:

                if SAVE_SESSION:
                    app.eeg_device.save_session(os.path.join(SAVE_PATH, 'session.csv'))
                    app.save_hystory(os.path.join(SAVE_PATH, 'history.csv'))

                app.close()

        else:
            sys.exit()







