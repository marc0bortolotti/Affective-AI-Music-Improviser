import logging
import mne
import os
from app.application import AI_AffectiveMusicImproviser
from app.pretraining import pretraining
from app.validation import validation
from EEG.classifier import save_eeg_classifier, load_eeg_classifier
from generative_model.tokenization import BCI_TOKENS
import threading
import brainflow
from PyQt5 import QtWidgets
from gui.dialog_window import SetupDialog, CustomDialog, SIMULATE_INSTRUMENT


# Set log levels
mne.set_log_level(verbose='WARNING', return_old_level=False, add_frames=None)
brainflow.BoardShim.set_log_level(3)

format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO,datefmt="%H:%M:%S")

asr_logger = logging.getLogger('asrpy')
asr_logger.setLevel(logging.ERROR)

# EEG PARAMETERS
WINDOW_OVERLAP = 0.875 # percentage
WINDOW_DURATION = 4 # seconds

# TRAINING AND VALIDATION PARAMETERS
TRAINING_SESSIONS = 1
TRAINING_TIME = 10 # must be larger than 2*WINDOW_DURATION (>8sec)
VALIDATION_TIME = 10 # must be larger than 2*WINDOW_DURATION (>8sec)

# APPLICATION PARAMETERS
SKIP_TRAINING = False
SAVE_SESSION = True

# TEST AND TRAINING PATHS
user_name = 'user_5'
test_idx = 0
test_name_list = ['BCI_RELAXED', 'BCI_EXCITED', 'UTENTE_EEG', 'UTENTE_NO_EEG']
test_name = test_name_list[0]

PROJECT_PATH = os.path.dirname(__file__)
MODELS_PATH = os.path.join(PROJECT_PATH, 'generative_model/pretrained_models')
RUNS_PATH = os.path.join(PROJECT_PATH, f'runs/{user_name}')
TRAINING_PATH = os.path.join(RUNS_PATH, 'training')
TEST_PATH = os.path.join(RUNS_PATH, f'test_{test_name}_{test_idx}')
METRICS_PATH = os.path.join(RUNS_PATH, 'EEG_classifier_metrics.txt')

# Setup the application
win = QtWidgets.QApplication([])
app = None
setup_dialog = SetupDialog(models_path=MODELS_PATH)

while True:              

    if setup_dialog.exec_() == QtWidgets.QDialog.Accepted:

        # Get the setup parameters from the dialog window
        setup_parameters = setup_dialog.get_data()

        if app is not None:
            app.close()
            thread_app.join()
            if SAVE_SESSION:
                app.eeg_device.save_session(os.path.join(TEST_PATH, 'session.csv'))
                app.save_hystory(os.path.join(TEST_PATH))

        # Check if the session should be saved and create the folder
        if SAVE_SESSION == True:
            
            if not os.path.exists(TRAINING_PATH):
                os.makedirs(TRAINING_PATH)

            while os.path.exists(TEST_PATH):
                TEST_PATH = '_'.join(TEST_PATH.split('_')[:-1]) + f'_{test_idx}'
                test_idx += 1 
            os.makedirs(TEST_PATH)   

        generation_type = 'rhythm' if 'rhythm' in setup_parameters['MODEL'] else 'melody'
        input_track_type = 'melody' if generation_type == 'rhythm' else 'rhythm'

        start_mood = 'RELAXED' if 'RELAXED' in setup_parameters['STARTING_MOOD'] else 'CONCENTRATED'
        fixed_mood = True if 'FIXED' in setup_parameters['STARTING_MOOD'] else False

        simulation_track_path = os.path.join(PROJECT_PATH, f'generative_model/dataset/{input_track_type}/{input_track_type}_{start_mood}.mid')
        init_track_path = os.path.join(PROJECT_PATH, f'generative_model/dataset/{generation_type}/{generation_type}_{start_mood}.mid')

        instrument_out_port_name = setup_parameters['MELODY_OUT_PORT_NAME'] if generation_type == 'rhythm'  else setup_parameters['RHYTHM_OUT_PORT_NAME']
        generation_play_port_name = setup_parameters['RHYTHM_OUT_PORT_NAME'] if generation_type == 'rhythm' else setup_parameters['MELODY_OUT_PORT_NAME']

        ticks_per_beat = 4 if generation_type == 'rhythm' else 4
        generate_rhythm = True if generation_type == 'rhythm' else False

        module_name = 'musicTransformer.py' if 'MT' in setup_parameters['MODEL'] else 'tcn.py'
        model_param_path = os.path.join(MODELS_PATH, setup_parameters['MODEL'])
        model_module_path = os.path.join(PROJECT_PATH, 'generative_model/architectures', module_name)
        model_class_name = 'MusicTransformer' if 'MT' in setup_parameters['MODEL'] else 'TCN'

        logging.info(f"Serial number: {setup_parameters['EEG_DEVICE_SERIAL_NUMBER']}")
        # Initialize the application
        app = AI_AffectiveMusicImproviser(  instrument_in_port_name = setup_parameters['INSTRUMENT_IN_PORT_NAME'], 
                                            instrument_out_port_name = instrument_out_port_name,
                                            generation_out_port_name = generation_play_port_name,
                                            eeg_device_type = setup_parameters['EEG_DEVICE_SERIAL_NUMBER'],
                                            window_duration = WINDOW_DURATION,
                                            model_param_path = model_param_path,
                                            model_module_path = model_module_path,
                                            model_class_name = model_class_name,
                                            init_track_path = init_track_path,
                                            ticks_per_beat = ticks_per_beat,
                                            generate_rhythm = generate_rhythm,
                                            n_tokens = ticks_per_beat,
                                            fixed_mood = fixed_mood,
                                            parse_message=True)

        # Set starting mood
        start_emotion = BCI_TOKENS[0] if start_mood == 'RELAXED' else BCI_TOKENS[1]
        app.append_eeg_classification(start_emotion)

        # Set the simulation mode
        if setup_parameters['INSTRUMENT_IN_PORT_NAME'] == SIMULATE_INSTRUMENT:
            app.set_application_status('SIMULATE_MIDI', True)
            app.midi_in.set_midi_simulation(simulation_port = instrument_out_port_name,
                                            simulation_track_path = simulation_track_path)

        # Set if the EEG device should be used in the application    
        if not setup_parameters['EEG_DEVICE_SERIAL_NUMBER'] == 'None':
            app.set_application_status('USE_EEG', True)

        # Train the EEG classifier
        if not SKIP_TRAINING and app.get_application_status()['USE_EEG']:
            dialog = CustomDialog('Do you want to TRAIN EEG classifiers?', buttons=['Yes', 'No'])
            if dialog.exec_() == 0:

                # train the EEG classification model
                scaler, svm_model, lda_model, baseline, accuracy_lda, f1_lda, accuracy_svm, f1_svm = pretraining(app.eeg_device, 
                                                                                                                 app.WINDOW_SIZE, 
                                                                                                                 WINDOW_OVERLAP, 
                                                                                                                 steps=TRAINING_SESSIONS, 
                                                                                                                 rec_time=TRAINING_TIME)

                # Save the EEG classifier and the EEG raw data
                if SAVE_SESSION:
                    save_eeg_classifier(TRAINING_PATH, scaler=scaler, svm_model=svm_model, lda_model=lda_model, baseline=baseline)
                    app.eeg_device.save_session(os.path.join(TRAINING_PATH, 'training.csv'))
                    with open(METRICS_PATH, 'w') as f:
                        f.write('TRAINING\n')
                        f.write(f'LDA-Accuracy: {accuracy_lda}\nLDA-F1 Score: {f1_lda}\nSVM-Accuracy: {accuracy_svm}\nSVM-F1 Score: {f1_svm}')
                    
            else:

                try:
                    # Load the EEG classifier from the file
                    scaler, lda_model, svm_model, baseline = load_eeg_classifier(TRAINING_PATH)
                except:
                    logging.error("No classifier found. Please, train the classifier first.")
                    exit()
            
            # Validate the EEG classifier with both LDA and SVM
            dialog = CustomDialog('Do you want to VALIDATE classifiers?')
            if dialog.exec_() == 0:
                app.eeg_device.set_classifier(baseline=baseline, classifier=lda_model, scaler=scaler)
                print('\nLDA')
                accuracy_lda, f1_lda = validation(app.eeg_device, app.WINDOW_SIZE, WINDOW_OVERLAP, rec_time=VALIDATION_TIME)
                if SAVE_SESSION:
                    app.eeg_device.save_session(os.path.join(TRAINING_PATH, 'validation_lda.csv'))
                    with open(METRICS_PATH, 'a') as f:
                        f.write('\n\nVALIDATION\n')
                        f.write(f'LDA-Accuracy: {accuracy_lda}\nLDA-F1 Score: {f1_lda}')

                app.eeg_device.set_classifier(baseline=baseline, classifier=svm_model, scaler=scaler)
                print('\nSVM')
                accuracy_svm, f1_svm = validation(app.eeg_device, app.WINDOW_SIZE, WINDOW_OVERLAP, rec_time=VALIDATION_TIME)
                if SAVE_SESSION:
                    app.eeg_device.save_session(os.path.join(TRAINING_PATH, 'validation_svm.csv'))
                    with open(METRICS_PATH, 'a') as f:
                        f.write(f'\nSVM-Accuracy: {accuracy_svm}\nSVM-F1 Score: {f1_svm}')

        else:
            try:
                # Load the EEG classifier from the file
                scaler, lda_model, svm_model, baseline = load_eeg_classifier(TRAINING_PATH)
            except:
                logging.error("No classifier found. Please, train the classifier first.")
                exit()

        # Set the classifier to be used in the application
        dialog = CustomDialog('Which classifier do you want to use?', buttons=['LDA', 'SVM'])
        if dialog.exec_() == 0:
            app.eeg_device.set_classifier(baseline=baseline, classifier=lda_model, scaler=scaler)
        else:
            app.eeg_device.set_classifier(baseline=baseline, classifier=svm_model, scaler=scaler)

        # Start the application
        start_dialog = CustomDialog('Do you want to START the application?')
        if start_dialog.exec_() == 0:
            pass
        else:
            break

        # Start the application in a separate thread
        thread_app = threading.Thread(target=app.run, args=())
        thread_app.start()

        # Close the application
        close_dialog = CustomDialog('Do you want to CLOSE the application?')
        if close_dialog.exec_() == 0:
            break
    else:
        break   

if app is not None:
    app.close()
    thread_app.join()
    if SAVE_SESSION:
        app.eeg_device.save_session(os.path.join(TEST_PATH, f'session.csv'))
        app.save_hystory(os.path.join(TEST_PATH))








