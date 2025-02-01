import logging
import mne
import os
from app.application import AI_AffectiveMusicImproviser
from app.pretraining import pretraining
from app.validation import validation
from EEG.classifier import load_eeg_classifier
from app.utils import retrieve_eeg_devices, retrieve_midi_ports, retrieve_models
import threading
import brainflow
import time 

# TEST AND TRAINING PATHS
user_name = 'user_5'
test_idx = 0
test_name_list = ['BCI_RELAXED', 'BCI_EXCITED', 'UTENTE_EEG', 'UTENTE_NO_EEG']
test_name = test_name_list[0]

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

# PATHS
PROJECT_PATH = os.path.dirname(__file__)
MODELS_PATH = os.path.join(PROJECT_PATH, 'generative_model/pretrained_models/MT_melody_separateTokens_0')
RUN_PATH = os.path.join(PROJECT_PATH, f'runs/{user_name}')
TEST_PATH = os.path.join(RUN_PATH, f'test_{test_name}_{test_idx}')
TRAINING_PATH = os.path.join(RUN_PATH, 'training')
METRICS_PATH = os.path.join(RUN_PATH, 'EEG_classifier_metrics.txt')

eeg_device_list = retrieve_eeg_devices()
midi_in_port_list, midi_out_port_list = retrieve_midi_ports()
# model_list = retrieve_models(MODELS_PATH)

if not os.path.exists(TRAINING_PATH):
    os.makedirs(TRAINING_PATH)

while os.path.exists(TEST_PATH):
    TEST_PATH = '_'.join(TEST_PATH.split('_')[:-1]) + f'_{test_idx}'
    test_idx += 1 
os.makedirs(TEST_PATH)  

kwargs = {}

kwargs['generation_type'] = 'melody' # melody or rhythm
kwargs['fixed_mood'] = True if 'BCI' in test_name else False
kwargs['instrument_in_port_name'] = midi_out_port_list[0]
kwargs['instrument_out_port_name'] = midi_out_port_list[1]
kwargs['generation_out_port_name'] = midi_out_port_list[2]
kwargs['eeg_device_type'] = eeg_device_list[0][1]

logging.info(f"EEG Device: {kwargs['eeg_device_type']}")
logging.info(f"Generation Type: {kwargs['generation_type']}")
logging.info(f"Fixed Mood: {kwargs['fixed_mood']}")
logging.info(f"Instrument In Port: {kwargs['instrument_in_port_name']}")
logging.info(f"Instrument Out Port: {kwargs['instrument_out_port_name']}")
logging.info(f"Generation Out Port: {kwargs['generation_out_port_name']}")

kwargs['model_module_path'] = os.path.join(PROJECT_PATH, 'generative_model/architectures', 'musicTransformer.py')
kwargs['model_class_name'] = 'MusicTransformer' 
kwargs['model_param_path'] = MODELS_PATH

# Initialize the application
app = AI_AffectiveMusicImproviser(kwargs)

# train the EEG classification model
dialog = input('Do you want to VALIDATE classifiers? y/n: ')

if dialog == 'y':
    pretraining(TRAINING_PATH, METRICS_PATH, app.eeg_device, app.WINDOW_SIZE, WINDOW_OVERLAP, steps=TRAINING_SESSIONS, rec_time=TRAINING_TIME)
            
    # Validate the EEG classifier with both LDA and SVM
    dialog = input('Do you want to VALIDATE classifiers? y/n: ')
    if dialog == 'y':
        validation(TRAINING_PATH, METRICS_PATH, app.eeg_device, app.WINDOW_SIZE, WINDOW_OVERLAP, rec_time=VALIDATION_TIME)

try:
    # Load the EEG classifier from the file
    scaler, lda_model, svm_model, baseline = load_eeg_classifier(TRAINING_PATH)
except:
    logging.error("No classifier found. Please, train the classifier first.")
    exit()

# Set the classifier to be used in the application
dialog = input('Which classifier do you want to use? lda/svm: ')
if dialog == 'lda':
    app.eeg_device.set_classifier(baseline=baseline, classifier=lda_model, scaler=scaler)
else:
    app.eeg_device.set_classifier(baseline=baseline, classifier=svm_model, scaler=scaler)

# Start the application in a separate thread
thread_app = threading.Thread(target=app.run, args=())
thread_app.start()

time.sleep(2 * 65)

app.close()
thread_app.join()
app.eeg_device.save_session(os.path.join(TEST_PATH, f'session.csv'))
app.save_hystory(os.path.join(TEST_PATH))








