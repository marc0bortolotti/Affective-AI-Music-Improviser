import logging
import mne
import os
from app.application import AI_AffectiveMusicImproviser, thread_function_osc
from app.pretraining import pretraining
from app.validation import validation
from EEG.classifier import load_eeg_classifier
from app.utils import retrieve_eeg_devices, retrieve_midi_ports, retrieve_models
from OSC.osc_connection import Client_OSC, Server_OSC, EEG_DEVICES_MSG, MIDI_IN_PORTS_MSG, MIDI_OUT_PORTS_MSG, MODELS_MSG, TX_FINISHED_MSG
import threading
from generative_model.tokenization import BCI_TOKENS
import brainflow
import time 

# TEST PARAMETERS
USE_GUI = False
SKIP_TRAINING = True
SIMULATE_INSTRUMENT = False
user_name = 'user_8'
test_name =  '4_UTENTE_NO_EEG' # 'BCI_RELAXED' 'BCI_EXCITED' 'UTENTE_EEG' 'UTENTE_NO_EEG'

# PATHS
model_name = 'MT_melody_separateTokens_NO_EEG_0' if 'NO' in test_name else 'MT_melody_separateTokens_0'
MODEL_PATH = f'generative_model/pretrained_models/{model_name}'


# Set log levels
mne.set_log_level(verbose='WARNING', return_old_level=False, add_frames=None)
brainflow.BoardShim.set_log_level(3)
format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO,datefmt="%H:%M:%S")
asr_logger = logging.getLogger('asrpy')
asr_logger.setLevel(logging.ERROR)

# CONNECTION PARAMETERS
LOCAL_HOST = "127.0.0.1"
OSC_SERVER_PORT = 9000
OSC_PROCESSING_PORT = 7000

# EEG PARAMETERS
WINDOW_DURATION = 4 # seconds
WINDOW_OVERLAP = 0.875 # percentage

# MIDI PARAMETERS
BPM = 120

# TRAINING AND VALIDATION PARAMETERS
TRAINING_SESSIONS = 1
TRAINING_TIME = 60 # must be larger than 2*WINDOW_DURATION (>8sec)
VALIDATION_TIME = 40 # must be larger than 2*WINDOW_DURATION (>8sec)

# Retrieve the EEG devices and MIDI ports
eeg_device_list = retrieve_eeg_devices()
midi_in_port_list, midi_out_port_list = retrieve_midi_ports()
model_list = retrieve_models('generative_model/pretrained_models')

# Initialize the OSC server and client
osc_server = Server_OSC(LOCAL_HOST, OSC_SERVER_PORT, BPM, parse_message=False)
thread_osc = threading.Thread(target=thread_function_osc, args=('OSC', osc_server))
thread_osc.start()
osc_client = Client_OSC(parse_message=False)

if USE_GUI:
    logging.info("Waiting for the Processing application to be ready...")
    while not osc_server.processing_ready:
        time.sleep(1)

    # Send the EEG devices to the Processing application
    for device in eeg_device_list:
        osc_client.send(LOCAL_HOST, OSC_PROCESSING_PORT, EEG_DEVICES_MSG, device[1])

    for midi_port in midi_in_port_list:
        osc_client.send(LOCAL_HOST, OSC_PROCESSING_PORT, MIDI_IN_PORTS_MSG, midi_port)

    for midi_port in midi_out_port_list:
        osc_client.send(LOCAL_HOST, OSC_PROCESSING_PORT, MIDI_OUT_PORTS_MSG, midi_port)

    for models in model_list:
        osc_client.send(LOCAL_HOST, OSC_PROCESSING_PORT, MODELS_MSG, models)

    osc_client.send(LOCAL_HOST, OSC_PROCESSING_PORT, TX_FINISHED_MSG, 'True')

    # Wait for the Processing application to send the selected parameters
    logging.info("Waiting for the Processing application to set the parameters...")
    while not osc_server.parameters_setted:
        time.sleep(1)

    instrument_in_port_name = osc_server.instrument_midi_in_port 
    instrument_out_port_name = osc_server.instrument_midi_out_port 
    generation_out_port_name = osc_server.generated_midi_out_port
    generation_type = osc_server.generation_type 
    user_name = osc_server.user_name 
    eeg_device_type = osc_server.eeg_device 

else:
    instrument_in_port_name = midi_in_port_list[0] if SIMULATE_INSTRUMENT else midi_in_port_list[-1]
    instrument_out_port_name = midi_out_port_list[1]
    generation_out_port_name = midi_out_port_list[2]
    generation_type = 'melody'
    eeg_device_type = eeg_device_list[0][1] if 'NO' in test_name else eeg_device_list[3][1] 

kwargs = {}
kwargs['generation_type'] = generation_type
kwargs['fixed_mood'] = True if 'BCI' in test_name else False
kwargs['instrument_in_port_name'] = instrument_in_port_name
kwargs['instrument_out_port_name'] = instrument_out_port_name
kwargs['generation_out_port_name'] = generation_out_port_name
kwargs['eeg_device_type'] = eeg_device_type
kwargs['model_module_path'] = 'generative_model/architectures/musicTransformer.py'
kwargs['model_class_name'] = 'MusicTransformer' 
kwargs['osc_client'] = osc_client
kwargs['osc_server'] = osc_server
kwargs['initial_mood'] = BCI_TOKENS[0] if 'RELAXED' in test_name else BCI_TOKENS[1]
kwargs['model_param_path'] = MODEL_PATH
kwargs['window_duration'] = WINDOW_DURATION
kwargs['window_overlap'] = WINDOW_OVERLAP
kwargs["OSC_PROCESSING_PORT"] = OSC_PROCESSING_PORT
kwargs["BPM"] = BPM

logging.info(f"EEG Device: {kwargs['eeg_device_type']}")
logging.info(f"Generation Type: {kwargs['generation_type']}")
logging.info(f"Fixed Mood: {kwargs['fixed_mood']}")
logging.info(f"Instrument In Port: {kwargs['instrument_in_port_name']}")
logging.info(f"Instrument Out Port: {kwargs['instrument_out_port_name']}")
logging.info(f"Generation Out Port: {kwargs['generation_out_port_name']}")
logging.info(f"Model Path: {kwargs['model_param_path']}")

RUN_PATH = f'runs/{user_name}'
test_idx = 0
TEST_PATH = os.path.join(RUN_PATH, f'test_{test_name}_{test_idx}')
TRAINING_PATH = os.path.join(RUN_PATH, 'training')
METRICS_PATH = os.path.join(RUN_PATH, 'EEG_classifier_metrics.txt')

# Create the training and test directories
if not os.path.exists(TRAINING_PATH):
    os.makedirs(TRAINING_PATH)

while os.path.exists(TEST_PATH):
    TEST_PATH = '_'.join(TEST_PATH.split('_')[:-1]) + f'_{test_idx}'
    test_idx += 1 
os.makedirs(TEST_PATH)  

# Initialize the application
app = AI_AffectiveMusicImproviser(kwargs)

start_mood = 1 if 'EXCITED' in test_name else 0
app.append_eeg_classification(BCI_TOKENS[start_mood])

if not SKIP_TRAINING:
    # train the EEG classification model
    dialog = input('\nDo you want to TRAIN classifiers? y/n: ')

    if dialog == 'y':

        pretraining(TRAINING_PATH, METRICS_PATH, app.eeg_device, app.WINDOW_SIZE, WINDOW_OVERLAP, steps=TRAINING_SESSIONS, rec_time=TRAINING_TIME)
                
        dialog = input('\nDo you want to VALIDATE classifiers? y/n: ')
        if dialog == 'y':
            validation(TRAINING_PATH, METRICS_PATH, app.eeg_device, app.WINDOW_SIZE, WINDOW_OVERLAP, rec_time=VALIDATION_TIME)

try:
    # Load the EEG classifier from the file
    scaler, lda_model, svm_model, baseline = load_eeg_classifier(TRAINING_PATH)
except:
    logging.error("\nNo classifier found. Please, train the classifier first.")
    osc_server.close()  # serve_forever() must be closed outside the thread
    thread_osc.join()
    exit()

# Set the classifier to be used in the application
dialog = input('\nWhich classifier do you want to use? lda/svm: ')

if dialog == 'lda':
    app.eeg_device.set_classifier(baseline=baseline, classifier=lda_model, scaler=scaler)
else:
    app.eeg_device.set_classifier(baseline=baseline, classifier=svm_model, scaler=scaler)

dialog = input('\nDo you want to start the application? y/n: ')

if dialog == 'y':
    # Start the application in a separate thread
    thread_app = threading.Thread(target=app.run, args=())
    thread_app.start()

    time.sleep(2*61)

    app.close()
    thread_app.join()
    app.eeg_device.save_session(os.path.join(TEST_PATH, f'session.csv'))
    app.save_hystory(os.path.join(TEST_PATH))
      
osc_server.close()  # serve_forever() must be closed outside the thread
thread_osc.join()










