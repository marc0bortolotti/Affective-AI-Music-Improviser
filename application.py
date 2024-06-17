import threading
import logging
import threading
import time
import numpy as np
import mne
import rtmidi
import mido
import os
import torch

from UDP.udp_connection import Server_UDP 
from OSC.osc_connection import Server_OSC
from MIDI.PRETTY_MIDI.pretty_midi_tokenization import PrettyMidiTokenizer
import BCI.unicorn_brainflow as unicorn_brainflow
from BCI.pretraining import pretraining
from BCI.utils.feature_extraction import extract_features, baseline_correction
import MIDI.midi_in as midi_input
from TCN.word_cnn.utils import *
from TCN.word_cnn.model import *
from TCN import *


mne.set_log_level(verbose='ERROR', return_old_level=False, add_frames=None)
format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO,datefmt="%H:%M:%S")



'''--------------------PATH---------------------'''
PROJECT_PATH = os.path.dirname(__file__)
MIDI_PATH = os.path.join(PROJECT_PATH, 'MIDI')
MODEL_PATH = os.path.join(PROJECT_PATH, 'TCN/model', 'generated_model_state.pth')
INPUT_VOCAB_PATH = os.path.join(PROJECT_PATH, 'TCN/vocabularies', 'input_vocab.txt')
OUTPUT_VOCAB_PATH = os.path.join(PROJECT_PATH, 'TCN/vocabularies', 'output_vocab.txt')
'''---------------------------------------------''' 


'''---------- CONNECTION PARAMETERS ------------'''
UDP_SERVER_IP = '127.0.0.1'
UDP_SERVER_PORT = 1001
OSC_SERVER_IP = "127.0.0.1"
OSC_SERVER_PORT = 9000
OSC_REAPER_IP = "127.0.0.1"
OSC_REAPER_PORT = 8000
'''---------------------------------------------'''


'''--------------- MIDI PARAMETERS --------------'''
BPM = 120
BEAT_PER_BAR = 4
TICKS_PER_BEAT = 32 # quantization of a beat
'''----------------------------------------------'''


'''---------------MODEL PARAMETERS---------------'''
# Set the hyperparameters
EMBEDDING_SIZE = 20 # size of word embeddings -> Embedding() is used to encode input token into [192, 20] vectors (see model.py)
LEVELS = 7
HIDDEN_UNITS = 192
NUM_CHANNELS = [HIDDEN_UNITS] * (LEVELS - 1) + [EMBEDDING_SIZE]



'''--------------- EEG PARAMETERS --------------'''
UNICORN_FS = 250
EEG_CLASSES = ['relax', 'excited']
WINDOW_DURATION = 4 # seconds
WINDOW_SIZE = WINDOW_DURATION * UNICORN_FS # samples
WINDOW_OVERLAP = 0.875 # percentage
'''---------------------------------------------''' 



'''--------------- THREAD FUNCTIONS --------------'''
def thread_function_unicorn(name, unicorn):
    logging.info("Thread %s: starting", name)
    unicorn.start_unicorn_recording()
    unicorn.stop_unicorn_recording()
    eeg = unicorn.get_eeg_data()
    logging.info(f"EEG data shape: {eeg.shape}")


def thread_function_midi(name, midi_input):
    logging.info("Thread %s: starting", name)
    midi_input.run()


def thread_function_osc(name, osc_server):
    logging.info("Thread %s: starting", name)
    osc_server.run()


def thread_function_server(name, server, midi_in):
    logging.info("Thread %s: starting", name)
    server.run()

    # logging.info(mido.get_output_names())
    playing_port = mido.open_output('loopMIDI Port Playing 2')
    recording_port = mido.open_output('loopMIDI Port Recording 3') 

    INPUT_TOK = PrettyMidiTokenizer()
    INPUT_TOK.load_vocab(INPUT_VOCAB_PATH)
    BAR_LENGTH = INPUT_TOK.BAR_LENGTH

    OUTPUT_TOK = PrettyMidiTokenizer()
    OUTPUT_TOK.load_vocab(OUTPUT_VOCAB_PATH)
    OUTPUT_VOCAB_SIZE = len(OUTPUT_TOK.VOCAB)


    model = TCN(input_size = EMBEDDING_SIZE, 
                output_size = OUTPUT_VOCAB_SIZE, 
                num_channels = NUM_CHANNELS, 
                kernel_size = 3) 
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


    tokens_buffer = []
    predicted_sequence = None
    while not midi_in.exit:
    
        msg = server.get_message() # NB: it must receive at least one packet, otherwise it will block the loop

        if 'One' in msg:

            logging.info(f"Received synchronization message from Reaper")

            if predicted_sequence is not None:
                logging.info(f"Sending MIDI to Reaper")
                OUTPUT_TOK.send_midi_to_reaper(predicted_sequence, playing_port)
                
            # get the notes from the buffer
            notes = midi_in.get_note_buffer()

            # clear the buffer
            midi_in.clear_note_buffer()

            # tokenize the notes
            if len(notes) > 0:
                tokens = INPUT_TOK.real_time_tokenization(notes)
                tokens_buffer.append(tokens)
 
            # if the buffer is full (3 bars), make the prediction
            if len(tokens_buffer) == 3:

                # Flatten the tokens buffer
                input_data = np.array(tokens_buffer, dtype = np.int32).flatten()

                # Convert the tokens to tensor
                input_data = torch.LongTensor(input_data).to(device)

                # Add the batch dimension.
                input_data = input_data.unsqueeze(0)

                # Mask the last bar of the input data.
                input_data = torch.cat((input_data[:, :BAR_LENGTH*3], torch.ones([1, BAR_LENGTH], dtype=torch.long)), dim = 1)

                # Make the prediction.
                prediction = model(input_data)
                prediction = prediction.contiguous().view(-1, OUTPUT_VOCAB_SIZE)

                # Get the predicted tokens.
                predicted_tokens = torch.argmax(prediction, 1)

                # Get the predicted sequence.
                predicted_sequence = predicted_tokens.cpu().numpy().tolist()

                # Convert the predicted sequence to MIDI.
                predicted_sequence =  OUTPUT_TOK.tokens_to_midi(predicted_sequence)

                # get only the last predicted bar
                predicted_sequence = predicted_sequence[-BAR_LENGTH:]

                # remove the first bar from the tokens buffer
                tokens_buffer.pop(0)

    server.close()  

    


def run_application():

    midi_in = rtmidi.MidiIn()
    available_ports = midi_in.get_port_count()
        
    if available_ports == 0:
        logging.info('MIDI Input: No MIDI devices detected. Retrying...')
    while available_ports == 0:
        available_ports = midi_in.get_port_count()
        time.sleep(0.001)

    print(f'Available ports: {midi_in.get_ports()}')
    midi_in.open_port(3)
    logging.info(f'MIDI Input: Connected to port {midi_in.get_port_name(3)}') 

    midi_in = midi_input.MIDI_Input(midi_in)

    # it receives the synchronization msg 
    server = Server_UDP(ip= UDP_SERVER_IP, port= UDP_SERVER_PORT)
    
    # send the synchronization msg when receives the beat=1 from Reaper
    osc_server = Server_OSC(self_ip = OSC_SERVER_IP, 
                            self_port = OSC_SERVER_PORT, 
                            udp_ip = UDP_SERVER_IP, 
                            udp_port = UDP_SERVER_PORT, 
                            bpm = BPM)

    
    # threads
    thread_midi_input = threading.Thread(target=thread_function_midi, args=('MIDI', midi_in))
    thread_server = threading.Thread(target=thread_function_server, args=('Server', server, midi_in))
    thread_osc = threading.Thread(target=thread_function_osc, args=('OSC', osc_server))

    # start the threads
    thread_midi_input.start()
    thread_server.start()
    thread_osc.start()

    time.sleep(120)

    # close the threads
    midi_in.close()
    server.close()
    osc_server.close()

    thread_midi_input.join()
    thread_server.join()
    thread_osc.join()

    logging.info('All done')
    


if __name__ == "__main__":
    run_application()












    # PRETRAINING
    # scaler, svm_model, lda_model, baseline = pretraining(unicorn, click, WINDOW_SIZE, WINDOW_OVERLAP)
    
    # REAL TIME CLASSIFICATION
    # logging.info("Main: REAL TIME CLASSIFICATION")
    # time.sleep(5)

    # for i in range (20):
    #     time.sleep(1)
    #     eeg = unicorn.get_eeg_data(recording_time = WINDOW_DURATION)
    #     print(f"EEG data shape: {eeg.shape}")
    #     eeg_features = extract_features([eeg])
    #     eeg_features_corrected = baseline_correction(eeg_features, baseline)

    #     # Prediction
    #     sample = scaler.transform(eeg_features_corrected)
    #     prediction = svm_model.predict(sample)
    #     logging.info(f'Prediction: {EEG_CLASSES[int(prediction)]}')
    #     prediction_lda = lda_model.predict(sample)
    #     logging.info(f'Prediction LDA: {EEG_CLASSES[int(prediction_lda)]}')
    #     # prediction_lda_proba = lda_model.predict_proba(sample)
    #     # logging.info(f'Prediction LDA Probability: {prediction_lda_proba}')

    
    # unicorn.stop_unicorn_recording()


    # midi = midi_input.MIDI_Input(server_ip = SERVER_IP, server_port = SERVER_PORT)
    # thread_midi = threading.Thread(target=thread_function_midi, args=('MIDI Input', midi))
    # thread_unicorn = threading.Thread(target=thread_function_unicorn, args=('Unicorn', unicorn))
    # # thread_midi.start()
    # thread_unicorn.start()
    # logging.info("Main: wait for the thread to finish")

    # # time.sleep(60)
    # # click.stop()
    # # midi.close()

    # # thread_click.join()
    # # thread_midi.join()
    # thread_unicorn.join()
    # logging.info("Thread MIDI Input: finishing")
    # logging.info("Thread Click: finishing")
    # logging.info("Thread unicorn: finishing")
    # logging.info("Main: all done")

    # def countdown(duration):
    # for i in range(0, duration):
    #     print('Countdown: {:2d} seconds left'.format(duration - i), end='\r')
    #     time.sleep(1)
    # print('Countdown:  0 seconds left\n')


