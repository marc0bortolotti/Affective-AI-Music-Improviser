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
from OSC.osc_connection import Server_OSC, Client_OSC, SYNCH_MSG, REC_MSG, SEND_MSG
from MIDI.PRETTY_MIDI.pretty_midi_tokenization import PrettyMidiTokenizer, BCI_TOKENS
from BCI.unicorn_brainflow import Unicorn
from BCI.utils.loader import generate_samples
from BCI.utils.feature_extraction import extract_features, baseline_correction
from MIDI.midi_communication import MIDI_Input, MIDI_Output
from TCN.word_cnn.utils import *
from TCN.word_cnn.model import *
from TCN import *


mne.set_log_level(verbose='ERROR', return_old_level=False, add_frames=None)
format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO,datefmt="%H:%M:%S")


APPLICATION_STATUS = {'READY': False, 'STOPPED': False}


'''--------------------PATH---------------------'''
PROJECT_PATH = os.path.dirname(__file__)
MIDI_PATH = os.path.join(PROJECT_PATH, 'MIDI')
MODEL_DICT = os.path.join(PROJECT_PATH, 'TCN/results/model_short')
MODEL_PATH = os.path.join(MODEL_DICT, 'model_state_dict.pth')
INPUT_VOCAB_PATH = os.path.join(MODEL_DICT, 'input_vocab.txt')
OUTPUT_VOCAB_PATH = os.path.join(MODEL_DICT, 'output_vocab.txt')
'''---------------------------------------------''' 


'''---------- CONNECTION PARAMETERS ------------'''
UDP_SERVER_IP = '127.0.0.1'
UDP_SERVER_PORT = 7000
OSC_SERVER_IP = "127.0.0.1"
OSC_SERVER_PORT = 9000
OSC_REAPER_IP = "127.0.0.1"
OSC_REAPER_PORT = 8000
'''---------------------------------------------'''


'''--------------- MIDI PARAMETERS --------------'''
BPM = 120
BEAT_PER_BAR = 4
TICKS_PER_BEAT = 12 # quantization of a beat
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
def thread_function_unicorn(name):
    logging.info("Thread %s: starting", name)

    global eeg_classification_buffer 
    eeg_classification_buffer = [BCI_TOKENS['concentrate']]

    if unicorn is not None:
        unicorn.start_unicorn_recording()

        while True:

            time.sleep(WINDOW_DURATION)
            eeg = unicorn.get_eeg_data(recording_time = WINDOW_DURATION)
            # eeg_features = extract_features([eeg])
            # eeg_features_corrected = baseline_correction(eeg_features, baseline)

            # # Prediction
            # sample = scaler.transform(eeg_features_corrected)
            # prediction = svm_model.predict(sample)
            # eeg_classification_buffer.append(BCI_TOKENS[prediction])


            if APPLICATION_STATUS['STOPPED']:
                logging.info("Thread %s: closing", name)
                break

        unicorn.stop_unicorn_recording()
        unicorn.close()


def thread_function_midi(name):
    logging.info("Thread %s: starting", name)
    midi_in.run()
    logging.info("Thread %s: closing", name)

    # new_server = Server_UDP(ip= UDP_SERVER_IP, port= 1111)
    # new_server.run()
    # MIDI_FILE_PATH = os.path.join(PROJECT_PATH, 'TCN/dataset/test/drum_rock_relax_one_bar.mid')
    # while True:
    #     msg = new_server.get_message() # NB: it must receive at least one packet, otherwise it will block the loop
    #     if SYNCH_MSG in msg:
    #         midi_in.run_simulation(MIDI_FILE_PATH)
        
    #     if APPLICATION_STATUS['STOPPED']:
    #         logging.info("Thread %s: closing", name)
    #         break
    # new_server.close()
    



def thread_function_osc(name):
    logging.info("Thread %s: starting", name)
    osc_server.run()
    logging.info("Thread %s: closing", name)


def thread_function_server(name):
    logging.info("Thread %s: starting", name)
    server.run()

    global INPUT_TOK
    INPUT_TOK = PrettyMidiTokenizer()
    INPUT_TOK.load_vocab(INPUT_VOCAB_PATH)
    INPUT_VOCAB_SIZE = len(INPUT_TOK.VOCAB)
    BAR_LENGTH = INPUT_TOK.BAR_LENGTH

    OUTPUT_TOK = PrettyMidiTokenizer()
    OUTPUT_TOK.load_vocab(OUTPUT_VOCAB_PATH)
    OUTPUT_VOCAB_SIZE = len(OUTPUT_TOK.VOCAB)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TCN(input_size = INPUT_VOCAB_SIZE,
                embedding_size = EMBEDDING_SIZE, 
                output_size = OUTPUT_VOCAB_SIZE, 
                num_channels = NUM_CHANNELS, 
                kernel_size = 3) 
    model.load_state_dict(torch.load(MODEL_PATH, map_location = device))
    model.eval()
    model.to(device)

    tokens_buffer = []
    generated_track = None
    while True:
    
        msg = server.get_message() # NB: it must receive at least one packet, otherwise it will block the loop

        if SEND_MSG in msg:
            if generated_track is not None:
                midi_out_play.send_midi_to_reaper(generated_track)

        elif SYNCH_MSG in msg:

            if generated_track is not None:
                midi_out_rec.send_midi_to_reaper(generated_track)

            start_time = time.time()
                
            # get the notes from the buffer
            notes = midi_in.get_note_buffer()

            # clear the buffer
            midi_in.clear_note_buffer()

            # tokenize the notes
            if len(notes) > 0:
                tokens = INPUT_TOK.real_time_tokenization(notes, eeg_classification_buffer[-1])
                tokens_buffer.append(tokens)
            else:
                INPUT_TOK.real_time_tokenization([], eeg_classification_buffer[-1])
 
            # if the buffer is full (3 bars), make the prediction
            if len(tokens_buffer) == 3:

                # Flatten the tokens buffer
                input_data = np.array(tokens_buffer, dtype = np.int32).flatten()

                # Convert the tokens to tensor
                input_data = torch.LongTensor(input_data)

                # Add the batch dimension.
                input_data = input_data.unsqueeze(0)

                # Mask the last bar of the input data.
                input_data = torch.cat((input_data[:, :BAR_LENGTH*3], torch.ones([1, BAR_LENGTH], dtype=torch.long)), dim = 1)

                # Make the prediction.
                prediction = model(input_data.to(device))
                prediction = prediction.contiguous().view(-1, OUTPUT_VOCAB_SIZE)

                # Get the predicted tokens.
                predicted_tokens = torch.argmax(prediction, 1)

                # Get the predicted sequence.
                predicted_sequence = predicted_tokens.cpu().numpy().tolist()

                # get only the last predicted bar
                predicted_sequence = predicted_sequence[-BAR_LENGTH:]

                # Convert the predicted sequence to MIDI.
                predicted_sequence =  OUTPUT_TOK.tokens_to_midi(predicted_sequence, ticks_filter=3)
                
                # Generate the track
                generated_track = midi_out_rec.generate_track(predicted_sequence, TICKS_PER_BEAT, BPM)

                # remove the first bar from the tokens buffer
                tokens_buffer.pop(0)

            logging.info(f"Elapsed time: {time.time() - start_time}")  

        else:
            time.sleep(0.001)

        
        if APPLICATION_STATUS['STOPPED']:
            logging.info("Thread %s: closing", name)
            break



    server.close()
'''---------------------------------------------'''     


def get_last_eeg_classification():
    return eeg_classification_buffer[-1]

    
def application_status():
    return APPLICATION_STATUS


def close_application():

    APPLICATION_STATUS['STOPPED'] = True

    logging.info('Thread Main: Closing application...')

    # close the threads
    midi_in.close()
    thread_midi_input.join()

    thread_server.join()

    osc_server.close()  # serve_forever() must be closed outside the thread
    thread_osc.join()

    thread_unicorn.join()

    # stop recording in Reaper
    osc_client.send(REC_MSG)

    logging.info('All done')


def run_application(drum_in_port, drum_out_port, bass_play_port, bass_record_port):

    # PRETRAINING
    # scaler, svm_model, lda_model, baseline = pretraining(unicorn, click, WINDOW_SIZE, WINDOW_OVERLAP)

    logging.info('Thread Main: Starting application...')

    global thread_midi_input, thread_server, thread_osc, thread_unicorn
    global midi_in, midi_out_rec, midi_out_play, server, osc_server, unicorn

    midi_in = MIDI_Input(drum_in_port, drum_out_port)
    midi_out_rec = MIDI_Output(bass_record_port)
    midi_out_play = MIDI_Output(bass_play_port)

    # it receives the synchronization msg 
    server = Server_UDP(ip= UDP_SERVER_IP, port= UDP_SERVER_PORT, parse_message=False)
    
    # send the synchronization msg when receives the beat=1 from Reaper
    osc_server = Server_OSC(self_ip = OSC_SERVER_IP, 
                            self_port = OSC_SERVER_PORT, 
                            udp_ip = UDP_SERVER_IP, 
                            udp_port = UDP_SERVER_PORT, 
                            bpm = BPM,
                            parse_message=False)
    
    # unicorn = Unicorn()
    unicorn = None
    
    # threads
    thread_midi_input = threading.Thread(target=thread_function_midi, args=('MIDI',))
    thread_server = threading.Thread(target=thread_function_server, args=('Server',))
    thread_osc = threading.Thread(target=thread_function_osc, args=('OSC',))
    thread_unicorn = threading.Thread(target=thread_function_unicorn, args=('Unicorn',))

    # start the threads
    thread_server.start()
    thread_midi_input.start()
    thread_osc.start()
    thread_unicorn.start()
    
    APPLICATION_STATUS['READY'] = True

    # start recording in Reaper
    global osc_client
    osc_client = Client_OSC(OSC_REAPER_IP, OSC_REAPER_PORT, parse_message = False)
    osc_client.send(REC_MSG)


def countdown(duration):
    for i in range(0, duration):
        print('Countdown: {:2d} seconds left'.format(duration - i), end='\r')
        time.sleep(1)
    print('Countdown:  0 seconds left\n')


def pretraining():
    logging.info("Pretraining: Start Training")

    # start recording eeg
    unicorn.start_unicorn_recording()

    # Baseline
    logging.info("Pretraining: Pause for 20 seconds. Please, do not move or think about anything. Just relax.")
    countdown(20+5) # Wait 5 seconds for the unicorn signal to stabilize
    eeg = unicorn.get_eeg_data(recording_time = 20)
    eeg_samples_baseline_1 = generate_samples(eeg, WINDOW_SIZE, WINDOW_OVERLAP)

    # Relax 
    logging.info("Pretraining: Play a relaxed rythm on the metronome for 30 seconds")
    countdown(30)
    eeg = unicorn.get_eeg_data(recording_time = 30)
    eeg_samples_relax = generate_samples(eeg, WINDOW_SIZE, WINDOW_OVERLAP)

    # Baseline
    logging.info("Pretraining: Pause for 20 seconds. Please, do not move or think about anything. Just relax.")
    countdown(20)
    eeg = unicorn.get_eeg_data(recording_time = 20)
    eeg_samples_baseline_2 = generate_samples(eeg, WINDOW_SIZE, WINDOW_OVERLAP)

    # Excited
    logging.info("Pretraining: Play an excited rythm on the metronome")
    countdown(30)
    eeg = unicorn.get_eeg_data(recording_time = 30)
    eeg_samples_excited = generate_samples(eeg, WINDOW_SIZE, WINDOW_OVERLAP)


    logging.info("Pretraining: Training Finished")
    eeg_samples_baseline = np.concatenate((eeg_samples_baseline_1, eeg_samples_baseline_2))

    # stop recording eeg
    unicorn.stop_unicorn_recording()

    #------------CLASSIFICATION----------------
    scaler, svm_model, lda_model, baseline = unicorn.eeg_classification(eeg_samples_baseline, [eeg_samples_relax, eeg_samples_excited])

    return scaler, svm_model, lda_model, baseline



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

    run_application(drum_in_port, drum_out_port, bass_play_port, bass_record_port)

    # time.sleep(60)

    # close_application()

    
  




