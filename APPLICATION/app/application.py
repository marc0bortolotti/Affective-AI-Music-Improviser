import threading
import logging
from connections.midi_communication import MIDI_Input, MIDI_Output
from connections.osc_connection import Server_OSC, Client_OSC, REC_MSG, SYNCH_MSG, SEND_MSG
from connections.udp_connection import Server_UDP
from model.model import TCN
from model.tokenization import PrettyMidiTokenizer, BCI_TOKENS
import torch
import numpy as np
import time
import os
import logging
import time
import yaml


APPLICATION_STATUS = {'READY': False, 'STOPPED': False}


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


'''--------------- EEG PARAMETERS --------------'''
EEG_CLASSES = ['relax', 'excited']




def thread_function_unicorn(name):
    logging.info("Thread %s: starting", name)

    global eeg_classification_buffer 
    eeg_classification_buffer = [BCI_TOKENS['concentrated']]

    if unicorn is not None:
        unicorn.start_unicorn_recording()
        time.sleep(5) # wait for signal to stabilize

        while True:

            time.sleep(WINDOW_DURATION)
            eeg = unicorn.get_eeg_data(recording_time = WINDOW_DURATION)
            prediction = unicorn.get_prediction(eeg)
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


def get_last_eeg_classification():
    return eeg_classification_buffer[-1]


def get_application_status():
    return APPLICATION_STATUS


def set_application_status(key, value):
    APPLICATION_STATUS[key] = value


def load_model(model_dict):

    model_path = os.path.join(model_dict, 'model_state_dict.pth')
    input_vocab_path = os.path.join(model_dict, 'input_vocab.txt')
    output_vocab_path = os.path.join(model_dict, 'output_vocab.txt')

    INPUT_TOK = PrettyMidiTokenizer()
    INPUT_TOK.load_vocab(input_vocab_path)

    OUTPUT_TOK = PrettyMidiTokenizer()
    OUTPUT_TOK.load_vocab(output_vocab_path)

    config_path = os.path.join(model_dict, 'config.yaml')
    with open(config_path, 'r') as file:
        param = yaml.safe_load(file)
        EMBEDDING_SIZE = param['EMBEDDING_SIZE'] 
        NUM_CHANNELS = param['NUM_CHANNELS']
        INPUT_SIZE = param['INPUT_SIZE']
        OUTPUT_SIZE = param['OUTPUT_SIZE']
        KERNEL_SIZE = param['KERNEL_SIZE']


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TCN(input_size = INPUT_SIZE,
                embedding_size = EMBEDDING_SIZE, 
                output_size = OUTPUT_SIZE, 
                num_channels = NUM_CHANNELS, 
                kernel_size = KERNEL_SIZE) 
    
    model.load_state_dict(torch.load(model_path, map_location = device))
    model.eval()
    model.to(device)

    return model, device, INPUT_TOK, OUTPUT_TOK


def initialize_application( drum_in_port, 
                            drum_out_port, 
                            bass_play_port, 
                            bass_record_port,
                            window_duration,
                            model_dict):
    
    '''
    Parameters:
    - drum_in_port (Int): MIDI input port for the drum
    - drum_out_port (Int): MIDI output port for the drum
    - bass_play_port (Int): MIDI output port for the bass (play)
    - bass_record_port (Int): MIDI output port for the bass (record)
    - window_duration: duration of the window in seconds
    - model_dict: path to the model dictionary
    '''

    global WINDOW_DURATION
    WINDOW_DURATION = window_duration

    logging.info('Thread Main: Starting application...')
    global midi_in, midi_out_rec, midi_out_play
    midi_in = MIDI_Input(drum_in_port, drum_out_port)
    midi_out_rec = MIDI_Output(bass_record_port)
    midi_out_play = MIDI_Output(bass_play_port)

    
    # send the synchronization msg when receives the beat=1 from Reaper
    global osc_server
    osc_server = Server_OSC(self_ip = OSC_SERVER_IP, 
                            self_port = OSC_SERVER_PORT, 
                            udp_ip = UDP_SERVER_IP, 
                            udp_port = UDP_SERVER_PORT, 
                            bpm = BPM,
                            parse_message=False)
    
    global unicorn
    unicorn = None
    # unicorn = Unicorn()
    
    # threads
    global thread_midi_input, thread_osc, thread_unicorn
    thread_midi_input = threading.Thread(target=thread_function_midi, args=('MIDI',))
    thread_osc = threading.Thread(target=thread_function_osc, args=('OSC',))
    thread_unicorn = threading.Thread(target=thread_function_unicorn, args=('Unicorn',))

    # start the threads
    thread_midi_input.start()
    thread_osc.start()
    thread_unicorn.start()

    # load the model
    global model, device, INPUT_TOK, OUTPUT_TOK, BAR_LENGTH
    model, device, INPUT_TOK, OUTPUT_TOK = load_model(model_dict)
    BAR_LENGTH = INPUT_TOK.BAR_LENGTH


def get_eeg_device():
    return unicorn


def run_application():
    
    # start recording in Reaper
    global osc_client
    osc_client = Client_OSC(OSC_REAPER_IP, OSC_REAPER_PORT, parse_message = False)
    osc_client.send(REC_MSG)

    # it receives the synchronization msg 
    server = Server_UDP(ip= UDP_SERVER_IP, port= UDP_SERVER_PORT, parse_message=False)
    server.run()

    APPLICATION_STATUS['READY'] = True

    tokens_buffer = []
    generated_track = None
    hystory = []

    softmax = torch.nn.Softmax(dim=1)

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
                tokens = INPUT_TOK.real_time_tokenization(notes, get_last_eeg_classification(), 'drum')
                tokens_buffer.append(tokens)
 
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
                prediction = prediction.contiguous().view(-1, len(OUTPUT_TOK.VOCAB))

                # Get the predicted tokens.
                prediction = softmax(prediction)

                # Get the confidence of the prediction.
                confidence = torch.mean(torch.max(prediction, 1))
                logging.info(f"Confidence: {confidence}")

                # Get the predicted tokens.
                predicted_tokens = torch.argmax(prediction, 1)

                # Get the predicted sequence.
                predicted_sequence = predicted_tokens.cpu().numpy().tolist()

                # get only the last predicted bar
                predicted_sequence = predicted_sequence[-BAR_LENGTH:]

                # save the hystory
                hystory.append(tokens_buffer.copy())
                hystory.append(predicted_sequence)

                # Convert the predicted sequence to MIDI.
                generated_track =  OUTPUT_TOK.tokens_to_midi(predicted_sequence, ticks_filter=2)

                # remove the first bar from the tokens buffer
                tokens_buffer.pop(0)

            logging.info(f"Elapsed time: {time.time() - start_time}")  

        else:
            time.sleep(0.001)

        if APPLICATION_STATUS['STOPPED']:
            break

    server.close()

    # save the hystory in a txt file
    with open('hystory.txt', 'w') as file:
        for i, item in enumerate(hystory):
            if i % 2 == 0:
                file.write("Input: \n")
                for bar_id, bar in enumerate(item):
                    file.write(f"Bar {bar_id}: ")
                    for tok in bar:
                        file.write(INPUT_TOK.VOCAB.idx2word[tok] + ' ')
                    file.write("\n")
            else:
                file.write("Output: \n")
                for tok in item:
                    file.write(OUTPUT_TOK.VOCAB.idx2word[tok] + ' ')
                file.write("\n\n")


def close_application():
    
    logging.info('Thread Main: Closing application...')

    # close the threads
    midi_in.close()
    thread_midi_input.join()

    osc_server.close()  # serve_forever() must be closed outside the thread
    thread_osc.join()

    thread_unicorn.join()

    # stop recording in Reaper
    osc_client.send(REC_MSG)

    logging.info('All done')