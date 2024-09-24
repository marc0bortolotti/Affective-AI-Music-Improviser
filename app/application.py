import threading
import logging
from MIDI.midi_communication import MIDI_Input, MIDI_Output
from OSC.osc_connection import Server_OSC, Client_OSC, REC_MSG
from EEG.eeg_device import EEG_Device
from generative_model.tokenization import PrettyMidiTokenizer, BCI_TOKENS
import torch
import numpy as np
import time
import os
import logging
import time
import yaml
import importlib
import sys

'''---------- CONNECTION PARAMETERS ------------'''
LOCAL_HOST = "127.0.0.1"
OSC_SERVER_PORT = 9000
OSC_REAPER_PORT = 8000
OSC_PROCESSING_PORT = 7000
'''---------------------------------------------'''

'''--------------- MIDI PARAMETERS --------------'''
BPM = 120
BEAT_PER_BAR = 4
TICKS_PER_BEAT = 12  # quantization of a beat
'''----------------------------------------------'''


def load_model(model_param_path, architecture_file_path, class_name):

    '''
    Load the model from the model dictionary.

    Parameters:
    - model_param_path: path to the model parameters (weights, vocabularies, config)
    - architecture_file_path: path to the architecture file definition
    - class_name: name of the model architecture to load

    '''

    weights_path = os.path.join(model_param_path, 'model_state_dict.pth')
    input_vocab_path = os.path.join(model_param_path, 'input_vocab.txt')
    output_vocab_path = os.path.join(model_param_path, 'output_vocab.txt')

    INPUT_TOK = PrettyMidiTokenizer()
    INPUT_TOK.load_vocab(input_vocab_path)

    OUTPUT_TOK = PrettyMidiTokenizer()
    OUTPUT_TOK.load_vocab(output_vocab_path)

    directory, architecture_filename = os.path.split(architecture_file_path)
    architecture_module_name = architecture_filename.replace('.py', '')
    sys.path.append(directory)
    architecture_module = importlib.import_module(architecture_module_name)

    config_path = os.path.join(model_param_path, 'config.yaml')
    with open(config_path, 'r') as file:
        model_class = getattr(architecture_module, class_name)
        params = yaml.safe_load(file)
        model = model_class(**params)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.load_state_dict(torch.load(weights_path, map_location = device))
    model.eval()
    model.to(device)

    return model, device, INPUT_TOK, OUTPUT_TOK

def thread_function_eeg(name, app):
    logging.info("Thread %s: starting", name)

    if app.STATUS['USE_EEG']:
        app.eeg_device.start_recording()
        time.sleep(5)  # wait for signal to stabilize

        while True:
            time.sleep(app.WINDOW_DURATION)
            eeg = app.eeg_device.get_eeg_data(recording_time=app.WINDOW_DURATION)
            prediction = app.eeg_device.get_prediction(eeg) 
            app.eeg_classification_buffer.append(BCI_TOKENS[prediction])

            if not app.STATUS['RUNNING']:
                logging.info("Thread %s: closing", name)
                break
        app.eeg_device.stop_recording()
        app.eeg_device.close()

def thread_function_midi(name, app):
    logging.info("Thread %s: starting", name)   
    if app.STATUS['SIMULATE_MIDI']:

        # SIMULATION_EVENT = threading.Event()
        # app.midi_in.set_simulation_event(SIMULATION_EVENT)

        while True:
            SYNCH_EVENT.wait()

            app.midi_in.simulate()

            if not app.STATUS['RUNNING']:
                logging.info("Thread %s: closing", name)
                break
    else:
        app.midi_in.run()
    logging.info("Thread %s: closing", name)

def thread_function_osc(name, app):
    logging.info("Thread %s: starting", name)
    app.osc_server.set_event(SYNCH_EVENT)
    app.osc_server.run()
    logging.info("Thread %s: closing", name)




class AI_AffectiveMusicImproviser():

    def __init__(self,  instrument_in_port_name,
                        instrument_out_port_name,
                        generation_play_port_name,
                        # generation_record_port_name,
                        eeg_device_type,
                        window_duration,
                        model_dict,
                        parse_message=False):
        '''
        Parameters:
        - instrument_in_port_name: name of the MIDI input port for the instrument
        - instrument_out_port_name: name of the MIDI output port for the instrument
        - generation_play_port_name: name of the MIDI output port for the generated melody
        - generation_record_port_name: name of the MIDI output port for the generated melody
        - window_duration: duration of the window in seconds
        - model_dict: path to the model dictionary
        '''

        self.STATUS = {'READY': False, 
                       'RUNNING': False, 
                       'SIMULATE_MIDI': False,
                       'USE_EEG': False}
        
        self.parse_message = parse_message

        # MIDI
        self.midi_in = MIDI_Input(instrument_in_port_name, instrument_out_port_name, parse_message=False)
        self.midi_out_play = MIDI_Output(generation_play_port_name)
        # self.midi_out_rec = MIDI_Output(generation_record_port_name)

        # SYNCHRONIZATION
        self.osc_server = Server_OSC(LOCAL_HOST, OSC_SERVER_PORT, BPM, parse_message=False)
        self.osc_client = Client_OSC()

        # EEG 
        self.eeg_device = EEG_Device(eeg_device_type)
        self.eeg_classification_buffer = [BCI_TOKENS[0]]
        self.WINDOW_DURATION = window_duration
        self.WINDOW_SIZE = int(self.WINDOW_DURATION * self.eeg_device.sample_frequency)

        # AI-MODEL
        model, device, INPUT_TOK, OUTPUT_TOK = load_model(model_dict)
        self.model = model
        self.device = device
        self.INPUT_TOK = INPUT_TOK
        self.OUTPUT_TOK = OUTPUT_TOK
        self.BAR_LENGTH = INPUT_TOK.BAR_LENGTH

        # THREADS
        self.thread_midi_input = threading.Thread(target=thread_function_midi, args=('MIDI', self))
        self.thread_osc = threading.Thread(target=thread_function_osc, args=('OSC', self))
        self.thread_eeg = threading.Thread(target=thread_function_eeg, args=('EEG', self))
    
        # set the status of the application
        self.set_application_status('READY', True)

        global SYNCH_EVENT 
        SYNCH_EVENT = threading.Event()

    def get_last_eeg_classification(self):
        return self.eeg_classification_buffer[-1]

    def get_application_status(self):
        return self.STATUS

    def set_application_status(self, key, value):
        self.STATUS[key] = value

    def get_eeg_device(self):
        return self.eeg_device
    
    def save_hystory(self, path):
        path = os.path.join(path, 'hystory.txt')
        with open(path, 'w') as file:
            for i, item in enumerate(self.hystory):
                if i % 2 == 0:
                    file.write("Input: \n")
                    for bar_id, bar in enumerate(item):
                        file.write(f"Bar {bar_id}: ")
                        for tok in bar:
                            file.write(self.INPUT_TOK.VOCAB.idx2word[tok] + ' ')
                        file.write("\n")
                else:
                    file.write("Output: \n")
                    for tok in item:
                        file.write(self.OUTPUT_TOK.VOCAB.idx2word[tok] + ' ')
                    file.write("\n\n")

    def run(self):

        self.STATUS['RUNNING'] = True

        # start recording in Reaper
        self.osc_client.send(LOCAL_HOST, OSC_REAPER_PORT, REC_MSG, 1)

        # start the threads
        self.thread_midi_input.start()
        self.thread_osc.start()
        self.thread_eeg.start()

        tokens_buffer = []
        generated_track = None
        confidence = 0.0
        temperature = 1.0

        self.hystory = []

        softmax = torch.nn.Softmax(dim=1)

        while True:

            SYNCH_EVENT.wait()
            SYNCH_EVENT.clear()

            if generated_track is not None:
                self.midi_out_play.send_midi_to_reaper(generated_track)

            start_time = time.time()

            # get the notes from the buffer
            notes = self.midi_in.get_note_buffer()

            # clear the buffer
            self.midi_in.clear_note_buffer()

            # tokenize the notes
            if len(notes) > 0:
                tokens = self.INPUT_TOK.real_time_tokenization(notes, self.get_last_eeg_classification(), 'drum')
                tokens_buffer.append(tokens)

            # if the buffer is full (3 bars), make the prediction
            if len(tokens_buffer) == 3:
                # Flatten the tokens buffer
                input_data = np.array(tokens_buffer, dtype=np.int32).flatten()

                # Convert the tokens to tensor
                input_data = torch.LongTensor(input_data)

                # Add the batch dimension.
                input_data = input_data.unsqueeze(0)

                # Mask the last bar of the input data.
                input_data = torch.cat((input_data[:, :self.BAR_LENGTH * 3], torch.ones([1, self.BAR_LENGTH], dtype=torch.long)),
                                    dim=1)
                
                logging.info(f"Input sequence: {input_data}")

                # Make the prediction and flatten the output.
                prediction = self.model(input_data.to(self.device))
                prediction = prediction.contiguous().view(-1, len(self.OUTPUT_TOK.VOCAB))

                # Get the probability distribution of the prediction by applying the softmax function and the temperature.
                prediction_no_temperature = prediction
                prediction_no_temperature = softmax(prediction_no_temperature)

                temperature = self.osc_server.get_temperature()
                prediction = prediction / temperature
                prediction = softmax(prediction)

                # Get the confidence of the prediction withouth the temperature contribution.
                confidence = torch.mean(torch.max(prediction_no_temperature, 1)[0]).item() # torch.max returns a tuple (values, indices)
                self.osc_client.send(LOCAL_HOST, OSC_PROCESSING_PORT, '/confidence', confidence)

                # Get the predicted tokens.
                predicted_tokens = torch.argmax(prediction, 1)

                # Get the predicted sequence.
                predicted_sequence = predicted_tokens.cpu().numpy().tolist()

                # get only the last predicted bar
                predicted_sequence = predicted_sequence[-self.BAR_LENGTH:]
                logging.info(f"Generated sequence: {predicted_sequence}")

                # save the hystory
                self.hystory.append(tokens_buffer.copy())
                self.hystory.append(predicted_sequence)

                # Convert the predicted sequence to MIDI.
                generated_track = self.OUTPUT_TOK.tokens_to_midi(predicted_sequence, ticks_filter=0)

                # remove the first bar from the tokens buffer
                tokens_buffer.pop(0)

            if self.parse_message:
                logging.info(f"Confidence: {confidence}")
                logging.info(f"Temperature: {temperature}")
                logging.info(f"Elapsed time: {time.time() - start_time}")

            if not self.STATUS['RUNNING']:
                break

    def close(self):
        logging.info('Thread Main: Closing application...')

        self.set_application_status('RUNNING', False)

        # close the threads
        self.midi_in.close()
        self.thread_midi_input.join()

        self.osc_server.close()  # serve_forever() must be closed outside the thread
        self.thread_osc.join()

        self.thread_eeg.join()

        # stop recording in Reaper
        self.osc_client.send(LOCAL_HOST, OSC_REAPER_PORT, REC_MSG, 1)

        logging.info('All done')
