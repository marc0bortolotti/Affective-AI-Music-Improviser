import threading
import logging
from MIDI.midi_communication import MIDI_Input, MIDI_Output
from OSC.osc_connection import Server_OSC, Client_OSC, REC_MSG, CONFIDENCE_MSG, EMOTION_MSG
from EEG.eeg_device import EEG_Device
from generative_model.tokenization import PrettyMidiTokenizer, BCI_TOKENS
from generative_model.architectures.transformer import generate_square_subsequent_mask
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


def load_model(model_param_path, model_module_path, model_class_name):

    '''
    Load the model from the model dictionary.

    Parameters:
    - model_param_path: path to the model parameters (weights, vocabularies, config)
    - model_module_path: path to the model module
    - model_module_name: name of the model class

    '''
    weights_path = os.path.join(model_param_path, 'model_state_dict.pth')
    input_vocab_path = os.path.join(model_param_path, 'input_vocab.txt')
    output_vocab_path = os.path.join(model_param_path, 'output_vocab.txt')

    INPUT_TOK = PrettyMidiTokenizer()
    INPUT_TOK.load_vocab(input_vocab_path)

    OUTPUT_TOK = PrettyMidiTokenizer()
    OUTPUT_TOK.load_vocab(output_vocab_path)

    directory, architecture_filename = os.path.split(model_module_path)
    architecture_module_name = architecture_filename.replace('.py', '')
    sys.path.append(directory)
    architecture_module = importlib.import_module(architecture_module_name)

    config_path = os.path.join(model_param_path, 'config.yaml')
    with open(config_path, 'r') as file:
        model_class = getattr(architecture_module, model_class_name)
        params = yaml.safe_load(file)
        model = model_class(**params)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        model.load_state_dict(torch.load(weights_path, map_location = device))
    except Exception as e:
        print(f"Error loading model: {e}")

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
        app.midi_in.set_simulation_event(SYNCH_EVENT)
        app.midi_in.simulate()
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
                        model_param_path,
                        model_module_path,
                        model_class_name,
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
        model, device, INPUT_TOK, OUTPUT_TOK = load_model(model_param_path, model_module_path, model_class_name)
        self.model_class_name = model_class_name
        self.model = model
        self.device = device
        self.INPUT_TOK = INPUT_TOK
        self.OUTPUT_TOK = OUTPUT_TOK
        self.BAR_LENGTH = INPUT_TOK.BAR_LENGTH
        self.SEQ_LENGTH = OUTPUT_TOK.SEQ_LENGTH

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

        input_tokens_buffer = []
        generated_track = None
        confidence = 0.0
        temperature = 1.0

        self.hystory = []

        softmax = torch.nn.Softmax(dim=-1)
        n_tokens = 4

        # initialize the target tensor for the transformer
        init_midi_path = os.path.join(os.path.dirname(__file__), 'start_token_RELAXED.mid')
        last_output = self.OUTPUT_TOK.midi_to_tokens(init_midi_path)[: 3 * self.BAR_LENGTH]
        last_output = torch.LongTensor(last_output.tolist()).unsqueeze(0).to(self.device)

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

            # Get emotion from the EEG
            emotion_token = self.get_last_eeg_classification()

            # tokenize the input notes
            if len(notes) > 0:
                input_tokens = self.INPUT_TOK.real_time_tokenization(notes, emotion_token, 'drum')
                input_tokens_buffer.append(input_tokens)

            # if the buffer is full (4 bars), make the prediction
            if len(input_tokens_buffer) == 3:
                # Flatten the tokens buffer
                input_data = np.array(input_tokens_buffer, dtype=np.int32).flatten()

                # Convert the tokens to tensor
                input_data = torch.LongTensor(input_data)

                # add mask to the input data
                input_data = torch.cat((input_data, torch.zeros(self.BAR_LENGTH, dtype=torch.long)))

                # Add the batch dimension.
                input_data = input_data.unsqueeze(0).to(self.device)

                # Make the prediction 
                if 'Transformer' in self.model_class_name:

                    predicted_proba = []
                    predicted_bar = []

                    for i in range(int(self.BAR_LENGTH/n_tokens)):

                        # Generate the mask for the target sequence
                        last_output_mask = generate_square_subsequent_mask(last_output.size(1)).to(self.device)

                        # Make the prediction
                        output = self.model(input_data, last_output, tgt_mask=last_output_mask)
                        output = output.contiguous().view(-1, len(self.OUTPUT_TOK.VOCAB))

                        # Apply the softmax function to the output
                        temperature = self.osc_server.get_temperature()
                        output = softmax(output / temperature)

                        # Get the probability of the prediction
                        proba = torch.max(output, dim=-1)[0][-n_tokens:]
                        predicted_proba.append(proba)

                        # Get the last token of the output
                        last_output = torch.argmax(output, dim=-1)
                        next_tokens = last_output[-n_tokens:]
                        last_output = last_output.unsqueeze(0)

                        # Update the target tensor
                        predicted_bar+=next_tokens.cpu().numpy().tolist()
                    
                    # predicted_bar = last_output.squeeze(0).cpu().numpy().tolist() [-self.BAR_LENGTH:]
                    # predicted_proba = torch.max(output, dim=-1)[0][-self.BAR_LENGTH:]
                    predicted_proba = torch.cat(predicted_proba, dim=0)

                else:
                    output = self.model(input_data)

                    # Remove the batch dimension.
                    output = output.contiguous().view(-1, len(self.OUTPUT_TOK.VOCAB))

                    # Get the probability distribution of the prediction by applying the softmax function and the temperature.
                    temperature = self.osc_server.get_temperature()
                    output = output / temperature
                    output = softmax(output)
                    predicted_proba = torch.max(output, 1)[0] # torch.max returns a tuple (values, indices)

                    # Get the predicted tokens.
                    predicted_bar = torch.argmax(output, 1) [-self.BAR_LENGTH:]

                    # Convert the predicted tokens to a list.
                    predicted_bar = predicted_bar.cpu().numpy().tolist()
                
                logging.info(f"Generated sequence: {predicted_bar}")

                # Get the confidence of the prediction withouth the temperature contribution.
                confidence = torch.mean(predicted_proba).item() 

                # send the confidence and the emotion state to the processing application 
                self.osc_client.send(LOCAL_HOST, OSC_PROCESSING_PORT, CONFIDENCE_MSG, confidence)
                self.osc_client.send(LOCAL_HOST, OSC_PROCESSING_PORT, EMOTION_MSG, emotion_token)

                # # save the hystory
                # self.hystory.append(input_tokens_buffer.copy())
                # self.hystory.append(predicted_tokens)

                # Convert the predicted sequence to MIDI.
                generated_track = self.OUTPUT_TOK.tokens_to_midi(predicted_bar, ticks_filter=0)

                # remove the first bar from the tokens buffer
                input_tokens_buffer.pop(0)

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
