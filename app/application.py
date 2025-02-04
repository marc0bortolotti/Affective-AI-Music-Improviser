import threading
import logging
from MIDI.midi_communication import MIDI_Input, MIDI_Output
from OSC.osc_connection import Server_OSC, Client_OSC, REC_MSG, CONFIDENCE_MSG, EMOTION_MSG, MOVE_CURSOR_TO_ITEM_END_MSG, MOVE_CURSOR_TO_NEXT_MEASURE_MSG
from EEG.eeg_device import EEG_Device
from generative_model.tokenization import PrettyMidiTokenizer, BCI_TOKENS, IN_OUT_SEPARATOR_TOKEN, SILENCE_TOKEN
from generative_model.architectures.transformer import generate_square_subsequent_mask
from app.utils import SIMULATE_INSTRUMENT
import torch
import numpy as np
import time
import os
import logging
import time
import yaml
import importlib
import sys

# CONNECTION PARAMETERS
OSC_REAPER_PORT = 8000
OSC_PROCESSING_PORT = 7000
LOCAL_HOST = "127.0.0.1"

# PATHS
SIMULATION_TRACK_PATH = 'app/music/rhythm_RELAXED.mid'
INIT_TRACK_PATH_R = 'app/music/melody_RELAXED.mid'
INIT_TRACK_PATH_C = 'app/music/melody_CONCENTRATED.mid'

def load_model(model_param_path, model_module_path, model_class_name, ticks_per_beat):

    '''
    Load the model from the model dictionary.

    Parameters:
    - model_param_path: path to the model parameters (weights, vocabularies, config)
    - model_module_path: path to the model module
    - model_module_name: name of the model class
    - ticks_per_beat: number of ticks per beat

    '''
    weights_path = os.path.join(model_param_path, 'model_state_dict.pth')
    input_vocab_path = os.path.join(model_param_path, 'input_vocab.txt')
    output_vocab_path = os.path.join(model_param_path, 'output_vocab.txt')

    INPUT_TOK = PrettyMidiTokenizer(TICKS_PER_BEAT=ticks_per_beat)
    INPUT_TOK.load_vocab(input_vocab_path)

    OUTPUT_TOK = PrettyMidiTokenizer(TICKS_PER_BEAT=ticks_per_beat)
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

    device = torch.device('cpu') # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        model.load_state_dict(torch.load(weights_path, map_location = device))
    except Exception as e:
        print(f"Error loading model: {e}")

    model.eval()

    logging.info(f"Model {model_class_name} loaded correctly")

    return model, device, INPUT_TOK, OUTPUT_TOK

def thread_function_eeg(name, app):
    logging.info("Thread %s: starting", name)

    app.eeg_device.start_recording()
    app.eeg_device.insert_marker('A')
    time.sleep(5)  # wait for signal to stabilize

    while True:
        time.sleep(app.WINDOW_DURATION)

        app.eeg_device.insert_marker('WD')
        # get the EEG data
        eeg = app.eeg_device.get_eeg_data(recording_time=app.WINDOW_DURATION)
        # get the predictied emotion
        prediction = app.eeg_device.get_prediction(eeg) 
        # send the emotion to the processing application
        app.osc_client.send(LOCAL_HOST, app.OSC_PROCESSING_PORT, EMOTION_MSG, float(prediction))
        # append the emotion to the buffer
        app.append_eeg_classification(BCI_TOKENS[prediction])

        if app.EXIT:
            logging.info("Thread %s: closing", name)
            break

    app.eeg_device.stop_recording()
    app.eeg_device.close()

def thread_function_midi(name, app):
    
    logging.info("Thread %s: starting", name)   

    app.midi_in.open_port() 
    app.midi_out_play.open_port()
    
    if app.SIMULATE_MIDI:
        app.midi_in.set_simulation_event(SYNCH_EVENT)
        app.midi_in.simulate()
    else:
        app.midi_in.run()
    logging.info("Thread %s: closing", name)

def thread_function_osc(name, osc_server):
    global SYNCH_EVENT 
    SYNCH_EVENT = threading.Event()
    logging.info("Thread %s: starting", name)
    osc_server.set_event(SYNCH_EVENT)
    osc_server.run()
    logging.info("Thread %s: closing", name)


class AI_AffectiveMusicImproviser():

    def __init__(self, kwargs):

        self.instrument_in_port_name = kwargs['instrument_in_port_name']
        self.instrument_out_port_name = kwargs['instrument_out_port_name']
        self.generation_out_port_name = kwargs['generation_out_port_name']
        self.eeg_device_type = kwargs['eeg_device_type']
        self.model_param_path = kwargs['model_param_path']
        self.model_module_path = kwargs['model_module_path']
        self.model_class_name = kwargs['model_class_name']
        self.osc_client = kwargs['osc_client']
        self.osc_server = kwargs['osc_server']
        self.ticks_per_beat = kwargs.get('ticks_per_beat', 4)
        self.window_duration = kwargs.get('window_duration', 4)
        self.generate_rhythm = kwargs.get('generate_rhythm', True)
        self.n_tokens = kwargs.get('n_tokens', 4)
        self.parse_message = kwargs.get('parse_message', True)
        self.BPM = kwargs.get('BPM', 120)
        self.OSC_PROCESSING_PORT = kwargs.get('OSC_PROCESSING_PORT', 7000)
        self.initial_mood = kwargs.get('initial_mood', BCI_TOKENS[0])
        self.fixed_mood = kwargs.get('fixed_mood', False)
        self.EXIT = False
        self.SIMULATE_MIDI = False
        self.USE_EEG = False
        
        self.hystory = []

        # MIDI
        self.midi_in = MIDI_Input(self.instrument_in_port_name, self.instrument_out_port_name, parse_message=False)
        self.midi_out_play = MIDI_Output(self.generation_out_port_name)
        if self.instrument_in_port_name == SIMULATE_INSTRUMENT:
            self.SIMULATE_MIDI = True
            self.midi_in.set_midi_simulation(simulation_port = self.instrument_out_port_name,
                                             simulation_track_path = SIMULATION_TRACK_PATH)

        # EEG 
        if self.eeg_device_type != 'None':
            self.USE_EEG = True
        self.eeg_device = EEG_Device(self.eeg_device_type)
        self.eeg_classification_buffer = []
        self.WINDOW_DURATION = self.window_duration
        self.WINDOW_SIZE = int(self.WINDOW_DURATION * self.eeg_device.sample_frequency)

        # AI-MODEL
        model, device, INPUT_TOK, OUTPUT_TOK = load_model(self.model_param_path, 
                                                          self.model_module_path, 
                                                          self.model_class_name, 
                                                          self.ticks_per_beat)
        self.model = model
        self.device = device
        self.INPUT_TOK = INPUT_TOK
        self.OUTPUT_TOK = OUTPUT_TOK 
        self.BAR_LENGTH = INPUT_TOK.BAR_LENGTH
        self.SEQ_LENGTH = OUTPUT_TOK.SEQ_LENGTH
        self.combine_in_out = False
        for word in self.INPUT_TOK.VOCAB.word2idx.keys():
            if IN_OUT_SEPARATOR_TOKEN in word:
                self.combine_in_out = True
                break

        init_track = INIT_TRACK_PATH_R if self.initial_mood == BCI_TOKENS[0] else INIT_TRACK_PATH_C
        self.init_tokens = self.OUTPUT_TOK.midi_to_tokens(init_track, 
                                                        max_len = 30 * self.BAR_LENGTH,
                                                        update_vocab=False,
                                                        convert_to_integers=not self.combine_in_out) 
        
        i = np.random.randint(0, len(self.init_tokens) - 3*self.BAR_LENGTH)
        self.init_tokens = self.init_tokens[i:i+3*self.BAR_LENGTH]  

        # THREADS
        self.thread_midi_input = threading.Thread(target=thread_function_midi, args=('MIDI', self))
        self.thread_eeg = threading.Thread(target=thread_function_eeg, args=('EEG', self))

    def get_last_eeg_classification(self):
        return self.eeg_classification_buffer[-1]
    
    def append_eeg_classification(self, classification):
        self.eeg_classification_buffer.append(classification)

    def get_eeg_device(self):
        return self.eeg_device
    
    def start_reaper_recording(self):
        self.osc_client.send(LOCAL_HOST, OSC_REAPER_PORT, MOVE_CURSOR_TO_ITEM_END_MSG, 1)
        self.osc_client.send(LOCAL_HOST, OSC_REAPER_PORT, MOVE_CURSOR_TO_NEXT_MEASURE_MSG, 1)
        self.osc_client.send(LOCAL_HOST, OSC_REAPER_PORT, REC_MSG, 1)

    def stop_reaper_recording(self):
        self.osc_client.send(LOCAL_HOST, OSC_REAPER_PORT, REC_MSG, 1)
    
    def save_hystory(self, path):
        if len(self.hystory) > 0:
            history_path = os.path.join(path, 'hystory.txt')
            with open(history_path, 'w') as file:
                for item in self.hystory:
                    in_bars, out_bars = item
                    for tok in in_bars:
                        text = '{:<30} \t'.format(self.INPUT_TOK.VOCAB.idx2word[tok]) 
                        file.write(text)
                    file.write("\n")
                    for tok in out_bars:
                        text = '{:<30} \t'.format(self.OUTPUT_TOK.VOCAB.idx2word[tok]) 
                        file.write(text)
                    file.write("\n\n")
                    
            emotion_path = os.path.join(path, 'emotions.txt')
            with open(emotion_path, 'w') as file:
                for emotion in self.eeg_classification_buffer:
                    file.write(f'{emotion}\n')

    def run(self):

        logging.info('Thread App: started')

        self.start_reaper_recording()

        # start the threads
        self.thread_midi_input.start()
        if self.USE_EEG:
            self.thread_eeg.start()

        input_tokens_buffer = np.array([])
        generated_track = None
        confidence = 0.0
        temperature = 1.0

        softmax = torch.nn.Softmax(dim=-1)

        # initialize the target tensor for the transformer
        last_output_only_output = self.init_tokens.copy()
        last_output =  self.init_tokens.copy()
        emotion_token = None
        predicted_bar = None
        first_iter = True

        count = 0
        elapsed_time_list = []
        elapsed_time = 0
        mean_elapsed_time = 0

        while True:

            SYNCH_EVENT.wait()
            SYNCH_EVENT.clear()

            if generated_track is not None:
                self.midi_out_play.send_midi_to_reaper(generated_track)

            start_time = time.time()

            # get the notes from the buffer and flush it
            notes = self.midi_in.get_note_buffer()

            # tokenize the input notes
            if len(notes) > 0:
                input_tokens = self.INPUT_TOK.real_time_tokenization(notes, 
                                                                     rhythm=self.generate_rhythm,
                                                                     convert_to_integers=not self.combine_in_out)

                input_tokens_buffer = np.concatenate((input_tokens_buffer, input_tokens))

            # if the buffer is full (4 bars), make the prediction
            if len(input_tokens_buffer) == 3*self.BAR_LENGTH:

                count += 1

                input_data = input_tokens_buffer.copy()

                if self.combine_in_out:
                    for i in range(len(input_tokens_buffer)):
                        # Add the separator token between the input and the output 
                        token_string = input_tokens_buffer[i] + IN_OUT_SEPARATOR_TOKEN + last_output_only_output[i]

                        # Convert the token to the integer representation
                        if not self.INPUT_TOK.VOCAB.is_in_vocab(token_string):
                            if self.USE_EEG:
                                if self.fixed_mood:
                                    token_string = self.eeg_classification_buffer[0] 
                                else:
                                    token_string = self.get_last_eeg_classification()
                            else:
                                token_string = SILENCE_TOKEN
                            
                        input_data[i] = self.INPUT_TOK.VOCAB.word2idx[token_string] 

                    if first_iter:
                        last_output = input_data.copy()
                        first_iter = False
                
                input_data = np.array(input_data, dtype=np.int32)
                last_output = np.array(last_output, dtype=np.int32)
                input_data = torch.LongTensor(input_data)
                last_output = torch.LongTensor(last_output)

                # add mask to the input data
                if self.USE_EEG:
                    if self.fixed_mood:
                        emotion_token = self.eeg_classification_buffer[0]
                    else:
                        emotion_token = self.get_last_eeg_classification()
                    mask = torch.ones(self.BAR_LENGTH, dtype=torch.long) * self.INPUT_TOK.VOCAB.word2idx[emotion_token]
                else:
                    mask = torch.zeros(self.BAR_LENGTH, dtype=torch.long)
                input_data = torch.cat((input_data, mask))

                # Add the batch dimension.
                input_data = input_data.unsqueeze(0)

                # Make the prediction 
                if 'Transformer' in self.model_class_name:

                    predicted_proba = torch.tensor([])
                    predicted_bar = np.array([], dtype=np.int32)

                    for i in range(int(self.BAR_LENGTH/self.n_tokens)):

                        # add the batch dimension
                        last_output = last_output.unsqueeze(0)

                        # Generate the mask for the target sequence
                        last_output_mask = generate_square_subsequent_mask(last_output.size(1))

                        # Make the prediction
                        output = self.model(input_data, last_output, tgt_mask=last_output_mask)[0]

                        # Apply the softmax function to the output
                        temperature = self.osc_server.get_temperature()
                        output_no_temp = softmax(output)
                        output = softmax(output / temperature)

                        # Get the probability of the prediction
                        proba = torch.max(output_no_temp, dim=-1)[0][-self.n_tokens:]
                        predicted_proba = torch.cat((predicted_proba, proba))

                        # Get the last token of the output
                        prediction = torch.multinomial(output, 1).squeeze(1)    
                        next_tokens = prediction[-self.n_tokens:]
                        last_output = torch.cat((last_output[0, self.n_tokens:], next_tokens))

                        # Update the target tensor
                        predicted_bar = np.concatenate((predicted_bar, next_tokens.numpy()))

                        # save the hystory
                        self.hystory.append([input_data[0], prediction])

                else:
                    # Make the prediction
                    output = self.model(input_data)[0]

                    # Get the probability distribution of the prediction by applying the softmax function and the temperature.
                    temperature = self.osc_server.get_temperature()
                    output_no_temp = softmax(output)
                    output = softmax(output / temperature)
                    predicted_proba = torch.max(output_no_temp, 1)[0] # torch.max returns a tuple (values, indices)

                    # Get the predicted tokens.
                    prediction = torch.multinomial(output, 1).squeeze(1)
                    predicted_bar = prediction.numpy()[-self.BAR_LENGTH:] 

                    # save the hystory
                    self.hystory.append([input_data[0], prediction])

                if self.combine_in_out:
                    for tok in predicted_bar:
                        # Convert the predicted tokens to the string representation
                        tok = self.OUTPUT_TOK.VOCAB.idx2word[tok]
                        # Split the string in input and output
                        tok = tok.split(IN_OUT_SEPARATOR_TOKEN)[1]
                        # Add the predicted tokens to the last output
                        last_output_only_output = np.append(last_output_only_output, tok)
                    # Remove the first bar from the last output
                    last_output_only_output = last_output_only_output[self.BAR_LENGTH:]                    

                # Get the confidence of the prediction withouth the temperature contribution.
                confidence = torch.mean(predicted_proba).item() 

                # send the confidence to the processing application 
                self.osc_client.send(LOCAL_HOST, self.OSC_PROCESSING_PORT, CONFIDENCE_MSG, float(confidence))

                # Convert the predicted sequence to MIDI.
                generated_track = self.OUTPUT_TOK.tokens_to_midi(predicted_bar, ticks_filter=0, emotion_token=emotion_token)

                # remove the first bar from the tokens buffer
                input_tokens_buffer = input_tokens_buffer[self.BAR_LENGTH:]

                # calculate the elapsed time
                elapsed_time = time.time() - start_time
                elapsed_time_list.append(elapsed_time)

                if count % 100 == 0:
                    mean_elapsed_time = np.mean(elapsed_time_list)
                    count = 0
                    elapsed_time_list = []

            if self.parse_message:
                logging.info(f"Confidence: {confidence}")
                logging.info(f"Temperature: {temperature}")
                logging.info(f'Emotion: {emotion_token}')
                logging.info(f"Elapsed time: {elapsed_time}")
                # logging.info(f"Average elapsed time: {mean_elapsed_time}")
                logging.info("Generated sequence:")
                print(predicted_bar)
                print()

            if self.EXIT:
                break

    def close(self):
        logging.info('Thread Main: Closing application...')

        self.EXIT = True

        time.sleep(7)   # wait for the threads to close

        # close the threads
        self.midi_in.close()
        self.thread_midi_input.join()

        if self.USE_EEG:
            self.thread_eeg.join()

        self.stop_reaper_recording()
        self.midi_out_play.close()

        logging.info('All done')
