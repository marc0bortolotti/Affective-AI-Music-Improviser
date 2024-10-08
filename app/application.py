import threading
import logging
from MIDI.midi_communication import MIDI_Input, MIDI_Output
from OSC.osc_connection import Server_OSC, Client_OSC, REC_MSG, CONFIDENCE_MSG, EMOTION_MSG, MOVE_CURSOR_TO_ITEM_END_MSG, MOVE_CURSOR_TO_NEXT_MEASURE_MSG
from EEG.eeg_device import EEG_Device
from generative_model.tokenization import PrettyMidiTokenizer, BCI_TOKENS, IN_OUT_SEPARATOR_TOKEN, SILENCE_TOKEN
from generative_model.architectures.transformer import generate_square_subsequent_mask
import torch
import numpy as np
import time
import os
import logging
import time
import yaml
import importlib
from rapidfuzz import process
import sys

'''---------- CONNECTION PARAMETERS ------------'''
LOCAL_HOST = "127.0.0.1"
OSC_SERVER_PORT = 9000
OSC_REAPER_PORT = 8000
OSC_PROCESSING_PORT = 7000
'''---------------------------------------------'''


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        model.load_state_dict(torch.load(weights_path, map_location = device))
    except Exception as e:
        print(f"Error loading model: {e}")

    model.eval()
    model.to(device)

    logging.info(f"Model {model_class_name} loaded correctly")

    return model, device, INPUT_TOK, OUTPUT_TOK

def thread_function_eeg(name, app):
    logging.info("Thread %s: starting", name)

    app.eeg_device.start_recording()
    time.sleep(5)  # wait for signal to stabilize

    while True:
        time.sleep(app.WINDOW_DURATION)
        if app.eeg_device.params.serial_number == 'Synthetic Board':
            prediction = 0 if app.get_last_eeg_classification() == 'R' else 1
        else:
            eeg = app.eeg_device.get_eeg_data(recording_time=app.WINDOW_DURATION)
            prediction = app.eeg_device.get_prediction(eeg) 

        app.osc_client.send(LOCAL_HOST, OSC_PROCESSING_PORT, EMOTION_MSG, float(prediction))
        app.append_eeg_classification(BCI_TOKENS[prediction])

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
                        init_track_path,
                        ticks_per_beat,
                        generate_rhythm,
                        n_tokens,
                        parse_message=False):
        '''
        Parameters:
        - instrument_in_port_name: name of the MIDI input port for the instrument
        - instrument_out_port_name: name of the MIDI output port for the instrument
        - generation_play_port_name: name of the MIDI output port for the generated melody
        - generation_record_port_name: name of the MIDI output port for the generated melody
        - window_duration: duration of the window in seconds
        - model_dict: path to the model dictionary
        - init_track_path: path to the initial track to start the generation
        - ticks_per_beat: number of ticks per beat
        - generate_rhythm: boolean to generate melody or rhythm
        - n_tokens: number of tokens to generate at each step
        '''

        self.STATUS = {'READY': False, 
                       'RUNNING': False, 
                       'SIMULATE_MIDI': False,
                       'USE_EEG': False,
                       'IS_RECORDING': False}
        
        self.parse_message = parse_message 

        # GENERATION 
        self.generate_rhythm = generate_rhythm
        self.n_tokens = n_tokens

        # MIDI
        self.midi_in = MIDI_Input(instrument_in_port_name, instrument_out_port_name, parse_message=False)
        self.midi_out_play = MIDI_Output(generation_play_port_name)
        # self.midi_out_rec = MIDI_Output(generation_record_port_name)

        # SYNCHRONIZATION
        BPM = 120
        self.osc_server = Server_OSC(LOCAL_HOST, OSC_SERVER_PORT, BPM, parse_message=False)
        self.osc_client = Client_OSC()

        # EEG 
        self.eeg_device = EEG_Device(eeg_device_type)
        self.eeg_classification_buffer = []
        self.WINDOW_DURATION = window_duration
        self.WINDOW_SIZE = int(self.WINDOW_DURATION * self.eeg_device.sample_frequency)

        # AI-MODEL
        model, device, INPUT_TOK, OUTPUT_TOK = load_model(model_param_path, model_module_path, model_class_name, ticks_per_beat)
        self.model_class_name = model_class_name
        self.model = model
        self.device = device
        self.INPUT_TOK = INPUT_TOK
        self.OUTPUT_TOK = OUTPUT_TOK 
        self.BAR_LENGTH = INPUT_TOK.BAR_LENGTH
        self.SEQ_LENGTH = OUTPUT_TOK.SEQ_LENGTH
        self.init_track_path = init_track_path
        self.combine_in_out = False
        for word in self.INPUT_TOK.VOCAB.word2idx.keys():
            if IN_OUT_SEPARATOR_TOKEN in word:
                self.combine_in_out = True
                break
        self.init_tokens = self.OUTPUT_TOK.midi_to_tokens(self.init_track_path, 
                                                          max_len = 8 * self.BAR_LENGTH,
                                                          update_vocab=False,
                                                          convert_to_integers=not self.combine_in_out) 
        self.init_tokens = self.init_tokens[:3*self.BAR_LENGTH]  

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
    
    def append_eeg_classification(self, classification):
        self.eeg_classification_buffer.append(classification)

    def get_application_status(self):
        return self.STATUS

    def set_application_status(self, key, value):
        self.STATUS[key] = value

    def get_eeg_device(self):
        return self.eeg_device
    
    def start_reaper_recording(self):
        self.osc_client.send(LOCAL_HOST, OSC_REAPER_PORT, MOVE_CURSOR_TO_ITEM_END_MSG, 1)
        self.osc_client.send(LOCAL_HOST, OSC_REAPER_PORT, MOVE_CURSOR_TO_NEXT_MEASURE_MSG, 1)
        self.osc_client.send(LOCAL_HOST, OSC_REAPER_PORT, REC_MSG, 1)
        self.set_application_status('IS_RECORDING', True)

    def stop_reaper_recording(self):
        self.osc_client.send(LOCAL_HOST, OSC_REAPER_PORT, REC_MSG, 1)
        self.set_application_status('IS_RECORDING', False)
    
    def save_hystory(self, path):
        path = os.path.join(path, 'hystory.txt')
        with open(path, 'w') as file:
            for item in self.hystory:
                in_bars, out_bars = item
                for i in range(0, 4):
                    file.write(f"bar_{i+1}_IN_OUT:\n")
                    for tok in in_bars[(i)*self.BAR_LENGTH : (i+1)*self.BAR_LENGTH]:
                        text = '{:<30} \t'.format(self.INPUT_TOK.VOCAB.idx2word[tok]) 
                        file.write(text)
                    file.write("\n")
                    for tok in out_bars[(i)*self.BAR_LENGTH: (i+1)*self.BAR_LENGTH]:
                        text = '{:<30} \t'.format(self.OUTPUT_TOK.VOCAB.idx2word[tok]) 
                        file.write(text)
                    file.write("\n")
                file.write("\n")

    def run(self):

        self.STATUS['RUNNING'] = True

        self.start_reaper_recording()

        # start the threads
        self.thread_midi_input.start()
        self.thread_osc.start()
        if self.STATUS['USE_EEG']:
            self.thread_eeg.start()

        input_tokens_buffer = []
        generated_track = None
        confidence = 0.0
        temperature = 1.0

        self.hystory = []

        softmax = torch.nn.Softmax(dim=-1)

        # initialize the target tensor for the transformer
        last_output_string = self.init_tokens.tolist()
        last_output = last_output_string.copy()
        emotion_token = None
        predicted_bar = None
        first_prediction = True

        while True:

            SYNCH_EVENT.wait()
            SYNCH_EVENT.clear()

            if generated_track is not None:
                self.midi_out_play.send_midi_to_reaper(generated_track)

            start_time = time.time()

            # get the notes from the buffer
            notes = self.midi_in.get_note_buffer()
            # for note in notes:
            #     print(note)
            # print()

            # tokenize the input notes
            if len(notes) > 0:
                input_tokens = self.INPUT_TOK.real_time_tokenization(notes, 
                                                                     rhythm=self.generate_rhythm,
                                                                     convert_to_integers=not self.combine_in_out)
                # print(input_tokens.tolist())    
                # print()
                input_tokens_buffer.append(input_tokens)

            # if the buffer is full (4 bars), make the prediction
            if len(input_tokens_buffer) == 3:
                # Flatten the tokens buffer
                input_data = np.array(input_tokens_buffer).flatten().tolist()

                # Add the separator token between the input and the output 
                if self.combine_in_out:
                    count = 0
                    for i in range(len(last_output_string)):
                        input_data[i] = input_data[i] + IN_OUT_SEPARATOR_TOKEN + last_output_string[i]
                        if first_prediction:
                            last_output[i] = input_data[i]
                            first_prediction = False

                        if self.INPUT_TOK.VOCAB.is_in_vocab(input_data[i]):
                            input_data[i] = self.INPUT_TOK.VOCAB.word2idx[input_data[i]] 
                        else:
                            
                            closest_token_string = process.extractOne(input_data[i], self.OUTPUT_TOK.VOCAB.word2idx.keys())
                            if closest_token_string:
                                closest_token_string = closest_token_string[0]
                            else:
                                count += 1
                                closest_token_string = SILENCE_TOKEN+IN_OUT_SEPARATOR_TOKEN+SILENCE_TOKEN
                            input_data[i] = self.INPUT_TOK.VOCAB.word2idx[closest_token_string]
                            
                        if self.OUTPUT_TOK.VOCAB.is_in_vocab(last_output[i]):
                            last_output[i] = self.OUTPUT_TOK.VOCAB.word2idx[last_output_string[i]]
                        else: 
                            # count += 1
                            last_output[i] = self.OUTPUT_TOK.VOCAB.word2idx[SILENCE_TOKEN+IN_OUT_SEPARATOR_TOKEN+SILENCE_TOKEN]
                print(count)
                input_data = np.array(input_data, dtype=np.int32)
                last_output = np.array(last_output, dtype=np.int32)
                input_data = torch.LongTensor(input_data).to(self.device)
                last_output = torch.LongTensor(last_output).to(self.device)

                # add mask to the input data
                if self.STATUS['USE_EEG']:
                    emotion_token = self.get_last_eeg_classification()
                    mask = torch.ones(self.BAR_LENGTH, dtype=torch.long) * self.INPUT_TOK.VOCAB.word2idx[emotion_token]
                else:
                    mask = torch.zeros(self.BAR_LENGTH, dtype=torch.long)
                input_data = torch.cat((input_data, mask))

                # Add the batch dimension.
                input_data = input_data.unsqueeze(0)

                # Make the prediction 
                if 'Transformer' in self.model_class_name:

                    predicted_proba = []
                    predicted_bar = []

                    for i in range(int(self.BAR_LENGTH/self.n_tokens)):

                        # add the batch dimension
                        last_output = last_output.unsqueeze(0)

                        # Generate the mask for the target sequence
                        last_output_mask = generate_square_subsequent_mask(last_output.size(1)).to(self.device)

                        # Make the prediction
                        output = self.model(input_data, last_output, tgt_mask=last_output_mask)
                        output = output.contiguous().view(-1, len(self.OUTPUT_TOK.VOCAB))

                        # Apply the softmax function to the output
                        temperature = self.osc_server.get_temperature()
                        output_no_temp = softmax(output)
                        output = softmax(output / temperature)

                        # Get the probability of the prediction
                        proba = torch.max(output_no_temp, dim=-1)[0][-self.n_tokens:]
                        predicted_proba.append(proba)

                        # Get the last token of the output
                        prediction = torch.argmax(output, 1)
                        next_tokens = prediction[-self.n_tokens:]
                        last_output = torch.cat((last_output[0, self.n_tokens:], next_tokens))

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
                    output_no_temp = softmax(output)
                    output = softmax(output / temperature)
                    predicted_proba = torch.max(output_no_temp, 1)[0] # torch.max returns a tuple (values, indices)

                    # Get the predicted tokens.
                    predicted_bar = torch.argmax(output, 1) [-self.BAR_LENGTH:]
                    predicted_bar = predicted_bar.cpu().numpy().tolist()

                if self.combine_in_out:
                    for i in range(self.BAR_LENGTH):
                        predicted_bar[i] = self.OUTPUT_TOK.VOCAB.idx2word[predicted_bar[i]]
                        predicted_bar[i] = predicted_bar[i].split(IN_OUT_SEPARATOR_TOKEN)[1]
                    last_output_string += predicted_bar
                    last_output_string = last_output_string[self.BAR_LENGTH:]                    

                # Get the confidence of the prediction withouth the temperature contribution.
                confidence = torch.mean(predicted_proba).item() 

                # send the confidence to the processing application 
                self.osc_client.send(LOCAL_HOST, OSC_PROCESSING_PORT, CONFIDENCE_MSG, float(confidence))

                # # save the hystory
                self.hystory.append([input_data.squeeze().tolist(), torch.argmax(output, 1).tolist()])

                # Convert the predicted sequence to MIDI.
                generated_track = self.OUTPUT_TOK.tokens_to_midi(predicted_bar, ticks_filter=0, emotion_token=emotion_token)

                # remove the first bar from the tokens buffer
                input_tokens_buffer.pop(0)

            if self.parse_message:
                logging.info(f"Confidence: {confidence}")
                logging.info(f"Temperature: {temperature}")
                logging.info(f'Emotion: {emotion_token}')
                logging.info(f"Elapsed time: {time.time() - start_time}")
                logging.info(f"Generated sequence: {predicted_bar}")

            if not self.STATUS['RUNNING']:
                break

    def close(self):
        logging.info('Thread Main: Closing application...')

        self.set_application_status('RUNNING', False)

        time.sleep(7)   # wait for the threads to close

        # close the threads
        self.midi_in.close()
        self.thread_midi_input.join()

        self.osc_server.close()  # serve_forever() must be closed outside the thread
        self.thread_osc.join()

        if self.STATUS['USE_EEG']:
            self.thread_eeg.join()

        self.stop_reaper_recording()
        self.midi_out_play.close()

        logging.info('All done')
