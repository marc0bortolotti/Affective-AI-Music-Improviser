import mido
import os
from mido import Message, MidiFile, MidiTrack, MetaMessage, tick2second, second2tick
import time
import logging


# DEFAULT PARAMETERS
SILENCE_TOKEN = 0
BPM = 120 
BEAT_PER_BAR = 4
TICKS_PER_BEAT = 128 # quantization of a beat



def get_token_from_midi(mid, bpm = BPM, ticks_per_beat = TICKS_PER_BEAT, beat_per_bar = BEAT_PER_BAR):
    '''
    Description:
    Returns a list of bars, 
    where each bar is a list of tokens,
    representing notes and silences in the MIDI file
    
    Parameters:
    mid (mido.MidiFile): MIDI file to decode
    bpm (int): beats per minute
    ticks_per_beat (int): quantization of a beat
    beat_per_bar (int): number of beats per bar

    Returns:
    bars (list): list of bars, where each bar is a list of tokens
    '''

    logging.info(f'Extracting Token from MIDI file: {mid.filename}...')

    beat_duration =  60/bpm # seconds
    ticks_per_bar = ticks_per_beat * beat_per_bar
    tempo = int(beat_duration * 1000000)


    bars=[]
    tokens = []
    for msg in mid.play():
        if msg.type == 'note_on':
            silent_size = second2tick(msg.time, ticks_per_beat, tempo)
            for i in range(silent_size):
                tokens.append(SILENCE_TOKEN)
                if len(tokens) == ticks_per_bar:
                    bars.append(tokens)
                    tokens = []
        elif msg.type == 'note_off':
            note_size = second2tick(msg.time, ticks_per_beat, tempo)
            for i in range(note_size):
                tokens.append(msg.note)
                if len(tokens) == ticks_per_bar:
                    bars.append(tokens)
                    tokens = []
        else:
            logging.info(msg)
    bars.append(tokens)
    logging.info(f'Done! Number of bars: {len(bars)} of {len(bars[0])} tokens each')
    return bars


def generate_midi_from_tokens(bars, bpm = BPM, ticks_per_beat = TICKS_PER_BEAT,  beat_per_bar = BEAT_PER_BAR, parse_message = False):
    '''
    Description:
    Saves a MIDI file from a list of bars of tokens

    Parameters:
    bars (list): list of bars, where each bar is a list of tokens
    bpm (int): beats per minute
    ticks_per_beat (int): quantization of a beat
    beat_per_bar (int): number of beats per bar

    Returns:
    mid (mido.MidiFile): MIDI file
    '''

    logging.info('Generating MIDI from tokens...')

    beat_duration =  60/bpm # seconds
    tempo = int(beat_duration * 1000000)
    bar_duration = tempo * beat_per_bar

    bars_processed = []
    for bar in bars:
        tokens_vs_ticks_list = tokens_counter(bar)
        bars_processed.append(tokens_vs_ticks_list)

    mid = MidiFile(ticks_per_beat = ticks_per_beat)
    track = MidiTrack()
    mid.tracks.append(track)
    track.append(MetaMessage('set_tempo', tempo=tempo)) # tempo in microseconds per beat

    for i, bar in enumerate(bars_processed):
        if parse_message:
            logging.info(f'BAR {i+1}')
        total_ticks = 0
        for note, ticks in bar: 
            if note != SILENCE_TOKEN:   
                track.append(Message('note_on', note=note, velocity=127, time=0)) # NB: time from the previous message in ticks per beat
                track.append(Message('note_off', note=note, velocity=127, time=ticks))
                if parse_message:
                    logging.info(f'note {note} for {ticks} ticks, {tick2second(ticks, ticks_per_beat, tempo)} seconds')
            else:
                track.append(Message('note_on', note=0, velocity=0, time=0))
                track.append(Message('note_off', note=0, velocity=0, time=ticks))
                if parse_message:
                    logging.info(f'silence for {ticks} ticks, {tick2second(ticks, ticks_per_beat, tempo)} seconds')
            total_ticks += ticks
        if parse_message:
            logging.info(f'Expected ticks per bar = {ticks_per_beat * BEAT_PER_BAR}, Getted: {total_ticks}')
            logging.info(f'Expected bar durarion: {bar_duration} seconds, Getted: {tick2second(total_ticks, ticks_per_beat, tempo)}')

    logging.info('Done! MIDI generated')
    return mid


def tokens_counter(bar):
    '''
    Description:
    Processes a list of tokens (bar) to count the number of ticks for each token
    
    Parameters:
    tokens (list): list of tokens

    Returns:
    tokens_vs_ticks_list (list): list of lists, where each sublist contains a token and the number of ticks for that token
    '''

    last_token = None
    counter = 0
    tokens_vs_ticks_list = []
    start_time = time.time()
    for i, token in enumerate(bar):
        if token != last_token and last_token is not None:
            tokens_vs_ticks_list.append([last_token, counter])
            last_token = token
            counter = 1
            if i == len(bar) - 1:
                tokens_vs_ticks_list.append([token, 1])
        else:
            counter += 1
            last_token = token
    logging.info(f'Processing time: {time.time() - start_time}')
    return tokens_vs_ticks_list


def play_midi(mid):
    '''
    Description:
    Plays a MIDI file

    Parameters:
    mid (mido.MidiFile): MIDI file to play
    '''
    # out_port = mido.open_output('loopMIDI Port 1')
    duration = 0
    for msg in mid.play():
        duration += msg.time
        # print(msg)
        # out_port.send(msg)
    logging.info(f'Duration: {duration}')





if __name__ == "__main__":

    print('\nMIDI Tokenization\n')

    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,datefmt="%H:%M:%S")

    MIDI_FOLDER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)))

    mid = mido.MidiFile(os.path.join(MIDI_FOLDER_PATH, 'examples/bass_example.MID'))
    logging.info(f'Loaded MIDI file: {mid.filename}')
    logging.info(f'Number of ticks per beat: {mid.ticks_per_beat}')

    bars = get_token_from_midi(mid)
    
    decoded_midi = generate_midi_from_tokens(bars)

    save_path = os.path.join(MIDI_FOLDER_PATH, 'output/Decoded_example_2.mid')
    decoded_midi.save(save_path)
    logging.info(f'Done, decoded MIDI file saved in: {save_path}')

    logging.info('Playing MIDI file...')
    play_midi(decoded_midi)
    logging.info('Done, MIDI file played\n')
