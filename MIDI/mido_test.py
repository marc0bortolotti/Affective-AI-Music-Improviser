import mido
import os
from mido import Message, MidiFile, MidiTrack, MetaMessage, tick2second
import time

BPM = 120 
BEAT_PER_BAR = 4
TICKS_PER_BEAT = 32 # quantization of a beat
TICKS_PER_BAR = TICKS_PER_BEAT * BEAT_PER_BAR
BEAT_DURATION =  60/BPM # seconds
TEMPO = int(BEAT_DURATION * 1000000) # microseconds per beat
BAR_DURATION = BEAT_PER_BAR * BEAT_DURATION # seconds
DT = BEAT_DURATION/TICKS_PER_BEAT # seconds
SILENCE_TOKEN = 0
MIDI_FOLDER_PATH = 'C:/Users/Gianni/Desktop/MARCO/UNI/Magistrale/TESI/Code/MIDI'


def get_token_from_midi(mid):
    '''
    Description:
    Returns a list of measures, 
    where each measure is a list of tokens,
    representing notes and silences in the MIDI file
    
    Parameters:
    mid (mido.MidiFile): MIDI file to decode

    Returns:
    measures (list): list of measures, where each measure is a list of tokens
    '''
    bars=[]
    tokens = []
    for msg in mid.play():
        print(msg)
        if msg.type == 'note_on':
            silent_size = round(msg.time/DT)
            for i in range(silent_size):
                tokens.append(SILENCE_TOKEN)
        elif msg.type == 'note_off':
            note_size = round(msg.time/DT)
            for i in range(note_size):
                tokens.append(msg.note)
        if len(tokens) >= TICKS_PER_BAR:
            bars.append(tokens[:TICKS_PER_BAR])
            tokens = tokens[TICKS_PER_BAR:]
    return bars


def save_decoded_midi(bars):
    '''
    Description:
    Saves a MIDI file from a list of bars of tokens

    Parameters:
    bars (list): list of bars, where each measure is a list of tokens
    '''
    mid = MidiFile(ticks_per_beat = TICKS_PER_BEAT)
    track = MidiTrack()
    mid.tracks.append(track)
    track.append(MetaMessage('set_tempo', tempo=TEMPO)) # tempo in microseconds per beat

    for i, bar in enumerate(bars):
        print(f'BAR {i+1}')
        total_ticks = 0
        for note, ticks in bar: 
            if note != SILENCE_TOKEN:   
                track.append(Message('note_on', note=note, velocity=127, time=0)) # NB: time from the previous message in ticks per beat
                track.append(Message('note_off', note=note, velocity=127, time=ticks))
                print(f'note {note} for {ticks} ticks, {tick2second(ticks, TICKS_PER_BEAT, TEMPO)} seconds')
            else:
                track.append(Message('note_on', note=0, velocity=0, time=0))
                track.append(Message('note_off', note=0, velocity=0, time=ticks))
                print(f'silence for {ticks} ticks, {tick2second(ticks, TICKS_PER_BEAT, TEMPO)} seconds')
            total_ticks += ticks
        print(f'Expected ticks per bar = {TICKS_PER_BEAT * BEAT_PER_BAR}, Getted: {total_ticks}')
        print(f'Expected bar durarion: {BAR_DURATION} seconds, Getted: {tick2second(total_ticks, TICKS_PER_BEAT, TEMPO)}')

    save_path = os.path.join(MIDI_FOLDER_PATH, 'output/Decoded_example.mid')
    mid.save(save_path)
    print('\nDone, decoded MIDI file saved in: ', save_path)



def tokens_processing(tokens):
    '''
    Description:
    Processes a list of tokens to count the number of ticks for each token
    
    Parameters:
    tokens (list): list of tokens

    Returns:
    tokens_vs_ticks_list (list): list of lists, where each sublist contains a token and the number of ticks for that token
    '''

    last_token = None
    counter = 0
    tokens_vs_ticks_list = []
    start_time = time.time()
    for i, token in enumerate(tokens):
        if token != last_token and last_token is not None:
            tokens_vs_ticks_list.append([last_token, counter])
            last_token = token
            counter = 1
            if i == len(tokens) - 1:
                tokens_vs_ticks_list.append([token, 1])
        else:
            counter += 1
            last_token = token
    print(f'Processing time: {time.time() - start_time}')
    return tokens_vs_ticks_list






if __name__ == "__main__":

    # msg = mido.Message('note_on', note=60)
    # print(msg)
    # port = mido.open_output('Port Name')
    # port.send(msg)
    # with mido.open_input() as inport:
    #     for msg in inport:
    #         print(msg)

    mid = mido.MidiFile(os.path.join(MIDI_FOLDER_PATH, 'examples/bass_example.MID'))
    print(f'\nLoaded MIDI file: {mid.filename}')
    print(f'Number of ticks per beat: {mid.ticks_per_beat}')

    print('\nExtracting tokens from MIDI file...')
    bars = get_token_from_midi(mid)
    print(f'Number of bars: {len(bars)}')

    bars_processed = []
    print('\nProcessing tokens...')
    for tokens in bars:
        tokens_vs_ticks_list = tokens_processing(tokens)
        bars_processed.append(tokens_vs_ticks_list)
    print(f'Done, number of measures processed: {len(bars_processed)}')
    for i, bar in enumerate(bars_processed):
        print(f'BAR {i}: {bar}')
        
    print('\nSaving decoded MIDI...')
    save_decoded_midi(bars_processed)
    print()


