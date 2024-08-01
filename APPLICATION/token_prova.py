import re
from mido import MidiFile, MidiTrack, Message, MetaMessage
import pandas as pd
import numpy as np


VELOCITY_THRESHOLD = 80
MIN_VELOCITY = 40
NOTE_START_TOKEN = 'S'
VELOCITY_PIANO_TOKEN = 'p'
VELOCITY_FORTE_TOKEN = 'f'
SILENCE_TOKEN = 'O'
BCI_TOKENS = {'relaxed': 'R', 'concentrated': 'C'}
NOTE_SEPARATOR_TOKEN = '_'

def generate_track(command_df, resolution = 12, bpm = 120):  
 
    tempo = int((60 / bpm) * 1000000)
    mid = MidiFile(ticks_per_beat = resolution)
    track = MidiTrack()
    mid.tracks.append(track)
    track.append(MetaMessage('set_tempo', tempo=tempo))
    for idx, command in command_df.iterrows():
        pitch = int(command['pitch'])
        velocity = int(command['velocity'])
        dt = int(command['dt'])  # in ticks
        cmd = 'note_on' if command['note_on'] else 'note_off'
        track.append(Message(cmd, note=pitch, velocity=velocity, time=dt)) # NB: time from the previous message in ticks per beat
    mid.save('prova_77.mid') 


input = ['36fS_42fS', '36f_42f',  '36f_42f', '36f_42f', '36f_42f', '36f_42f', '42f', 'O', 'O', 'O', 'O', 'O', '42fS', '42f', '42f', '42f', '42f', '42f'] 
output = ['50fS', '50f', '50f', '43f', '43f', '43f',  '43f', '43f', '43f', '43f', '43f', '43f', '43f', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', '43f', '43f', 'O', 'O', 'O']

input = np.array(input)
output = np.array(output)

def detok(sequence, ticks_filter = 0):
    
    notes_df = pd.DataFrame(columns = ['pitch', 'start', 'duration', 'velocity'])
    prev_pitches = []

    # convert the sequence of tokens into a list of tokens and their duration and velocity
    for token_idx, token_string in enumerate(sequence):

        # token_string = self.VOCAB.idx2word[token] # convert token id to string

        if NOTE_SEPARATOR_TOKEN in token_string:
            notes = token_string.split(NOTE_SEPARATOR_TOKEN)
        else:
            notes = [token_string]

        pitches = re.findall('\d+', token_string)

        for note_string in notes:

            if SILENCE_TOKEN not in note_string:

                pitch = re.findall('\d+', note_string) [0]
                velocity = 80 if VELOCITY_PIANO_TOKEN in note_string else 127
                start = token_idx if (NOTE_START_TOKEN in note_string or pitch not in prev_pitches) else None

                if start is not None:
                    note = pd.DataFrame([{'pitch': pitch, 'start': start, 'duration': 1, 'velocity': velocity}])
                    notes_df = pd.concat([notes_df, note], ignore_index=True)

                else:
                    note_idx = notes_df.index[notes_df['pitch'] == pitch].tolist()[-1]
                    notes_df.loc[note_idx, 'duration'] += 1

        prev_pitches = pitches

    # filter notes with duration less than ticks_filter
    notes_df = notes_df[notes_df['duration'] > ticks_filter]

    # convert the notes into a list of commands (note_on, note_off, velocity, dt)
    command_df = pd.DataFrame(columns = ['pitch', 'start', 'note_on', 'velocity', 'dt'])
    for idx, note in notes_df.iterrows():
        pitch = note['pitch']
        start = note['start'] 
        duration = note['duration']
        velocity = note['velocity']
        end = start + duration
        cmd_on = pd.DataFrame([{'pitch': pitch, 'start': start, 'note_on': True, 'velocity': velocity, 'dt': 0}])
        cmd_off = pd.DataFrame([{'pitch': pitch, 'start': end, 'note_on': False, 'velocity': velocity, 'dt': 0}])
        command_df = pd.concat([command_df, cmd_on, cmd_off], ignore_index=True)

    # sort the commands by start time
    command_df = command_df.sort_values(by = 'start')
    command_df = command_df.reset_index(drop = True)

    # subtract the start time of the previous command from the start time of the current command 
    # to get the time interval between the two commands
    for idx, command in command_df.iterrows():
        if idx > 0:
            command_df.loc[idx, 'dt'] = command_df.loc[idx, 'start'] - command_df.loc[idx - 1, 'start']

    return command_df


if  __name__ == '__main__':
    commands = detok(output)
    generate_track(commands)
    print(commands)



                













       