import re
import numpy as np
import pandas as pd
import pretty_midi
from mido import MidiFile, MidiTrack, Message, MetaMessage


BPM = 120
TICKS_PER_BEAT = 12 # resolution of the MIDI file
BEATS_PER_BAR = 4

VELOCITY_THRESHOLD = 80
MIN_VELOCITY = 40
NOTE_START_TOKEN = 'S'
VELOCITY_PIANO_TOKEN = 'p'
VELOCITY_FORTE_TOKEN = 'f'
SILENCE_TOKEN = 'O'
BCI_TOKENS = {'relaxed': 'R', 'concentrated': 'C'}
NOTE_SEPARATOR_TOKEN = '_'


DRUM_MIDI_DICT = {    
  36: 'Kick',
  38: 'Snare',
  42: 'Closed Hi-Hat',
  43: 'Floor Tom',
  44: 'Pedal Hi-Hat',
  46: 'Open Hi-Hat',
  47: 'Tom 2',
  48: 'Tom 1',
  49: 'Crash',
  51: 'Ride'
}



class Dictionary(object):
  def __init__(self):
    self.word2idx = {} # to keep track of the index of each token
    self.idx2word = [] # to keep track of the order of the tokens
    self.counter = [] # to keep track of the frequency of each token
    self.weights = [] # to keep track of the probability of each token

  def add_word(self, word):
    if not self.is_in_vocab(word):
      self.idx2word.append(word)
      self.word2idx[word] = len(self.idx2word) - 1
      self.counter.append(1)
    else:
      self.counter[self.word2idx[word]] += 1

  def compute_weights(self):
    total = sum(self.counter)
    self.weights = [freq/total for freq in self.counter]
  
  def save(self, path):
    with open(path, 'w') as f:
      for word in self.idx2word:
        f.write(word + '\n')

  def load(self, path):
    with open(path, 'r') as f:
      for line in f:
        self.add_word(line.strip())
  
  def is_in_vocab(self, word):
    return word in self.word2idx

  def __len__(self):
      return len(self.idx2word)



class PrettyMidiTokenizer(object):
    
  def __init__(self, midi_file_path = None, eeg = False):

    self.BPM = BPM
    self.BEATS_PER_BAR = BEATS_PER_BAR
    self.TICKS_PER_BEAT = TICKS_PER_BEAT

    self.BAR_LENGTH = self.BEATS_PER_BAR * self.TICKS_PER_BEAT
    self.SEQ_LENGTH = self.BAR_LENGTH * 4
    self.BEAT_DURATION = 60 / self.BPM
    self.BAR_DURATION = self.BEAT_DURATION * self.BEATS_PER_BAR
    self.TEMPO = int(self.BEAT_DURATION * 1000000)

    self.real_time_notes = []

    self.sequences = []
    self.notes_df = pd.DataFrame(columns=['pitch', 'velocity', 'start', 'end', 'bar'])
    self.VOCAB = Dictionary()
    self.VOCAB.add_word(SILENCE_TOKEN)
    if eeg:
      self.VOCAB.add_word(BCI_TOKENS['relaxed'])
      self.VOCAB.add_word(BCI_TOKENS['concentrated'])
    self.source_paths = []


    if midi_file_path is not None:
      self.source_paths.append(midi_file_path)
      self.sequences, self.notes_df = self.midi_to_tokens(self.source_paths[-1], update_vocab = True)


  def load_vocab(self, path):
    self.VOCAB = Dictionary()
    self.VOCAB.load(path)


  def convert_time_to_ticks(self, time, resolution = TICKS_PER_BEAT, bpm = BPM):
    pm = pretty_midi.PrettyMIDI(midi_file=None, resolution=resolution, initial_tempo=bpm)
    return pm.time_to_tick(time)


  def new_note(self, pitch, velocity, start, end, bar, convert_to_ticks = True):
    # NB: start and end are relative to the bar they are in
    if convert_to_ticks:
      start = self.convert_time_to_ticks(start - bar*self.BAR_DURATION)
      end = self.convert_time_to_ticks(end - bar*self.BAR_DURATION)

    new_note = {
      'pitch': pitch,
      'velocity': velocity,
      'start': start,
      'end': end,
      'bar': bar
    }   

    return pd.DataFrame([new_note])  


  def data_augmentation_shift(self, shifts):
    '''
    Shifts the sequences by a number of ticks to create new sequences.
    '''
    seq_len = len(self.sequences)
    
    for ticks in shifts:
      for i in range(seq_len):
        seq = np.roll(self.sequences[i], ticks)
        self.sequences.append(seq)



  def data_augmentation_transposition(self, transpositions):
    '''
    Transpose the sequences by a number of semitones to create new sequences.

    Parameters:
    - transpositions: a list of integers representing the number of semitones to transpose the sequences.

    NB: The transposition is done by adding the number of semitones to the pitch of each note in the sequence.
    '''

    for transposition in transpositions:
      for seq in self.sequences:

        new_seq = np.copy(seq)

        for i in range(len(new_seq)):

          token = new_seq[i]
          word = self.VOCAB.idx2word[token]

          # check if the token is a note
          if word != SILENCE_TOKEN and word != BCI_TOKENS['relax'] and word != BCI_TOKENS['concentrate']:

            # extract all the pitches from the token 
            pitches = re.findall('\d+', word) # NB: pitches is a string list

            # transpose each pitch in the token 
            for pitch in pitches:
              new_pitch = str(int(pitch) + transposition)
              word = word.replace(pitch, new_pitch)

            # add the new token to the vocabulary
            self.VOCAB.add_word(word) 

            # update the sequence with the new token
            new_seq[i] = self.VOCAB.word2idx[word]
        
        # update sequence with the new tokens
        self.sequences.append(new_seq)

  def midi_to_df(self, midi_path):
    '''
    Converts a MIDI file into a pandas dataframe.

    Parameters:
    - midi_path: the path to the MIDI file (str)

    Returns:
    - notes_df: a pandas dataframe containing the notes of the MIDI file

    NB: The MIDI file should contain only one instrument.
    '''

    pm = pretty_midi.PrettyMIDI(midi_path)
    instrument = pm.instruments[0]

    if pm.resolution != TICKS_PER_BEAT:
      raise Exception(f'The resolution of the MIDI file is {pm.resolution} instead of {self.TICKS_PER_BEAT}.')

    elif pm.get_tempo_changes()[1][0] != BPM:
      raise Exception(f'The tempo of the MIDI file is {pm.get_tempo_changes[1][0]} instead of {self.BPM}.')
    

    # Sort the notes by start time
    sorted_notes = sorted(instrument.notes, key=lambda note: note.start)

    # create a dataframe from the notes dictionary
    notes_df = pd.DataFrame(columns=['pitch', 'velocity', 'start', 'end', 'bar'])

    for note in sorted_notes:

      pitch = note.pitch
      velocity = note.velocity
      start = note.start
      end = note.end
      # step = start - prev_start
      duration = end - start
      bar = int(start // self.BAR_DURATION) # integer part of the division

      # split the note in two if it spans multiple bars
      if start + duration > (bar + 1) * self.BAR_DURATION: 

        # update the current note to end at the end of the bar and update its duration
        note = self.new_note(pitch, velocity, start, (bar + 1) * self.BAR_DURATION, bar)
        notes_df = pd.concat([notes_df, note], ignore_index=True)

        # create new note in the succeeding bar with the remaining duration
        note = self.new_note(pitch, velocity, (bar + 1) * self.BAR_DURATION, end, bar + 1)
        notes_df = pd.concat([notes_df, note], ignore_index=True)

      else:
        note = self.new_note(pitch, velocity, start, end, bar)
        notes_df = pd.concat([notes_df, note], ignore_index=True)

    return notes_df


  def note_to_string(self, tokens, pitch, velocity, start, end):
    '''
    Converts a note into a string token and adds it to the time serie of tokens.

    Parameters:
    - tokens: the time serie of tokens (np.array(dtype=object))
    - pitch: the pitch of the note (int)
    - velocity: the velocity of the note (int)
    - start: the start time of the note (int)
    - end: the end time of the note (int)
    '''


    if velocity < MIN_VELOCITY:
      return tokens
    elif velocity > VELOCITY_THRESHOLD:
      velocity_token = VELOCITY_FORTE_TOKEN
    else:
      velocity_token = VELOCITY_PIANO_TOKEN

    for i in range(start, end):
      if tokens[i] != SILENCE_TOKEN:
        tokens[i] += NOTE_SEPARATOR_TOKEN + pitch + velocity_token
      else:
        tokens[i] = pitch + velocity_token
      if i == start: 
        tokens[i] += NOTE_START_TOKEN

      if NOTE_SEPARATOR_TOKEN in tokens[i]:
        sorted_tokens = sorted(tokens[i].split(NOTE_SEPARATOR_TOKEN)) # sort the tokens by pitch
        tokens[i] = NOTE_SEPARATOR_TOKEN.join(sorted_tokens) # add the separator token between the tokens

    return tokens


  def midi_to_tokens(self, midi_path, update_vocab = False, update_sequences = False, emotion_token = None):

    '''
    Converts a MIDI file into a sequence of tokens.

    Parameters:
    - midi_path: the path to the MIDI file (str)
    - update_vocab: a boolean indicating whether to update the vocabulary (bool)
    - update_sequences: a boolean indicating whether to update the sequences (bool)
    - emotion_token: the token representing the emotion of the user (str)

    Returns:
    - sequences: a list of sequences of tokens (np.array)
    - notes_df: a pandas dataframe containing the notes of the MIDI file (columns: pitch, start, duration, velocity)
    '''

    notes_df = self.midi_to_df(midi_path)
    
    # split notes into bars and convert notes ticks into a time serie of tokens 
    bars_time_series = []
    bar_ids = notes_df['bar'].unique()
    for bar_id in bar_ids:
      bar_df = notes_df[notes_df['bar'] == bar_id]
      bar_df = bar_df.reset_index(drop=True)

      # convert note ticks into a time serie of strings 
      bar_time_serie = np.empty((self.BAR_LENGTH), dtype=object)
      bar_time_serie[:] = SILENCE_TOKEN
      for note_id in range(len(bar_df)):
        pitch = str(bar_df.loc[note_id, 'pitch'])
        start = bar_df.loc[note_id, 'start']
        end = bar_df.loc[note_id, 'end']
        velocity = bar_df.loc[note_id, 'velocity']
        bar_time_serie = self.note_to_string(bar_time_serie, pitch, velocity, start, end)
      bars_time_series.append(bar_time_serie)

    # flat bars and create vocabulary of unique tokens
    tokens = np.concatenate(bars_time_series)

    # update the vocabulary if necessary
    if update_vocab:
      for i in range(0, len(tokens)):
        self.VOCAB.add_word(tokens[i])

    # create the sequences of tokens for the model 
    sequences = []
    stop_index = len(tokens) - self.SEQ_LENGTH
    seq_len = self.SEQ_LENGTH
 
    if stop_index <= 0:
      stop_index = 1
      seq_len = len(tokens)
      
    for idx in range(0, stop_index, self.BAR_LENGTH):
      seq = tokens[idx:(idx+seq_len)].copy() # NB: copy is necessary to avoid modifying the original array

      # remove the last token add the BCI token at the beginning
      if emotion_token is not None:
        seq = np.concatenate(([emotion_token], seq[:-1])) 

      for i in range(len(seq)):
        if self.VOCAB.is_in_vocab(seq[i]):
          seq[i] = self.VOCAB.word2idx[seq[i]] 
        else:
          seq[i] = self.VOCAB.word2idx[SILENCE_TOKEN] 

      sequences.append(seq)

    # concatenate the sequences if necessary
    if update_sequences: 
      if len(self.sequences) > 0:
        for i in range(1, self.BEATS_PER_BAR):
          prev = self.sequences[-1][self.BAR_LENGTH*i:]
          next = sequences[0][:self.BAR_LENGTH*i]
          self.sequences.append(np.concatenate((prev, next)))
      self.sequences += sequences        

    return sequences, notes_df
  
  def tokens_to_notes_df(self, sequence, ticks_filter = 0):
    '''
    Converts a sequence of tokens into a pandas dataframe of notes.

    Parameters:
    - sequence: a sequence of tokens (np.array)
    - ticks_filter: the minimum duration of a note in ticks (int)

    Returns:
    - notes_df: a pandas dataframe containing the notes of the MIDI file (columns: pitch, start, duration, velocity)
    '''

    
    notes_df = pd.DataFrame(columns = ['pitch', 'start', 'duration', 'velocity'])
    prev_pitches = []

    # convert the sequence of tokens into a list of tokens and their duration and velocity
    for token_idx, token in enumerate(sequence):

      token_string = self.VOCAB.idx2word[token] # convert token id to string

      if NOTE_SEPARATOR_TOKEN in token_string:
          notes = token_string.split(NOTE_SEPARATOR_TOKEN)
      else:
          notes = [token_string]

      # get current pitches in the token
      pitches = re.findall('\d+', token_string)

      for note_string in notes:

        # check if the token is a silence token
        if SILENCE_TOKEN not in note_string:

          pitch = re.findall('\d+', note_string) [0] # re.findall returns a list
          velocity = 80 if VELOCITY_PIANO_TOKEN in note_string else 127
          start = token_idx if (NOTE_START_TOKEN in note_string or pitch not in prev_pitches) else None

          if start is not None:
            note = pd.DataFrame([{'pitch': pitch, 'start': start, 'duration': 1, 'velocity': velocity}])
            notes_df = pd.concat([notes_df, note], ignore_index=True)

          else:
            note_idx = notes_df.index[notes_df['pitch'] == pitch].tolist()[-1] # get the index of the last note with the same pitch
            notes_df.loc[note_idx, 'duration'] += 1

      prev_pitches = pitches

    # filter notes with duration less than ticks_filter
    notes_df = notes_df[notes_df['duration'] > ticks_filter]

    return notes_df
  
  def notes_df_to_midi_command_df(self, notes_df):
    '''
    Converts a pandas dataframe of notes into a pandas dataframe of commands.

    Parameters:
    - notes_df: a pandas dataframe containing the notes of the MIDI file (columns: pitch, start, duration, velocity)

    Returns:
    - command_df: a pandas dataframe containing the commands of the MIDI file (columns: pitch, start, note_on, velocity, dt)
    '''

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
    for idx in range(1, len(command_df)):
      command_df.loc[idx, 'dt'] = command_df.loc[idx, 'start'] - command_df.loc[idx - 1, 'start']

    return command_df
  
  def generate_track(self, command_df, bpm = BPM, resolution = TICKS_PER_BEAT, instrument_name = 'Acoustic Grand Piano'):  
    '''
    Generates a MIDI track from a pandas dataframe of commands.

    Parameters:
    - command_df: a pandas dataframe containing the commands of the MIDI file (columns: pitch, start, note_on, velocity, dt)
    - bpm: the tempo of the MIDI file (int)
    - resolution: the resolution of the MIDI file (int)

    Returns:
    - mid: a MIDI file (MIDO object)
    '''
    tempo = int((60 / bpm) * 1000000)
    mid = MidiFile(ticks_per_beat = resolution)
    track = MidiTrack()

    if instrument_name == 'Drum':
      channel = 9
      program = 0
    elif instrument_name == 'Bass':
      channel = 0
      program = 34
    else:
      channel = 0
      program = 0
    
    track.append(Message('program_change', program=program))
    track.append(MetaMessage('set_tempo', tempo=tempo))

    for i in range(len(command_df)):
        command = command_df.loc[i]
        pitch = int(command['pitch'])
        velocity = int(command['velocity'])
        dt = int(command['dt'])  # in ticks
        cmd = 'note_on' if command['note_on'] else 'note_off'
        track.append(Message(cmd, note=pitch, velocity=velocity, time=dt, channel=channel)) # NB: time from the previous message in ticks per beat

    mid.tracks.append(track)
    return mid

  def tokens_to_midi(self, sequence, out_file_path = None, ticks_filter = 0, instrument_name = 'Acoustic Grand Piano'):
    '''
    Converts a sequence of tokens into a MIDI file.

    Parameters:
    - sequence: a sequence of tokens (np.array)
    - out_file_path: the path to the output MIDI file (str)

    Returns:
    - mid: a MIDI file (MIDO object)

    '''
    notes_df = self.tokens_to_notes_df(sequence, ticks_filter)
    command_df = self.notes_df_to_midi_command_df(notes_df)
    mid = self.generate_track(command_df, instrument_name = instrument_name)
    
    if out_file_path is not None:
      mid.save(out_file_path)
      print(f'MIDI file saved at {out_file_path}')

    return mid

  def real_time_tokenization(self, notes, emotion_token = None):
    '''
    Converts a list of notes into a sequence of tokens in real-time.

    Parameters:
    - notes: a list of notes (list of dictionaries)
    - emotion_token: the token representing the emotion of the user (str)

    Returns:
    - tokens: a sequence of tokens (np.array)
    '''

    self.real_time_notes = []
    tokens = np.empty((self.BAR_LENGTH), dtype=object)
    tokens[:] = SILENCE_TOKEN
    notes[0]['dt'] = 0
    duration = 0

    # convert notes into tokens time serie
    for note_id, note in enumerate(notes):
      
      pitch = str(note['pitch'])
      velocity = note['velocity']
      dt = note['dt']
      duration += dt

      if duration > self.BAR_DURATION:
        break

      elif velocity > MIN_VELOCITY:
        start = self.convert_time_to_ticks(duration)
        step = 0
        for i in range (note_id+1, len(notes)):
          step += notes[i]['dt']
          if notes[i]['velocity'] == 0 and str(notes[i]['pitch']) == pitch: 
            step = self.convert_time_to_ticks(step)
            break
        
        end = int(start + step)

        self.real_time_notes.append({'pitch': pitch, 
                                     'velocity': velocity, 
                                     'start': start,
                                     'end': end})
        if end > self.BAR_LENGTH :
          end = self.BAR_LENGTH

        tokens = self.note_to_string(tokens, pitch, velocity, start, end)

    # add the BCI token at the beginning
    if emotion_token is not None:
      tokens = np.concatenate(([emotion_token], tokens[:-1]))
        
    # convert string tokens into integer tokens
    for i in range(len(tokens)):      
      if self.VOCAB.is_in_vocab(tokens[i]):
        tokens[i] = self.VOCAB.word2idx[tokens[i]] 
      else: 
        tokens[i] = self.VOCAB.word2idx[SILENCE_TOKEN] 

    return tokens


  





if __name__ == '__main__':
  
  vocab_path = 'APPLICATION/model/trained_models/model/input_vocab.txt'
  
  tok = PrettyMidiTokenizer()
  tok.load_vocab(vocab_path)

  midi_path = 'TCN/dataset/input/0_Drum_HardRock_EXCITED.mid'
  sequences, notes_df = tok.midi_to_tokens(midi_path)
  tokens = sequences[0]

  mid = tok.tokens_to_midi(tokens, out_file_path = 'test_in.mid', ticks_filter = 0, instrument_name = 'Drum')
