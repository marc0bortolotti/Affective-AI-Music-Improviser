import re
import numpy as np
import pandas as pd
import pretty_midi
from mido import MidiFile, MidiTrack, Message, MetaMessage
from rapidfuzz import process

BPM = 120
TICKS_PER_BEAT = 4 # resolution of the MIDI file
BEATS_PER_BAR = 4

VELOCITY_TOKENS = {40:'pp', 60:'p', 90:'f', 110:'ff'} 
NOTE_START_TOKEN = 'S' # 'S'
SILENCE_TOKEN = 'O'
BCI_TOKENS = {0: 'R', 1: 'C'} # relaxed, concentrated
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
  '''
  A class to keep track of the vocabulary of tokens.

  Attributes:
  - word2idx: a dictionary to keep track of the index of each token (dict)
  - idx2word: a list to keep track of the order of the tokens (list)
  - counter: a list to keep track of the frequency of each token (list)
  - weights: a list to keep track of the probability of each token (list)

  Methods:
  - add_word(word): adds a word to the vocabulary
  - compute_weights(): computes the probability of each token
  - save(path): saves the vocabulary to a text file
  - load(path): loads the vocabulary from a text file
  - is_in_vocab(word): checks if a word is in the vocabulary
  - __len__(): returns the length of the vocabulary
  '''

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
    
  def remove_word(self, word):
    if self.is_in_vocab(word):
      idx = self.word2idx[word]
      self.idx2word.pop(idx)
      self.counter.pop(idx)
      if idx < len(self.weights):
        self.weights.pop(idx)
        self.compute_weights()
      self.word2idx = {word: idx for idx, word in enumerate(self.idx2word)}

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
  '''
  A class to tokenize MIDI files into sequences of tokens and vice versa.

  Attributes:
  - BPM: the tempo of the MIDI file (int)
  - BEATS_PER_BAR: the number of beats per bar (int)
  - TICKS_PER_BEAT: the resolution of the MIDI file (int)
  - BAR_LENGTH: the length of a bar in ticks (int)
  - SEQ_LENGTH: the length of a sequence in ticks (int)
  - BEAT_DURATION: the duration of a beat in seconds (float)
  - BAR_DURATION: the duration of a bar in seconds (float)
  - TEMPO: the tempo of the MIDI file in microseconds per beat (int)
  - sequences: a list of sequences of tokens (np.array)
  - notes_df: a pandas dataframe containing the notes of the MIDI file (columns: pitch, start, duration, velocity)
  - VOCAB: a Dictionary object containing the vocabulary of tokens
  - source_paths: a list of paths to the source MIDI files (str)
  
  Methods:
  - load_vocab(path): loads a vocabulary from a text file
  - convert_time_to_ticks(time, resolution, bpm): converts a time in seconds to ticks
  - new_note(pitch, velocity, start, end, bar, convert_to_ticks): creates a new note
  - midi_to_df(midi_path): converts a MIDI file into a pandas dataframe
  - note_to_string(tokens, pitch, velocity, start, end): converts a note into a string token
  - midi_to_tokens(midi_path, update_vocab, update_sequences, emotion_token): converts a MIDI file into a sequence of tokens
  - tokens_to_notes_df(sequence, ticks_filter): converts a sequence of tokens into a pandas dataframe of notes
  - notes_df_to_midi_command_df(notes_df): converts a pandas dataframe of notes into a pandas dataframe of commands
  - generate_track(command_df, bpm, resolution, instrument_name): generates a MIDI track from a pandas dataframe of commands
  - tokens_to_midi(sequence, out_file_path, ticks_filter, instrument_name): converts a sequence of tokens into a MIDI file
  - real_time_tokenization(notes, emotion_token): converts a list of notes into a sequence of tokens in real-time
  '''
    
  def __init__(self, BPM = BPM, BEATS_PER_BAR = BEATS_PER_BAR, TICKS_PER_BEAT = TICKS_PER_BEAT):

    self.BPM = BPM
    self.BEATS_PER_BAR = BEATS_PER_BAR
    self.TICKS_PER_BEAT = TICKS_PER_BEAT

    self.BAR_LENGTH = self.BEATS_PER_BAR * self.TICKS_PER_BEAT
    self.SEQ_LENGTH = self.BAR_LENGTH * 4
    self.BEAT_DURATION = 60 / self.BPM
    self.BAR_DURATION = self.BEAT_DURATION * self.BEATS_PER_BAR
    self.BAR_DURATION_IN_TICKS = self.convert_time_to_ticks(self.BAR_DURATION)
    self.TEMPO = int(self.BEAT_DURATION * 1000000)

    self.sequences = []
    self.source_paths = []
    self.notes_df = pd.DataFrame(columns=['pitch', 'velocity', 'start', 'end', 'bar'])
    self.VOCAB = Dictionary()
    self.VOCAB.add_word(SILENCE_TOKEN)

  def load_vocab(self, path):
    self.VOCAB = Dictionary()
    self.VOCAB.load(path)

  def convert_time_to_ticks(self, time):
    beat_duration = 60 / self.BPM
    tick_duration = beat_duration / self.TICKS_PER_BEAT
    ticks = int(time / tick_duration)
    return ticks

  def compute_velocity_token(self, target):
    keys = list(VELOCITY_TOKENS.keys())
    nearest_key = min(keys, key=lambda k: abs(k - target))
    return VELOCITY_TOKENS[nearest_key]
  
  def new_note(self, pitch, velocity, start, end, bar):
    new_note = {
      'pitch': pitch,
      'velocity': velocity,
      'start': start,
      'end': end,
      'bar': bar
    }   
    return pd.DataFrame([new_note])  

  def midi_to_df(self, midi_path, instrument = None):
    '''
    Converts a MIDI file into a pandas dataframe.

    Parameters:
    - midi_path: the path to the MIDI file (str)

    Returns:
    - notes_df: a pandas dataframe containing the notes of the MIDI file

    NB: The MIDI file should contain only one instrument.
    '''

    pm = pretty_midi.PrettyMIDI(midi_path)

    # if pm.resolution != TICKS_PER_BEAT:
    #   raise Exception(f'The resolution of the MIDI file is {pm.resolution} instead of {self.TICKS_PER_BEAT}.')

    # elif pm.get_tempo_changes()[1][0] != BPM:
    #   raise Exception(f'The tempo of the MIDI file is {pm.get_tempo_changes[1][0]} instead of {self.BPM}.')
    

    # Sort the notes by start time
    sorted_notes = sorted(pm.instruments[0].notes, key=lambda note: note.start)

    # create a dataframe from the notes dictionary
    notes_df = pd.DataFrame(columns=['pitch', 'velocity', 'start', 'end', 'bar'])
    
    for idx, note in enumerate(sorted_notes):

      print(f"Processing note: {idx}/{len(sorted_notes)}", end="\r")
      pitch = note.pitch
      velocity = note.velocity
      start = self.convert_time_to_ticks(note.start)
      end = self.convert_time_to_ticks(note.end)
      if instrument == 'drum':
        end = start + 1

      # bar is the integer part of the start time divided by the duration of a bar
      start_bar = start // self.BAR_DURATION_IN_TICKS
      end_bar = end // self.BAR_DURATION_IN_TICKS

      # update the start and end times to be with index of the bar
      start -= start_bar * self.BAR_DURATION_IN_TICKS
      end -= end_bar * self.BAR_DURATION_IN_TICKS

      # split the note if it spans multiple bars
      if start_bar != end_bar: 
        for bar in range(start_bar, end_bar):
          note = None
          if bar == start_bar: 
            if self.BAR_DURATION_IN_TICKS - start > 5: # filter notes with duration less than 5 ticks
              note = self.new_note(pitch, velocity, start, self.BAR_DURATION_IN_TICKS, bar)
          elif bar == end_bar: 
            if end > 5: # filter notes with duration less than 5 ticks  
              note = self.new_note(pitch, velocity, 0, end, bar)
          else:
            note = self.new_note(pitch, velocity, 0, self.BAR_DURATION_IN_TICKS, bar)
          
          if note is not None:
            notes_df = pd.concat([notes_df, note], ignore_index=True)
          
      else:
        bar = start_bar
        note = self.new_note(pitch, velocity, start, end, bar)
        notes_df = pd.concat([notes_df, note], ignore_index=True)
      
      # sort the notes by bar and start time
      notes_df = notes_df.sort_values(by = ['bar', 'start'])
      notes_df = notes_df.reset_index(drop=True)
    
    return notes_df


  def note_to_string(self, tokens, pitch, velocity, start, end, rhythm = False, single_notes = False):
    '''
    Converts a note into a string token and adds it to the time serie of tokens.

    Parameters:
    - tokens: the time serie of tokens (np.array(dtype=object))
    - pitch: the pitch of the note (int)
    - velocity: the velocity of the note (int)
    - start: the start time of the note (int)
    - end: the end time of the note (int)
    - rhythm: a boolean indicating whether to consider only the rhythm of the notes (bool)
    - single_notes: a boolean indicating whether to keep only one note per token (bool)
    '''

    velocity_token = self.compute_velocity_token(velocity)

    if rhythm:
      pitch = '1'
      end = start + 1
    else:
      pitch = str(pitch)

    for i in range(start, end):
      if pitch in tokens[i]:
        break
      elif tokens[i] != SILENCE_TOKEN:
        tokens[i] += NOTE_SEPARATOR_TOKEN + pitch + velocity_token
      else:
        tokens[i] = pitch + velocity_token
      if i == start and not rhythm: 
        tokens[i] += NOTE_START_TOKEN

      # sort the tokens by pitch to avoid redundancy
      if NOTE_SEPARATOR_TOKEN in tokens[i]:
        sorted_tokens = sorted(tokens[i].split(NOTE_SEPARATOR_TOKEN)) # sort the tokens by pitch
        if single_notes:
          tokens[i] = sorted_tokens[-1] # keep only the note with the highest pitch
        else:
          tokens[i] = NOTE_SEPARATOR_TOKEN.join(sorted_tokens) # add the separator token between the tokens

    return tokens
  
  def generate_sequences(self, token_sequence, emotion_token = None, update_sequences = False):

    if emotion_token is not None:
      self.VOCAB.add_word(BCI_TOKENS[0])
      self.VOCAB.add_word(BCI_TOKENS[1])

    sequences = []
    step = self.BAR_LENGTH
    stop_index = len(token_sequence) - self.SEQ_LENGTH
    
    if stop_index <= 0:
      stop_index = 1

    for i in range(0, stop_index, step):

      if i + self.SEQ_LENGTH > len(token_sequence):
        # pad the sequence with silence tokens
        silence_token_id = self.VOCAB.word2idx[SILENCE_TOKEN]
        sequence = np.concatenate((token_sequence[i:], np.array([silence_token_id] * (self.SEQ_LENGTH - len(token_sequence)))))
      else:
        sequence = token_sequence[i:i+self.SEQ_LENGTH]

      if emotion_token is not None:
        emotion_token_id = self.VOCAB.word2idx[emotion_token]
        sequence = np.concatenate(([emotion_token_id], sequence[:-1]))
        
      sequences.append(sequence)

    # concatenate the sequences if necessary
    if update_sequences: 
      self.sequences += sequences 

    return sequences

  def midi_to_tokens(self, midi_path, update_vocab = False, rhythm = False, single_notes = False, max_len = None, instrument = None):

    '''
    Converts a MIDI file into a sequence of tokens.

    Parameters:
    - midi_path: the path to the MIDI file (str)
    - update_vocab: a boolean indicating whether to update the vocabulary (bool)
    - update_sequences: a boolean indicating whether to update the sequences (bool)
    - emotion_token: the token representing the emotion of the user (str)
    - rhythm: a boolean indicating whether to consider only the rhythm of the notes (bool)
    - single_notes: a boolean indicating whether to keep only one note per token (bool)
    - max_len: the maximum length of the sequence (int)

    Returns:
    - sequences: a list of sequences of tokens (np.array)
    - notes_df: a pandas dataframe containing the notes of the MIDI file (columns: pitch, start, duration, velocity)
    '''

    filename = midi_path.split('\\')[-1]
    print(f'Processing MIDI file: {filename}')

    pm = pretty_midi.PrettyMIDI(midi_path)
    sorted_notes = sorted(pm.instruments[0].notes, key=lambda note: note.start)
    
    # create a sequence of string tokens
    last_note = sorted_notes[-1]

    if max_len is None:
      tokens_len = self.convert_time_to_ticks(last_note.end)
    else:
      tokens_len = max_len

    token_sequence = np.empty((tokens_len), dtype=object)
    token_sequence[:] = SILENCE_TOKEN
    for idx, note in enumerate(sorted_notes):
      print(f"Processing note: {idx+1}/{len(sorted_notes)}", end="\r")
      pitch = note.pitch
      velocity = note.velocity
      start = self.convert_time_to_ticks(note.start)
      end = self.convert_time_to_ticks(note.end)
      if instrument == 'Drum':
        end = start + 1
      if start >= tokens_len or end >= tokens_len:
        break
      token_sequence = self.note_to_string(token_sequence, pitch, velocity, start, end, rhythm, single_notes)
      
    # update the vocabulary if necessary
    if update_vocab:
      for i in range(0, len(token_sequence)):
        self.VOCAB.add_word(token_sequence[i])

    # convert string tokens into integer tokens
    for i in range(len(token_sequence)):
      if self.VOCAB.is_in_vocab(token_sequence[i]):
        token_sequence[i] = self.VOCAB.word2idx[token_sequence[i]] 
      else: 
        token_sequence[i] = self.VOCAB.word2idx[SILENCE_TOKEN]

    print('\nDone!')       

    return token_sequence
  
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
          velocity = [key for key in VELOCITY_TOKENS.keys() if VELOCITY_TOKENS[key] in note_string] [0]
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

    # sort the commands by start time and then sort for note_off before note_on (when start times are equal)
    command_df = command_df.sort_values(by = ['start', 'note_on'])
    command_df = command_df.reset_index(drop = True)

    # subtract the start time of the previous command from the start time of the current command 
    # to get the time interval between the two commands
    for idx in range(1, len(command_df)):
      command_df.loc[idx, 'dt'] = command_df.loc[idx, 'start'] - command_df.loc[idx - 1, 'start']

    return command_df
  
  def generate_track(self, command_df, bpm = BPM, resolution = TICKS_PER_BEAT, instrument_name = 'Acoustic Grand Piano', emotion_token = None):  
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

    if emotion_token == BCI_TOKENS[0]:
      track.append(Message('control_change', control=64, value=127, time=0))

    for i in range(len(command_df)):
        command = command_df.loc[i]
        pitch = int(command['pitch'])
        velocity = int(command['velocity'])
        dt = int(command['dt'])  # in ticks
        cmd = 'note_on' if command['note_on'] else 'note_off'
        track.append(Message(cmd, note=pitch, velocity=velocity, time=dt, channel=channel)) # NB: time from the previous message in ticks per beat

    if emotion_token == BCI_TOKENS[0]:
      track.append(Message('control_change', control=64, value=0, time=0))

    mid.tracks.append(track)
    return mid

  def tokens_to_midi(self, tokens, out_file_path = None, ticks_filter = 0, instrument_name = 'Acoustic Grand Piano', emotion_token = None):
    '''
    Converts a sequence of tokens into a MIDI file.

    Parameters:
    - tokens: a sequence of tokens (np.array)
    - out_file_path: the path to the output MIDI file (str)

    Returns:
    - mid: a MIDI file (MIDO object)

    '''
    # convert the sequence of tokens into a pandas dataframe of notes
    notes_df = self.tokens_to_notes_df(tokens, ticks_filter)

    # convert the notes into a list of commands (note_on, note_off, velocity, dt)
    command_df = self.notes_df_to_midi_command_df(notes_df)

    # generate a MIDI track from the commands
    mid = self.generate_track(command_df, instrument_name = instrument_name, emotion_token = emotion_token)
    
    if out_file_path is not None:
      mid.save(out_file_path)
      print(f'MIDI file saved at {out_file_path}')

    return mid
  
  def update_sequences(self, count_th = None):
    if count_th is not None:

      original_vocab = self.VOCAB
      original_vocab.compute_weights()

      # Remove tokens that appear less than # times in the dataset
      for idx, count in enumerate(original_vocab.counter):
          if original_vocab.idx2word[idx] in BCI_TOKENS.values(): 
              pass
          elif count < count_th:
              original_vocab.counter[idx] = 0

      # Create a new vocab with the updated tokens
      updated_vocab = Dictionary()
      for word in original_vocab.word2idx.keys():
          if original_vocab.counter[original_vocab.word2idx[word]] > 0:
              updated_vocab.add_word(word)

      # Update the sequences with the new vocab
      for seq in self.sequences:
          for i, tok in enumerate(seq):
              if original_vocab.counter[tok] == 0 and original_vocab.idx2word[tok] not in BCI_TOKENS.values():
                  
                  closest_token_string = process.extractOne(original_vocab.idx2word[tok], updated_vocab.word2idx.keys())
                  closest_token_string = closest_token_string[0] if closest_token_string else SILENCE_TOKEN
                  seq[i] = updated_vocab.word2idx[closest_token_string]
                  updated_vocab.add_word(closest_token_string)
              else:
                  word = original_vocab.idx2word[tok]
                  seq[i] = updated_vocab.word2idx[word]
                  updated_vocab.add_word(word)
      
      self.VOCAB = updated_vocab
      self.VOCAB.compute_weights()

      # Verify that the vocab was updated
      print(f'Initial silence token weigth: {original_vocab.weights[original_vocab.word2idx[SILENCE_TOKEN]]}')
      print(f'Final silence token weigth: {self.VOCAB.weights[self.VOCAB.word2idx[SILENCE_TOKEN]]}')
      print(f'Inintial number of tokens: {len(original_vocab)}')
      print(f'Final number of tokens: {len(self.VOCAB)}')

  def real_time_tokenization(self, notes, emotion_token = None, rhythm = True):
    '''
    Converts a list of notes into a sequence of tokens in real-time.

    Parameters:
    - notes: a list of notes (list of dictionaries)
    - emotion_token: the token representing the emotion of the user (str)

    Returns:
    - tokens: a sequence of tokens (np.array)
    '''

    tokens = np.empty((self.BAR_LENGTH), dtype=object)
    tokens[:] = SILENCE_TOKEN
    notes[0]['dt'] = 0
    duration = 0

    # convert notes into tokens time serie
    for note_id, note in enumerate(notes):
      
      pitch = note['pitch']
      velocity = note['velocity']
      dt = note['dt']
      duration += dt
      start = self.convert_time_to_ticks(duration)

      if start >= self.BAR_LENGTH or velocity < 40:
        break

      else:
        step = 0
        if not rhythm:
          for i in range (note_id+1, len(notes)):
            step += notes[i]['dt']
            if notes[i]['velocity'] == 0 and notes[i]['pitch'] == pitch: 
              step = self.convert_time_to_ticks(step)
              break
          end = int(start + step)
        else:
          end = start + 1

        if end > self.BAR_LENGTH :
          end = self.BAR_LENGTH 

        tokens = self.note_to_string(tokens, pitch, velocity, start, end, rhythm=True, single_notes=False)

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
  import os
  file_path = os.path.join(os.path.dirname(__file__), 'tok_test/melody_RELAXED.mid')
  save_path = os.path.join(os.path.dirname(__file__), 'tok_test/melody_RELAXED.mid')

  tok = PrettyMidiTokenizer()
  tokens = tok.midi_to_tokens(file_path, update_vocab=True) [:120]

  mid = tok.tokens_to_midi(tokens, out_file_path = save_path, ticks_filter = 0, instrument_name = 'Piano')
