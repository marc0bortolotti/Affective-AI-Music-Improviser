import collections
import numpy as np
import pandas as pd
import pretty_midi

BPM = 120
TICKS_PER_BEAT = 12 # resolution of the MIDI file
BEATS_PER_BAR = 4

VELOCITY_THRESHOLD = 80
NOTE_START_TOKEN = 'S'
VELOCITY_PIANO_TOKEN = 'p'
VELOCITY_FORTE_TOKEN = 'f'
SILENCE_TOKEN = 'O'
BCI_TOKENS = {'relax': 'R', 'concentrate': 'C'}
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
    if word not in self.word2idx:
      self.idx2word.append(word)
      self.word2idx[word] = len(self.idx2word) - 1
      self.counter.append(1)
    else:
      self.counter[self.word2idx[word]] += 1
    return self.word2idx[word]

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
    
  def __init__(self, midi_file_path = None):

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


    if midi_file_path is not None:

      self.source_path = midi_file_path
      self.source_pm = pretty_midi.PrettyMIDI(self.source_path)

      if self.source_pm.resolution != TICKS_PER_BEAT:
        raise Exception('The resolution of the MIDI file is {self.pm.resolution} instead of {TICKS_PER_BEAT}.')

      elif self.source_pm.get_tempo_changes()[1][0] != BPM:
        raise Exception('The tempo of the MIDI file is {self.pm.get_tempo_changes[1][0]} instead of {BPM}.')
    
      else: 
        self.sequences, self.notes_df = self.midi_to_tokens(self.source_path, update_vocab = True)
    
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
    return new_note


  def append_note_to_notes_dict(self, notes, note):
    for key, value in note.items():
      notes[key].append(value)    


  def midi_to_tokens(self, midi_path, update_vocab = False, update_sequences = False, emotion_token = None):

    pm = pretty_midi.PrettyMIDI(midi_path)
    instrument = pm.instruments[0]

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
        notes_df = notes_df.append(note, ignore_index=True)

        # create new note in the succeeding bar with the remaining duration
        note = self.new_note(pitch, velocity, (bar + 1) * self.BAR_DURATION, end, bar + 1)
        notes_df = notes_df.append(note, ignore_index=True)

      else:
        note = self.new_note(pitch, velocity, start, end, bar)
        notes_df = notes_df.append(note, ignore_index=True)

    # split notes into bars and convert notes ticks into a time serie of tokens 
    bars_time_series = []
    bar_ids = notes_df['bar'].unique()
    for bar_id in bar_ids:
      bar_df = notes_df[notes_df['bar'] == bar_id]
      bar_df = bar_df.reset_index(drop=True)

      # convert note ticks into a time serie of strings 
      bar_time_serie = np.empty((self.BAR_LENGTH), dtype=object)
      bar_time_serie[:] = SILENCE_TOKEN
      for idx in range(len(bar_df)):
        pitch = str(bar_df.loc[idx, 'pitch'])
        start = bar_df.loc[idx, 'start']
        end = bar_df.loc[idx, 'end']
        velocity = bar_df.loc[idx, 'velocity']

        if velocity < 40:
          continue
        elif velocity > VELOCITY_THRESHOLD:
          velocity_token = VELOCITY_FORTE_TOKEN
        else:
          velocity_token = VELOCITY_PIANO_TOKEN

        for i in range(start, end):
          if bar_time_serie[i] != SILENCE_TOKEN and prev_pitch is not None and prev_pitch != pitch:
            bar_time_serie[i] += NOTE_SEPARATOR_TOKEN + pitch + velocity_token
          else:
            bar_time_serie[i] = pitch + velocity_token
          if i == start: 
            bar_time_serie[i] += NOTE_START_TOKEN

        prev_pitch = pitch

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

      # remove the last token and add the BCI token at the beginning
      if emotion_token is not None:
        seq = np.concatenate(([emotion_token], seq[:-1])) 
        self.VOCAB.add_word(emotion_token)

      for i in range(len(seq)):
        seq[i] = self.VOCAB.word2idx[seq[i]] 

      sequences.append(seq)

    # concatenate the sequences if necessary
    if update_sequences:
      self.sequences += sequences        

    return sequences, notes_df



  def tokens_to_midi(self, sequence, out_file_path = None, instrument_name = 'Acoustic Bass'):

    '''
    NB: works only for monophonic sequences of tokens. 
        So it doesn't work for drum sequences or chords.
    '''

    last_pitch = None
    last_velocity = None
    counter = 1
    pitch_ticks_velocity = []

    # convert the sequence of tokens into a list of tokens and their duration
    for i, token in enumerate(sequence):

      token_string = self.VOCAB.idx2word[token] # convert token id to string

      if NOTE_SEPARATOR_TOKEN in token_string:
        token_string = token_string.split(NOTE_SEPARATOR_TOKEN)[1]

      # extract pitch and velocity from the token string
      if VELOCITY_PIANO_TOKEN in token_string:
        velocity = 90
        token_string = token_string.replace(VELOCITY_PIANO_TOKEN, '') # remove the velocity token
      elif VELOCITY_FORTE_TOKEN in token_string:
        velocity = 127
        token_string = token_string.replace(VELOCITY_FORTE_TOKEN, '') # remove the velocity token
      else:
        velocity = 0

      # extract pitch from the token string
      note_start = False
      if token_string.isdigit():
        pitch = int(token_string)
      elif NOTE_START_TOKEN in token_string:
        token_string = token_string.replace(NOTE_START_TOKEN, '') # remove the note start token
        pitch = int(token_string)
        note_start = True
      else: # silence token or BCI token
        pitch = 0

      # update the pitch, duration and velocity lists
      if last_pitch != None and last_velocity != None:
        # if new note is started, add the previous pitch, duration and velocity to the list
        if  pitch != last_pitch or note_start:

          # filter notes with a duration of less than # ticks
          if counter > 3:
            pitch_ticks_velocity.append([last_pitch, counter, last_velocity])
          else:
            pitch_ticks_velocity.append([0, counter, 0])
          
          counter = 1
          last_pitch = pitch
          if i == len(sequence) - 1:
            pitch_ticks_velocity.append([pitch, 1, velocity])
        
        # if the pitch is the same or the note is sustained, increment the duration of the note
        else:
          counter += 1
      
      last_velocity = velocity  
      last_pitch = pitch

    # if necessary, generate midi file
    if out_file_path is not None:
      pm = pretty_midi.PrettyMIDI(midi_file=None, resolution=self.TICKS_PER_BEAT, initial_tempo=self.BPM)
      program = pretty_midi.instrument_name_to_program(instrument_name)
      instrument = pretty_midi.Instrument(program=program)
      prev_end = 0
      for note in pitch_ticks_velocity:
        (pitch, ticks, velocity) = note
        duration = pm.tick_to_time(ticks) # convert ticks to time
        start = prev_end
        end = start + duration
        instrument.notes.append(pretty_midi.Note(velocity=velocity, pitch=pitch, start=start, end=end))
        
        prev_end = end

      pm.instruments.append(instrument)
      if out_file_path is not None:
        pm.write(out_file_path)
        print(f'MIDI file saved at {out_file_path}')

    return pitch_ticks_velocity
  


  def real_time_tokenization(self, notes, emotion_token):

    self.real_time_notes = []
    tokens = np.empty((self.BAR_LENGTH), dtype=object)
    tokens[:] = SILENCE_TOKEN
    if len(notes) > 0:
      notes[0]['dt'] = 0
    duration = 0

    # convert notes into tokens time serie
    for note_idx, note in enumerate(notes):
      
      pitch = str(note['pitch'])
      velocity = note['velocity']
      dt = note['dt']
      duration += dt

      if duration > self.BAR_DURATION:
        break

      elif velocity > 40:
        start = self.convert_time_to_ticks(duration)
        step = 0
        for i in range (note_idx+1, len(notes)):
          step += notes[i]['dt']
          if notes[i]['velocity'] == 0 and str(notes[i]['pitch']) == pitch: 
            step = self.convert_time_to_ticks(step)
            break
        
        end = int(start + step)
      # start = note['start']
      # end = note['end']

      if velocity > 40:
        self.real_time_notes.append({'pitch': pitch, 
                                     'velocity': velocity, 
                                     'start': start,
                                     'end': end})

        if end > self.BAR_LENGTH :
          end = self.BAR_LENGTH
        
        if velocity < VELOCITY_THRESHOLD:
          velocity_token = VELOCITY_PIANO_TOKEN
        else:
          velocity_token = VELOCITY_FORTE_TOKEN

        for i in range(start, end):
          if tokens[i] != SILENCE_TOKEN and prev_pitch is not None and prev_pitch != pitch:
            tokens[i] += NOTE_SEPARATOR_TOKEN + pitch + velocity_token
          else:
            tokens[i] = pitch + velocity_token
          if i == start: 
            tokens[i] += NOTE_START_TOKEN
        
        prev_pitch = pitch
        
    # convert string tokens into integer tokens
    for i in range(len(tokens)):
      if i == 0:
        tokens[i] = emotion_token
      
      if not self.VOCAB.is_in_vocab(tokens[i]):
        tokens[i] = self.VOCAB.word2idx[SILENCE_TOKEN] 
      else: 
        tokens[i] = self.VOCAB.word2idx[tokens[i]] 


    return tokens


  
  






if __name__ == '__main__':
  import rtmidi


  # Tokenization of a MIDI file
  tok = PrettyMidiTokenizer('TCN/dataset/output/bass.mid')
  sample = tok.sequences[0]
  note_ticks_velocity = tok.tokens_to_midi(sample, 'MIDI/PRETTY_MIDI/decoded_sample.mid')


  # Real-time tokenization
  midi_in = rtmidi.MidiIn()
  available_ports = midi_in.get_port_count()
  print(f'Available MIDI ports: {midi_in.get_ports()}')

  midi_in.open_port(3)
  while True:
    msg_and_dt = midi_in.get_message()
    if msg_and_dt:
      (msg, dt) = msg_and_dt
      command, note, velocity = msg
      command = hex(command)
