import collections
import numpy as np
import pandas as pd
import pretty_midi


BPM = 120
TICKS_PER_BEAT = 12 # resolution of the MIDI file
BEATS_PER_BAR = 4


VELOCITY_RANGES = {'p': (0, 64), 'f': (65, 127)}
NOTE_START_TOKEN = 'S'
SILENCE_TOKEN = 'O'
BCI_TOKEN = 'BCI'




class Dictionary(object):
  def __init__(self):
      self.input = input
      self.word2idx = {}
      self.idx2word = []

  def add_word(self, word):
      if word not in self.word2idx:
          self.idx2word.append(word)
          self.word2idx[word] = len(self.idx2word) - 1
      return self.word2idx[word]

  def __len__(self):
      return len(self.idx2word)






class PrettyMidiTokenizer(object):
    
  def __init__(self, midi_file_path):


    self.midi_file_path = midi_file_path

    self.BPM = BPM
    self.BEATS_PER_BAR = BEATS_PER_BAR
    self.TICKS_PER_BEAT = TICKS_PER_BEAT

    self.BAR_LENGTH = self.BEATS_PER_BAR * self.TICKS_PER_BEAT
    self.SEQ_LENGTH = self.BAR_LENGTH * 4
    self.BAR_DURATION = (60 / self.BPM) * self.BEATS_PER_BAR

    self.VOCAB = Dictionary()
    self.sequences = []
    self.notes_df = pd.DataFrame()
    self.num_bars = 0


    self.pm = pretty_midi.PrettyMIDI(midi_file_path)

    if self.pm.resolution != TICKS_PER_BEAT:
      raise Exception('The resolution of the MIDI file is {self.pm.resolution} instead of {TICKS_PER_BEAT}.')

    elif self.pm.get_tempo_changes()[1][0] != BPM:
      raise Exception('The tempo of the MIDI file is {self.pm.get_tempo_changes[1][0]} instead of {BPM}.')
  
    else:
      self.midi_to_tokens()
      


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



  def midi_to_tokens(self):

    pm = self.pm
    instrument = pm.instruments[0]
    notes = collections.defaultdict(list) # Dictionary with values as list

    # Sort the notes by start time
    sorted_notes = sorted(instrument.notes, key=lambda note: note.start)

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
        self.append_note_to_notes_dict(notes, note)

        # create new note in the succeeding bar with the remaining duration
        note = self.new_note(pitch, velocity, (bar + 1) * self.BAR_DURATION, end, bar + 1)
        self.append_note_to_notes_dict(notes, note)

      else:
        note = self.new_note(pitch, velocity, start, end, bar)
        self.append_note_to_notes_dict(notes, note)

    # create a dataframe from the notes dictionary
    notes_df = pd.DataFrame({name: np.array(value) for name, value in notes.items()})


    # split notes into bars and convert notes ticks into a time serie of strings
    bars_time_series = []
    for bar_id in notes_df['bar'].unique():
      bar_df = notes_df[notes_df['bar'] == bar_id]
      bar_df = bar_df.reset_index(drop=True)

      # fill the beginning and end of each bar with empty notes if necessary
      if bar_df.loc[len(bar_df) - 1, 'end'] != self.BAR_LENGTH:
        note = self.new_note( pitch = 0,
                              velocity = 0,
                              start = bar_df.loc[len(bar_df) - 1, 'end'],
                              end = self.BAR_LENGTH,
                              bar = bar,
                              convert_to_ticks = False)
        bar_df = bar_df.append(note, ignore_index=True)

      if bar_df.at[0, 'start'] != 0:
        note = self.new_note( pitch = 0,
                              velocity = 0,
                              start = 0,
                              end = bar_df.at[0, 'start'],
                              bar = bar,
                              convert_to_ticks = False)
        bar_df = bar_df.append(note, ignore_index=True) 
        bar_df = bar_df.sort_values(by=['start']) 
        bar_df = bar_df.reset_index(drop=True)


      # convert note ticks into a time serie of strings 
      bar_time_serie = np.empty((self.BAR_LENGTH), dtype=object)
      bar_time_serie[:] = SILENCE_TOKEN
      for i in range(len(bar_df)):
        note = bar_df.loc[i, 'pitch']
        if note != 0:
          start = bar_df.loc[i, 'start']
          end = bar_df.loc[i, 'end']
          bar_time_serie[start] = str(note) + NOTE_START_TOKEN
          bar_time_serie[start+1:end] = str(note)
      bars_time_series.append(bar_time_serie)


    # flat bars and extract the string vocabulary
    flatten_time_series = np.concatenate(bars_time_series)
    token_list = list(set(flatten_time_series))

    # create the vocabulary
    VOCAB = Dictionary()
    for i in range(0, len(token_list)):
        VOCAB.add_word(token_list[i])

    # create the sequences of tokens for the model 
    sequences=[]
    num_sequences = len(flatten_time_series) - self.SEQ_LENGTH
    for i in range(0, num_sequences, self.BAR_LENGTH):
      seq = flatten_time_series[i:(i+self.SEQ_LENGTH)].copy() # NB: copy is necessary to avoid modifying the original array

      # add the BCI token to the input sequences at each time step
      if 'input' in self.midi_file_path:
        seq = np.concatenate(([BCI_TOKEN], seq[:-1]))
        VOCAB.add_word(BCI_TOKEN)

      for i in range(len(seq)):
        seq[i] = VOCAB.word2idx[seq[i]] 

      sequences.append(seq)
      

    self.VOCAB = VOCAB
    self.sequences = sequences
    self.notes_df = notes_df
    self.num_bars = len(bars_time_series)


