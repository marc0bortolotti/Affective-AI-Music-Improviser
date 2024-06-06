import collections
import numpy as np
import pandas as pd
import pretty_midi


# Tokenization parameters
BPM = 120
TICKS_PER_BEAT = 12 # resolution of the MIDI file
BEATS_PER_BAR = 4


BAR_LENGTH = BEATS_PER_BAR * TICKS_PER_BEAT
SEQ_LENGTH = BAR_LENGTH * 4 # 4 bars
BAR_DURATION = (60 / BPM) * BEATS_PER_BAR


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



def convert_time_to_ticks(time, resolution = TICKS_PER_BEAT, bpm = BPM):
  pm = pretty_midi.PrettyMIDI(midi_file=None, resolution=resolution, initial_tempo=bpm)
  return pm.time_to_tick(time)


def new_note(pitch, velocity, start, end, bar, convert_to_ticks = True):
  # NB: start and end are relative to the bar they are in
  if convert_to_ticks:
    start = convert_time_to_ticks(start - bar*BAR_DURATION)
    end = convert_time_to_ticks(end - bar*BAR_DURATION)

  new_note = {
    'pitch': pitch,
    'velocity': velocity,
    'start': start,
    'end': end,
    'bar': bar
  }
  
  return new_note


def append_note_to_notes_dict(notes: pd.DataFrame, note: dict):
    for key, value in note.items():
        notes[key].append(value)    


def midi_to_tokens(midi_file_path, bpm = BPM, beats_per_bar = BEATS_PER_BAR, ticks_per_beat = TICKS_PER_BEAT):

  pm = pretty_midi.PrettyMIDI(midi_file_path)
  instrument = pm.instruments[0]
  notes = collections.defaultdict(list) # Dictionary with values as list
  
  bar_duration = (60/bpm) * beats_per_bar

  ticks_per_beat = pm.resolution

  # Sort the notes by start time
  sorted_notes = sorted(instrument.notes, key=lambda note: note.start)

  for note in sorted_notes:

    pitch = note.pitch
    velocity = note.velocity
    start = note.start
    end = note.end
    # step = start - prev_start
    duration = end - start
    bar = int(start // bar_duration) # integer part of the division

    # split the note in two if it spans multiple bars
    if start + duration > (bar + 1) * bar_duration: 

      # update the current note to end at the end of the bar and update its duration
      note = new_note(pitch, velocity, start, (bar + 1) * bar_duration, bar)
      append_note_to_notes_dict(notes, note)

      # create new note in the succeeding bar with the remaining duration
      note = new_note(pitch, velocity, (bar + 1) * bar_duration, end, bar + 1)
      append_note_to_notes_dict(notes, note)

    else:
      note = new_note(pitch, velocity, start, end, bar)
      append_note_to_notes_dict(notes, note)

  # create a dataframe from the notes dictionary
  notes_df = pd.DataFrame({name: np.array(value) for name, value in notes.items()})


  # split notes into bars and convert notes ticks into a time serie of strings
  bars_time_series = []
  for bar_id in notes_df['bar'].unique():
    bar_df = notes_df[notes_df['bar'] == bar_id]
    bar_df = bar_df.reset_index(drop=True)

    # fill the beginning and end of each bar with empty notes if necessary
    if bar_df.loc[len(bar_df) - 1, 'end'] != BAR_LENGTH:
      note = new_note(pitch = 0,
                      velocity = 0,
                      start = bar_df.loc[len(bar_df) - 1, 'end'],
                      end = BAR_LENGTH,
                      bar = bar,
                      convert_to_ticks = False)
      bar_df = bar_df.append(note, ignore_index=True)

    if bar_df.at[0, 'start'] != 0:
      note = new_note(pitch = 0,
                      velocity = 0,
                      start = 0,
                      end = bar_df.at[0, 'start'],
                      bar = bar,
                      convert_to_ticks = False)
      bar_df = bar_df.append(note, ignore_index=True) 
      bar_df = bar_df.sort_values(by=['start']) 
      bar_df = bar_df.reset_index(drop=True)


    # convert note ticks into a time serie of strings 
    bar_time_serie = np.empty((BAR_LENGTH), dtype=object)
    bar_time_serie[:] = SILENCE_TOKEN
    for i in range(len(bar_df)):
      note = bar_df.loc[i, 'pitch']
      if note != 0:
        start = bar_df.loc[i, 'start']
        end = bar_df.loc[i, 'end']
        bar_time_serie[start] = str(note)+NOTE_START_TOKEN
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
  num_sequences = len(flatten_time_series) - SEQ_LENGTH
  for i in range(0, num_sequences, BAR_LENGTH):
    seq = flatten_time_series[i:(i+SEQ_LENGTH)].copy() # NB: copy is necessary to avoid modifying the original array

    # add the BCI token to the input sequences at each time step
    if 'input' in midi_file_path:
      seq = np.concatenate(([BCI_TOKEN], seq[:-1]))
      VOCAB.add_word(BCI_TOKEN)

    for i in range(len(seq)):
      seq[i] = VOCAB.word2idx[seq[i]] 

    sequences.append(seq)
    

  return sequences, VOCAB, notes_df


