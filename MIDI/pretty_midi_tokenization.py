import collections
import numpy as np
import pandas as pd
import pretty_midi
import tensorflow as tf


KEY_ORDER = ['pitch', 'step', 'duration']
BPM = 100
BEATS_PER_BAR = 4



def midi_to_notes(midi_file_path: str, bpm = BPM, beats_per_bar = BEATS_PER_BAR) -> pd.DataFrame:

  pm = pretty_midi.PrettyMIDI(midi_file_path)
  instrument = pm.instruments[0]
  notes = collections.defaultdict(list) # Dictionary with values as list
  bar_duration = (60/bpm) * beats_per_bar

  ticks_per_beat = pm.resolution
  print(f'Ticks per beat: {ticks_per_beat}')

  # Sort the notes by start time
  sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
  prev_start = sorted_notes[0].start

  for note in sorted_notes:

    pitch = note.pitch
    velocity = note.velocity
    start = note.start
    end = note.end
    step = start - prev_start
    duration = end - start
    bar = int(start // bar_duration) + 1 # integer part of the division
    
    notes['pitch'].append(pitch)
    notes['velocity'].append(velocity)
    notes['start'].append(start)
    notes['step'].append(step)
    notes['step_ticks'].append(pm.time_to_tick(step))
    notes['bar'].append(bar)

    # split the note in two if it spans multiple bars
    if start + duration > (bar + 1) * bar_duration: 

      # update the current note to end at the end of the bar and update its duration
      updated_end = (bar + 1) * bar_duration
      updted_duration = updated_end - start

      notes['end'].append(updated_end)
      notes['duration'].append(updted_duration)
      notes['duration_ticks'].append(pm.time_to_tick(updted_duration))

      # create new note in the succeeding bar with the remaining duration

      succ_pitch = pitch  
      succ_velocity = velocity
      succ_start = (bar + 1) * bar_duration
      succ_end = end
      succ_step = succ_start - start
      succ_duration = succ_end - succ_start
      succ_bar = bar + 1

      notes['pitch'].append(succ_pitch)
      notes['velocity'].append(succ_velocity)
      notes['start'].append(succ_start)
      notes['end'].append(succ_end)
      notes['step'].append(succ_step)
      notes['step_ticks'].append(pm.time_to_tick(succ_step))
      notes['duration'].append(succ_duration)
      notes['duration_ticks'].append(succ_duration)
      notes['bar'].append(succ_bar)

      prev_start = succ_start

    else:
      notes['end'].append(end)
      notes['duration'].append(duration)
      notes['duration_ticks'].append(pm.time_to_tick(duration))
      prev_start = start

  # create a dataframe from the notes dictionary
  notes_df = pd.DataFrame({name: np.array(value) for name, value in notes.items()})

  # put bar column at the end of the dataframe
  notes_df = notes_df[[col for col in notes_df.columns if col != 'bar'] + ['bar']]

  return notes_df, ticks_per_beat




def extract_bars_from_notes(notes: pd.DataFrame) -> list:

  bars = []

  # get the unique bars in the notes dataframe 
  unique_bars = notes['bar'].unique()

  for bar in unique_bars:
    bar_notes = notes[notes['bar'] == bar]
    bar = bar_notes.reset_index(drop = True)
    bar.loc[0, 'step'] = bar.loc[0, 'start'] - bar.loc[0, 'bar']
    if bar.loc[-1, 'end'] != bar.loc[-1, 'bar'] + 1:
       bar.loc[len(bar.index)] = bar.loc[len(bar.index)-1] 
       bar.loc[len(bar.index), 'velocity'] = 0
    bars.append(bar)

 
  return bars



def notes_to_midi(
  notes: pd.DataFrame,
  out_file: str, 
  instrument_name: str,
  velocity: int = 100,  # note loudness
) -> pretty_midi.PrettyMIDI:

  pm = pretty_midi.PrettyMIDI()
  program = pretty_midi.instrument_name_to_program(instrument_name)
  instrument = pretty_midi.Instrument(program=program)

  prev_start = 0
  for i, note in notes.iterrows():
    start = float(prev_start + note['step'])
    end = float(start + note['duration'])
    note = pretty_midi.Note(
        velocity=velocity,
        pitch=int(note['pitch']),
        start=start,
        end=end,
    )
    instrument.notes.append(note)
    prev_start = start

  pm.instruments.append(instrument)
  pm.write(out_file)
  return pm



def create_sequences(
    in_bars: list,
    out_bars: list,  
    seq_length: int,
    vocab_size = 128,
) -> tf.data.Dataset:
  
  """
  Returns TF Dataset of sequence and label examples. 
  The latter is the next note of the sequence that has to be predicted.
  """
  out_notes = pd.concat(out_bars)

  in_notes = pd.concat(in_bars)
  key_order = ['pitch', 'step', 'duration']
  in_notes = np.stack([in_notes[key] for key in key_order], axis=1)

  # Return a dataset of datasets of length `seq_length` 
  # Shift by 1 note each time
  # Stride = 1 means that it keeps all the windows created and doesn't skip any
  in_windows = in_bars.window(seq_length, shift=1, stride=1, drop_remainder=True)

  # add the output note to the input windows that will be used as labels 
  for i, window in enumerate(in_windows):
    in_windows[i] = window.concatenate(out_bars[i + seq_length - 1])

  # `The dataset of datasets is flatted into a unique dataset of tensors in batches of `seq_length`notes
  flatten = lambda x: x.batch(seq_length + 1, drop_remainder=True)
  sequences = in_windows.flat_map(flatten)
  
  # Normalize note pitch
  def scale_pitch(x):
    x = x/[vocab_size,1.0,1.0]
    return x

  # Split the labels from the input sequences
  def split_labels(sequences):
    inputs = sequences[:-1]
    labels_dense = sequences[-1]
    labels = {key:labels_dense[i] for i,key in enumerate(KEY_ORDER)}

    return scale_pitch(inputs), labels

  return sequences.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)