import collections
import datetime
# import fluidsynth
import glob
import numpy as np
import pathlib
import pandas as pd
import pretty_midi
import seaborn as sns
import tensorflow as tf


filenames = glob.glob(str('data/**/**/*.mid*'))
print('Number of files:', len(filenames))

sample_file = filenames[1000]
print(sample_file)

pm = pretty_midi.PrettyMIDI(sample_file)
print('Number of instruments:', len(pm.instruments))

instrument = pm.instruments[0]
instrument_name = pretty_midi.program_to_instrument_name(instrument.program)
print('Instrument name:', instrument_name)

for i, note in enumerate(instrument.notes[:10]):
  note_name = pretty_midi.note_number_to_name(note.pitch)
  duration = note.end - note.start
  print(f'{i}: pitch={note.pitch}, note_name={note_name},'
        f' duration={duration:.4f}')