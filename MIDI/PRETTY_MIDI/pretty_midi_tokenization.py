import collections
import numpy as np
import pandas as pd
import pretty_midi
import logging
from mido import Message
from mido import MidiFile, MidiTrack, MetaMessage
import mido

BPM = 120
TICKS_PER_BEAT = 12 # resolution of the MIDI file
BEATS_PER_BAR = 4


VELOCITY_THRESHOLD = 80
NOTE_START_TOKEN = 'S'
VELOCITY_PIANO_TOKEN = 'p'
VELOCITY_FORTE_TOKEN = 'f'
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
  
  def is_in_vocab(self, word):
      return word in self.word2idx

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
    self.BEAT_DURATION = 60 / self.BPM
    self.BAR_DURATION = self.BEAT_DURATION * self.BEATS_PER_BAR
    self.TEMPO = int(self.BEAT_DURATION * 1000000)

    self.VOCAB = None

    self.source_path = midi_file_path
    self.source_pm = pretty_midi.PrettyMIDI(self.source_path)
    self.midi_out_port = mido.open_output('loopMIDI Port 1')

    if self.source_pm.resolution != TICKS_PER_BEAT:
      raise Exception('The resolution of the MIDI file is {self.pm.resolution} instead of {TICKS_PER_BEAT}.')

    elif self.source_pm.get_tempo_changes()[1][0] != BPM:
      raise Exception('The tempo of the MIDI file is {self.pm.get_tempo_changes[1][0]} instead of {BPM}.')
  
    else:
      self.sequences, self.num_bars, self.tokens_weights, self.notes_df = self.midi_to_tokens(self.source_path)
      


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


  def midi_to_tokens(self, midi_path, update_vocab = True):

    pm = pretty_midi.PrettyMIDI(midi_path)
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
    bar_ids = notes_df['bar'].unique()
    for bar_id in bar_ids:
      bar_df = notes_df[notes_df['bar'] == bar_id]
      bar_df = bar_df.reset_index(drop=True)

      # convert note ticks into a time serie of strings 
      bar_time_serie = np.empty((self.BAR_LENGTH), dtype=object)
      bar_time_serie[:] = SILENCE_TOKEN
      for i in range(len(bar_df)):
        note = bar_df.loc[i, 'pitch']
        start = bar_df.loc[i, 'start']
        end = bar_df.loc[i, 'end']

        if bar_df.loc[i, 'velocity'] < VELOCITY_THRESHOLD:
          bar_time_serie[start] = str(note) + NOTE_START_TOKEN + VELOCITY_PIANO_TOKEN
          bar_time_serie[start+1:end] = str(note) + VELOCITY_PIANO_TOKEN
        else:
          bar_time_serie[start] = str(note) + NOTE_START_TOKEN + VELOCITY_FORTE_TOKEN
          bar_time_serie[start+1:end] = str(note) + VELOCITY_FORTE_TOKEN

      bars_time_series.append(bar_time_serie)

    num_bars = len(bar_ids)

    # flat bars and create vocabulary of unique tokens
    flatten_time_series = np.concatenate(bars_time_series)
    tokens_vs_frequency = collections.Counter(flatten_time_series)
    tokens = list(tokens_vs_frequency.keys())
    tokens_weights = [tokens_vs_frequency[token]/len(flatten_time_series)  for token in tokens]

    # create the vocabulary
    if update_vocab or self.VOCAB is None:
      self.VOCAB = Dictionary()
      for i in range(0, len(tokens)):
          self.VOCAB.add_word(tokens[i])

    # create the sequences of tokens for the model 
    sequences = []
    num_sequences = len(flatten_time_series) - self.SEQ_LENGTH
    if num_sequences == 0:
      num_sequences = 1
      
    for i in range(0, num_sequences, self.BAR_LENGTH):
      seq = flatten_time_series[i:(i+self.SEQ_LENGTH)].copy() # NB: copy is necessary to avoid modifying the original array

      # add the BCI token to the input sequences at each time step
      if 'input' in midi_path or 'test' in midi_path:
        seq = np.concatenate(([BCI_TOKEN], seq[:-1]))
        self.VOCAB.add_word(BCI_TOKEN)

      for i in range(len(seq)):
        seq[i] = self.VOCAB.word2idx[seq[i]] 

      sequences.append(seq)

    return sequences, num_bars, tokens_weights, notes_df



  def tokens_to_midi(self, sequence, out_file_path = None):

    instrument_name = 'Acoustic Bass'
    pm = pretty_midi.PrettyMIDI(midi_file=None, resolution=self.TICKS_PER_BEAT, initial_tempo=self.BPM)
    program = pretty_midi.instrument_name_to_program(instrument_name)
    instrument = pretty_midi.Instrument(program=program)

    last_pitch = None
    last_velocity = None
    counter = 1
    pitch_ticks_velocity = []

    # convert the sequence of tokens into a list of tokens and their duration
    for i, token in enumerate(sequence):

        token_string = self.VOCAB.idx2word[token] # convert token id to string

        # extract pitch and velocity from the token string
        if VELOCITY_PIANO_TOKEN in token_string:
          velocity = 90
          token_string = token_string.replace(VELOCITY_PIANO_TOKEN, '')
        elif VELOCITY_FORTE_TOKEN in token_string:
          velocity = 127
          token_string = token_string.replace(VELOCITY_FORTE_TOKEN, '')

        if token_string == SILENCE_TOKEN:
            pitch = 0
            velocity = 0
        elif NOTE_START_TOKEN in token_string:
            pitch = int(token_string.replace(NOTE_START_TOKEN, ''))
        else:
            pitch = int(token_string)

        # update the pitch, duration and velocity lists
        if last_pitch != None and last_velocity != None:

            if  pitch != last_pitch or NOTE_START_TOKEN in token_string:
                pitch_ticks_velocity.append([last_pitch, counter, last_velocity])
                counter = 1
                last_pitch = pitch
                if i == len(sequence) - 1:
                    pitch_ticks_velocity.append([pitch, 1, velocity])
            else:
                counter += 1
        
        last_velocity = velocity  
        last_pitch = pitch

    # convert ticks to time and generate midi file
    prev_start = 0
    for note in pitch_ticks_velocity:
        (pitch, ticks, velocity) = note
        duration = pm.tick_to_time(ticks) # convert ticks to time
        end = prev_start + duration

        instrument.notes.append(pretty_midi.Note(velocity=velocity, pitch=pitch, start=prev_start, end=end))
        prev_start = end

    pm.instruments.append(instrument)
    if out_file_path is not None:
      pm.write(out_file_path)
      print(f'MIDI file saved at {out_file_path}')

    return pitch_ticks_velocity
  

  def send_midi_to_reaper(self, bar, parse_message = False):

    mid = MidiFile(ticks_per_beat = TICKS_PER_BEAT)
    track = MidiTrack()
    mid.tracks.append(track)
    track.append(MetaMessage('set_tempo', tempo=self.TEMPO))
    for pitch, ticks, velocity in bar:  
      track.append(Message('note_on', note=pitch, velocity=velocity, time=0)) # NB: time from the previous message in ticks per beat
      track.append(Message('note_off', note=pitch, velocity=velocity, time=ticks))
      if parse_message:
          logging.info(f'note: {pitch}, ticks: {ticks}, velocity: {velocity}')

    for msg in mid.play():
      self.midi_out_port.send(msg)






if __name__ == '__main__':
   
  tok = PrettyMidiTokenizer('TCN/dataset/output/bass.mid')

  sample = tok.sequences[0]

  note_ticks_velocity = tok.tokens_to_midi(sample, 'MIDI/PRETTY_MIDI/sample.mid')
  print(note_ticks_velocity)

  tok.send_midi_to_reaper(note_ticks_velocity, parse_message = True)