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
MIDI_FOLDER_PATH = 'C:/Users/Gianni/Desktop/MARCO/UNI/Magistrale/TESI/Code/MIDI'


msg = mido.Message('note_on', note=60, velocity=64, time=32)
print(msg)

# Open a MIDI output port and send a message
print(mido.get_output_names())
port = mido.open_output('loopMIDI Port 1')
port.send(msg)

# Open a MIDI input port and print received messages
with mido.open_input() as inport:
    for msg in inport:
        print(msg)

# Read a MIDI file
mid = mido.MidiFile(os.path.join(MIDI_FOLDER_PATH, 'examples/bass_example.MID'))
print(f'\nLoaded MIDI file: {mid.filename}')
print(f'Number of ticks per beat: {mid.ticks_per_beat}')
for msg in mid.play():
    print(msg)


# Create a MIDI file
mid = MidiFile(ticks_per_beat = TICKS_PER_BEAT)
track = MidiTrack()
mid.tracks.append(track)
note = 60
ticks = 32
track.append(MetaMessage('set_tempo', tempo=TEMPO)) # tempo in microseconds per beat 
track.append(Message('note_on', note=note, velocity=127, time=0)) # NB: time from the previous message in ticks per beat
track.append(Message('note_off', note=note, velocity=127, time=ticks))
                











  


