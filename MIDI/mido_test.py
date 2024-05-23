import mido
import os
from mido import Message, MidiFile, MidiTrack, MetaMessage, tick2second
import time

# msg = mido.Message('note_on', note=60)
# print(msg)
# port = mido.open_output('Port Name')
# port.send(msg)
# with mido.open_input() as inport:
#     for msg in inport:
#         print(msg)

BPM = 120 
TICKS_PER_BEAT = 3
BEAT_DURATION =  60/BPM # seconds
TEMPO = int(BEAT_DURATION * 1000000) # microseconds per beat
MEASURE_DURATION = 4 * BEAT_DURATION # seconds
DT = BEAT_DURATION/TICKS_PER_BEAT # seconds
SILENCE_TOKEN = 0

midi_path = 'C:/Users/Gianni/Desktop/MARCO/UNI/Magistrale/TESI/Code/MIDI/examples'
mid = mido.MidiFile(os.path.join(midi_path, 'bass_example_2.MID'))

note_buffer = []
measure_duration = 0
for msg in mid.play():
    
    # print(msg)

    if msg.type == 'note_on':
        silent_size = round(msg.time/DT)
        for i in range(silent_size):
            note_buffer.append(SILENCE_TOKEN)

    if msg.type == 'note_off':
        note_size = round(msg.time/DT)
        for i in range(note_size):
            note_buffer.append(msg.note)

    measure_duration += msg.time
    if measure_duration >= MEASURE_DURATION:
        break


mid = MidiFile(ticks_per_beat = TICKS_PER_BEAT)
track = MidiTrack()
mid.tracks.append(track)
track.append(MetaMessage('set_tempo', tempo=TEMPO)) # tempo in microseconds per beat

last_note = None
counter = 0
note_time_buffer = []
start_time = time.time()
for i, note in enumerate(note_buffer):
    if note != last_note and last_note is not None:
        note_time_buffer.append([last_note, counter])
        last_note = note
        counter = 1
        if i == len(note_time_buffer) - 1:
            note_time_buffer.append([note, 1])
    else:
        counter += 1
        last_note = note
print(f'Elapsed time: {time.time() - start_time}')
print(note_buffer)
print(note_time_buffer)




last_note = 48
total_ticks = 0
for note, ticks in note_time_buffer: 
    if note != SILENCE_TOKEN:   
        track.append(Message('note_on', note=note, velocity=127, time=0)) # NB: time from the previous message in ticks per beat
        track.append(Message('note_off', note=note, velocity=127, time=ticks))
        print(f'note {note} for {ticks} ticks, {tick2second(ticks, TICKS_PER_BEAT, TEMPO)} seconds')
        last_note = note
    else:
        track.append(Message('note_on', note=last_note, velocity=0, time=0))
        track.append(Message('note_off', note=last_note, velocity=0, time=ticks))
        print(f'silence for {ticks} ticks, {tick2second(ticks, TICKS_PER_BEAT, TEMPO)} seconds')

    total_ticks += ticks

print(f'Total ticks: {total_ticks}')
print(f'Total seconds: {tick2second(total_ticks, TICKS_PER_BEAT, TEMPO)}')
print(f'Track tempo: {TEMPO}')

mid.save('MIDI/output/Decoded_example.mid')
print('Done')

