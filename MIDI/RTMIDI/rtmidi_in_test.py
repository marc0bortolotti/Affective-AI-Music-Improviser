import time
import rtmidi
import logging

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


midi_in = rtmidi.MidiIn()
available_ports = midi_in.get_ports()

print(available_ports)
idx = 4
midi_in.open_port(idx)
print(f'MIDI Input: Connected to port {midi_in.get_port_name(idx)}')

logging.info(f"MIDI Input: running")
while True:
    msg_and_dt = midi_in.get_message()
    if msg_and_dt:
        (msg, dt) = msg_and_dt
        command, note, velocity = msg
        command = hex(command)
        logging.info(f"MIDI Input: received messagge <<{command} [{DRUM_MIDI_DICT[note]}, {velocity}]\t| dt = {dt:.2f}>>")
    else:
        time.sleep(0.001)
        
