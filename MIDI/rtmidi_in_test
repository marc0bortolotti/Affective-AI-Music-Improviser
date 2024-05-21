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
available_ports = midi_in.get_port_count()

if available_ports == 0:
    logging.info('MIDI Input: No MIDI devices detected. Retrying...')
while available_ports == 0:
    available_ports = midi_in.get_port_count()
    time.sleep(0.001)

midi_in.open_port(0)

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
        
