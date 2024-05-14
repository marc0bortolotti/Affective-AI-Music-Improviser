import time
import rtmidi

BPM = 120
DT_SAME_TIME = 0.02
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

# initialise midi in class
midi_in = rtmidi.MidiIn()

# get the number of available ports
available_ports = midi_in.get_port_count()
if available_ports == 0:
    print('No MIDI devices detected. Retrying...')
while available_ports == 0:
    available_ports = midi_in.get_port_count()
    time.sleep(0.001)

# connect to a device
midi_in.open_port(0)
print(f'\nConnected to port: {midi_in.get_port_name(0)}\nListening...')

midi_buffer = []
bar_buffer = []

while True:

    msg_and_dt = midi_in.get_message()

    if msg_and_dt:
        # unpack the msg and time tuple
        (msg, dt) = msg_and_dt 
        command, note, velocity = msg

        if note in DRUM_MIDI_DICT and velocity > 10:
            command = hex(command) # convert the command integer to a hex so it's easier to read
            print(f"{command} [{DRUM_MIDI_DICT[note]}, {velocity}]\t| dt = {dt:.2f}")

    else:
        # add a short sleep so the while loop doesn't hammer your cpu
        time.sleep(0.001)
