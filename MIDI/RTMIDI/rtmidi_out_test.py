import time
import rtmidi

midiout = rtmidi.MidiOut()
available_ports = midiout.get_ports()

print(available_ports)

if available_ports:
    midiout.open_port(1)
else:
    midiout.open_virtual_port("My virtual output")

while True:
    note_on = [0x90, 60, 112] # channel 1, middle C, velocity 112
    note_off = [0x80, 60, 0]
    midiout.send_message(note_on)
    print("Note on")
    time.sleep(0.2)
    midiout.send_message(note_off)
    print("Note off")
    # time.sleep(3)

del midiout


