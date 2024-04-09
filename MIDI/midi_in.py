import time
import rtmidi

# initialise midi in class
midi_in = rtmidi.MidiIn()

# get the number of available ports
print(f'Available MIDI ports: {midi_in.get_port_count()}')

# connect to a device
midi_in.open_port(0)

# print the name of the connected device
print(f'Connected to port: {midi_in.get_port_name(0)}\nListening...')

# get midi msgs
while True:
    # get a message, returns None if there's no msg in queue
    # also include the time since the last msg
    msg_and_dt = midi_in.get_message()

    # check to see if there is a message
    if msg_and_dt:
        # unpack the msg and time tuple
        (msg, dt) = msg_and_dt

        # convert the command integer to a hex so it's easier to read
        command = hex(msg[0])
        # print(f"{command} {msg[1:]}\t| dt = {dt:.2f}")

        if msg[2] > 10:
            if msg[1] == 36:
                print("Kick")
            elif msg[1] == 38:
                print("Snare")
            elif msg[1] == 42:
                print("Closed Hi-Hat")
            elif msg[1] == 43:
                print("Floor Tom")
            elif msg[1] == 44:
                print("Pedal Hi-Hat")
            elif msg[1] == 46:
                print("Open Hi-Hat")
            elif msg[1] == 47:
                print("Tom-2")
            elif msg[1] == 48:
                print("Tom-1")
            elif msg[1] == 49:
                print("Crash")
            elif msg[1] == 51:
                print("Ride")
        
    else:
        # add a short sleep so the while loop doesn't hammer your cpu
        time.sleep(0.001)
