import time
import rtmidi

# initialise midi in class
midi_in = rtmidi.MidiIn()

# connect to a device
midi_in.open_port(3)

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
        print(f"{command} {msg[1:]}\t| dt = {dt:.2f}")
    else:
        # add a short sleep so the while loop doesn't hammer your cpu
        time.sleep(0.001)
