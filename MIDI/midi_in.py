import time
import rtmidi
import logging

    

class MIDI_Input:

    def __init__(self, midi_in_port, parse_message = False):
        
        self.midi_in_port = midi_in_port
        self.note_buffer = []
        self.exit = False
        self.parse_message = parse_message


    def run(self):
        logging.info(f"MIDI Input: running")
        while not self.exit:
            msg_and_dt = self.midi_in_port.get_message()
    
            if msg_and_dt:
                (msg, dt) = msg_and_dt
                command, note, velocity = msg
                command = hex(command)
                self.note_buffer.append({'pitch' : note, 'velocity' : velocity, 'dt': dt})
                if self.parse_message:
                    logging.info(f"MIDI Input: received messagge <<{command} [{note}, {velocity}]\t| dt = {dt:.2f}>>")
            else:
                time.sleep(0.001)
        
        self.midi_in_port.close_port()
        logging.info(f'MIDI Input: Disconnected')

    def get_note_buffer(self):
        return self.note_buffer
    
    def clear_note_buffer(self):
        self.note_buffer = []

    def close(self):
        self.exit = True


