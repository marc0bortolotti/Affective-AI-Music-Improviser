import time
import rtmidi
import logging
import UDP.udp_connection as udp
import threading


NOTE_BUFFER_SIZE = 16
BAR_BUFFER_SIZE = 4
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


def thread_function_server(name, midi, server_ip, server_port):
    logging.info("Thread %s: starting", name)
    server = udp.Server(ip= server_ip, port= server_port)
    server.start()
    while not midi.exit:
        msg = server.get_message() # NB: it must receive at least one packet, otherwise it will block the loop
        if 'CLICK: 1/4' in msg:
                midi.process_note_buffer()
                midi.update_bar_buffer()
                midi.clear_note_buffer()
    server.close()
    

class MIDI_Input:

    def __init__(self, server_ip, server_port, parse_message = False):
        self.midi_in = rtmidi.MidiIn()
        self.available_ports = self.midi_in.get_port_count()
        
        if self.available_ports == 0:
            logging.info('MIDI Input: No MIDI devices detected. Retrying...')
        while self.available_ports == 0:
            self.available_ports = self.midi_in.get_port_count()
            time.sleep(0.001)

        self.midi_in.open_port(0)
        logging.info(f'MIDI Input: Connected to port {self.midi_in.get_port_name(0)}')  

        self.server_thread = threading.Thread(target= thread_function_server, args= ('MIDI Server', self, server_ip, server_port))
        self.server_thread.start()
        self.note_buffer = []
        self.bar_buffer = []
        self.exit = False
        self.parse_message = parse_message

    def run(self):
        logging.info(f"MIDI Input: running")
        while not self.exit:
            msg_and_dt = self.midi_in.get_message()
            if msg_and_dt:
                (msg, dt) = msg_and_dt
                command, note, velocity = msg
                if note in DRUM_MIDI_DICT and velocity > 10:
                    command = hex(command)
                    self.note_buffer.append(DRUM_MIDI_DICT[note])
                    if self.parse_message:
                        logging.info(f"MIDI Input: received messagge <<{command} [{DRUM_MIDI_DICT[note]}, {velocity}]\t| dt = {dt:.2f}>>")
            else:
                time.sleep(0.001)
        
        self.midi_in.close_port()
        logging.info(f'MIDI Input: Disconnected')

    def get_note_buffer(self):
        return self.note_buffer
    
    def clear_note_buffer(self):
        self.note_buffer = []

    def process_note_buffer(self):
        if len(self.note_buffer) > 0:
            logging.info(f'MIDI Input: Bar <<{self.note_buffer}>>')
        else:
            logging.info(f'MIDI Input: Bar <<Empty>>')

    def update_bar_buffer(self):
        self.bar_buffer.append(self.note_buffer)
        if len(self.bar_buffer) > BAR_BUFFER_SIZE:
            self.bar_buffer.pop(0)

    def close(self):
        self.exit = True
        self.server_thread.join()
        logging.info(f'Thread MIDI Server: finishing')

