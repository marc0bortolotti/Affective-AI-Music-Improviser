import time
import rtmidi
import logging
import mido
import threading
import os
from gui.dialog_window import SIMULATE_INSTRUMENT

class MIDI_Input:
    '''
    Class to receive MIDI messages from a MIDI port

    Args:
    midi_in_port_name (str): MIDI input port name
    midi_out_port_name (str): MIDI output port name
    parse_message (bool): whether to print the message received from the port
    '''

    def __init__(self, midi_in_port_name, midi_out_port_name = None, parse_message = False):

        available_ports = rtmidi.MidiIn().get_ports()

        if midi_in_port_name != SIMULATE_INSTRUMENT:
            try:
                self.midi_in_port = rtmidi.MidiIn().open_port(available_ports.index(midi_in_port_name))
                logging.info(f'MIDI Input: Connected to port {midi_in_port_name}')
            except:
                self.midi_in_port = None
                logging.error(f'MIDI Input: Could not connect to port {midi_in_port_name}, check if the port is available')
                logging.error(f'MIDI Input Available ports: {available_ports}')

        if midi_out_port_name is not None:
            available_ports = rtmidi.MidiOut().get_ports()
            try:
                self.midi_out_port = rtmidi.MidiOut().open_port(available_ports.index(midi_out_port_name))
                logging.info(f'MIDI Output: Connected to port {midi_out_port_name}')
            except:
                self.midi_out_port = None
                logging.error(f'MIDI Output: Could not connect to port {midi_out_port_name}, check if the port is available')
                logging.error(f'MIDI Output Available ports: {available_ports}')
            
        self.note_buffer = []
        self.exit = False
        self.parse_message = parse_message
        self.midi_simulation_port = None
        self.simulation_event = None

    def run(self):
        logging.info(f"MIDI Input: running")
        while not self.exit:
            msg_and_dt = self.midi_in_port.get_message()
    
            if msg_and_dt:
                (msg, dt) = msg_and_dt
                if self.midi_out_port is not None:
                    self.midi_out_port.send_message(msg)
                command, note, velocity = msg
                command = hex(command)
                self.note_buffer.append({'pitch' : note, 'velocity' : velocity, 'dt': dt})
                if self.parse_message:
                    logging.info(f"MIDI Input: received messagge <<{command} [{note}, {velocity}]\t| dt = {dt:.2f}>>")
            else:
                time.sleep(0.001)
        
        self.midi_in_port.close_port()
        if self.midi_out_port is not None:
            self.midi_out_port.close_port()
        if self.midi_simulation_port is not None:
            self.midi_simulation_port.close()
            
        logging.info(f'MIDI Input: Disconnected')

    def set_midi_simulation_port(self, midi_simulation_port):
        self.midi_out_port.close_port()
        self.midi_simulation_port = mido.open_output(midi_simulation_port)

    def set_simulation_event(self, event):
        self.simulation_event = event

    def simulate(self, path=None):

        path = os.path.join(os.path.dirname(__file__), 'midi_simulation_tracks/drum_rock_relax.mid')

        def thread_function(path):
            mid = mido.MidiFile(path)
            for msg in mid.play(): 
                if not msg.is_meta and msg.type != 'control_change':
                    self.note_buffer.append({'pitch' : msg.note, 'velocity' : msg.velocity, 'dt': msg.time})
                    self.midi_simulation_port.send(msg)
                if self.exit:
                    break   
            if self.simulation_event is not None:
                self.simulation_event.set()

        thread = threading.Thread(target=thread_function, args=(path,))
        thread.start()

    def get_note_buffer(self):
        return self.note_buffer
    
    def clear_note_buffer(self):
        self.note_buffer = []

    def close(self):
        self.exit = True




class MIDI_Output:
    '''
    Class to send MIDI messages to a MIDI port

    Args:
    midi_out_port_name (str): MIDI output port name
    parse_message (bool): whether to print the message sent to the port
    '''

    def __init__(self, midi_out_port_name, parse_message = False):

        try:
            self.midi_out_port = mido.open_output(midi_out_port_name) 
            logging.info(f'MIDI Output: Connected to port {midi_out_port_name}')
        except:
            self.midi_out_port = None
            logging.error(f'MIDI Output: Could not connect to port {midi_out_port_name}, check if the port is available')
            logging.error(f'MIDI Output Available ports: {mido.get_output_names()}')
        self.parse_message = parse_message

    def send_message(self, message):
        self.midi_out_port.send_message(message)
        if self.parse_message:
            logging.info(f"MIDI Output: sent message <<{message}>>")

    def close(self):
        self.midi_out_port.close_port()


    def send_midi_to_reaper(self, mid, parse_message = False):
        def thread_reaper(mid, parse_message):
            start_time = time.time()
            for msg in mid.play():
                self.midi_out_port.send(msg)
                if parse_message:
                    logging.info(f"MIDI Output: sent message <<{msg}>>")
            if parse_message:
                logging.info(f"MIDI Output: finished sending midi to Reaper in {time.time() - start_time:.2f} s")

        thread = threading.Thread(target=thread_reaper, args=(mid, parse_message))
        thread.start()

    