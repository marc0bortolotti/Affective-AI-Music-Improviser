import time
import rtmidi
import logging
import mido
import threading
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

        self.midi_in_port_name = midi_in_port_name
        self.midi_out_port_name = midi_out_port_name        
        self.midi_in_port = None
        self.midi_out_port = None    
        self.note_buffer = []
        self.exit = False
        self.parse_message = parse_message
        self.midi_simulation_port = None
        self.simulation_event = None
        self.bar_duration = 4 * 60/120 # 4 beats per bar, 120 bpm
        self.simulation_track_path = None


    def open_port(self):
        available_ports = rtmidi.MidiIn().get_ports()

        if self.midi_in_port_name != SIMULATE_INSTRUMENT:
            try:
                self.midi_in_port = rtmidi.MidiIn().open_port(available_ports.index(self.midi_in_port_name))
                logging.info(f'MIDI Input: Connected to port {self.midi_in_port_name}')
            except:
                self.midi_in_port = None
                logging.error(f'MIDI Input: Could not connect to port {self.midi_in_port_name}, check if the port is available')
                logging.error(f'MIDI Input Available ports: {available_ports}')

        if self.midi_out_port_name is not None:
            available_ports = rtmidi.MidiOut().get_ports()
            try:
                self.midi_out_port = rtmidi.MidiOut().open_port(available_ports.index(self.midi_out_port_name))
                logging.info(f'MIDI Output: Connected to port {self.midi_out_port_name}')
            except:
                self.midi_out_port = None
                logging.error(f'MIDI Output: Could not connect to port {self.midi_out_port_name}, check if the port is available')
                logging.error(f'MIDI Output Available ports: {available_ports}')

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

    def set_midi_simulation(self, simulation_port, simulation_track_path):
        if self.midi_out_port is not None:
            self.midi_out_port.close_port()
        self.midi_simulation_port = mido.open_output(simulation_port)
        self.simulation_track_path = simulation_track_path

    def set_simulation_event(self, event):
        self.simulation_event = event

    def simulate(self):
        logging.info(f"MIDI Input: simulating")
        if self.simulation_track_path is None:
            logging.error(f'MIDI Input: No simulation track path defined')
            return
        mid = mido.MidiFile(self.simulation_track_path)
        while not self.exit:
            self.simulation_event.wait()
            for msg in mid.play(): 
                self.midi_simulation_port.send(msg)
                if msg.type in ['note_on', 'note_off']:
                    if msg.type == 'note_off':
                        msg.velocity = 0
                    self.note_buffer.append({'pitch' : msg.note, 'velocity' : msg.velocity, 'dt': msg.time})
                if self.exit:
                    break   
        self.midi_simulation_port.close()

    def get_note_buffer(self):
        notes = self.note_buffer.copy()
        self.note_buffer = self.note_buffer[len(notes):]
        return notes

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

        self.midi_out_port_name = midi_out_port_name
        self.midi_out_port = None
        self.parse_message = parse_message

    def open_port(self):
        try:
            self.midi_out_port = mido.open_output(self.midi_out_port_name) 
            logging.info(f'MIDI Output: Connected to port {self.midi_out_port_name}')
        except:
            self.midi_out_port = None
            logging.error(f'MIDI Output: Could not connect to port {self.midi_out_port_name}, check if the port is available')
            logging.error(f'MIDI Output Available ports: {mido.get_output_names()}')    

    def send_message(self, message):
        self.midi_out_port.send_message(message)
        if self.parse_message:
            logging.info(f"MIDI Output: sent message <<{message}>>")

    def close(self):
        self.midi_out_port.close()
        logging.info(f'MIDI Output: Disconnected')

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

    