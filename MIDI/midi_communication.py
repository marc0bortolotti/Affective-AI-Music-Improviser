import time
import rtmidi
import logging
from mido import MidiFile, MidiTrack, Message, MetaMessage
    

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




class MIDI_Output:

    def __init__(self, midi_out_port, parse_message = False):
        
        self.midi_out_port = midi_out_port
        self.parse_message = parse_message

    def send_message(self, message):
        self.midi_out_port.send_message(message)
        if self.parse_message:
            logging.info(f"MIDI Output: sent message <<{message}>>")

    def close(self):
        self.midi_out_port.close_port()

    def send_midi_to_reaper(self, pitch_ticks_velocity, resolution, parse_message = False):

        mid = MidiFile(ticks_per_beat = resolution)
        track = MidiTrack()
        mid.tracks.append(track)
        track.append(MetaMessage('set_tempo', tempo=self.TEMPO))
        for pitch, ticks, velocity in pitch_ticks_velocity:  
            track.append(Message('note_on', note=pitch, velocity=velocity, time=0)) # NB: time from the previous message in ticks per beat
            track.append(Message('note_off', note=pitch, velocity=velocity, time=ticks))
            if parse_message:
                logging.info(f'note: {pitch}, ticks: {ticks}, velocity: {velocity}')

        for msg in mid.play():
            self.midi_out_port.send(msg)






if __name__ == '__main__':

    midi_in = rtmidi.MidiIn()
    midi_in.open_port(3)
    print(f'MIDI Input: Connected to port {midi_in.get_port_name(3)}') 

    midi_in = MIDI_Input(midi_in)
    midi_in.run()