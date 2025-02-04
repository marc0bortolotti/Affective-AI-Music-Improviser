import re
import bluetooth
import rtmidi
import os

SIMULATE_INSTRUMENT = 'Simulate Instrument'

def retrieve_eeg_devices():
    saved_devices = bluetooth.discover_devices(duration=1, lookup_names=True, lookup_class=True)
    unicorn_devices = list(filter(lambda x: re.search(r'UN-\d{4}.\d{2}.\d{2}', x[1]), saved_devices))
    enophone_devices = list(filter(lambda x: re.search(r'enophone', x[1]), saved_devices))
    synthetic_devices = [('00:00:00:00:00:00', 'Synthetic Board', '0000')]
    ant_neuro_devices = [('ANT_NEURO_225', 'ANT Neuro 225', '0000'), ('ANT_NEURO_411', 'ANT Neuro 411', '0000')]
    lsl_device = [('LSL', 'LSL', '0000')]
    no_eeg_device = [('00:00:00:00:00:00', 'None', '0000')]
    all_devices = no_eeg_device + synthetic_devices + unicorn_devices + enophone_devices + ant_neuro_devices + lsl_device
    return all_devices

def retrieve_midi_ports():
    available_input_ports = []   
    available_input_ports.append(SIMULATE_INSTRUMENT)
    midi_in = rtmidi.MidiIn() 
    for port in midi_in.get_ports():
        available_input_ports.append(port)

    available_output_ports = []
    midi_out = rtmidi.MidiOut()
    for port in midi_out.get_ports():
        available_output_ports.append(port)
    return available_input_ports, available_output_ports

def retrieve_models(models_path):
    available_models = []
    for model in os.listdir(models_path):
        available_models.append(model)
    return available_models[:2]