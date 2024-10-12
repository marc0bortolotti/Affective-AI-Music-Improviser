from PyQt5 import QtWidgets
import re
import bluetooth
import rtmidi
from PyQt5.QtWidgets import QApplication, QMessageBox
import os

MODELS_PATH = os.path.join(os.path.dirname(__file__), '../generative_model/runs')
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
    for port in rtmidi.MidiIn().get_ports():
        available_input_ports.append(port)
    available_output_ports = []
    for port in rtmidi.MidiOut().get_ports():
        available_output_ports.append(port)
    return available_input_ports, available_output_ports

def retrieve_models():
    available_models = []
    for model in os.listdir(MODELS_PATH):
        available_models.append(model)
    return available_models

class SetupDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(SetupDialog, self).__init__(parent)

        self.setWindowTitle('Setup')

        # Create layout
        layout = QtWidgets.QVBoxLayout(self)

        # Create dropdown for models
        models = retrieve_models()
        self.model_combo = QtWidgets.QComboBox(self)
        self.model_combo.addItems(models)
        self.model_combo.setCurrentText(models[0])

        # Create dropdown for mood
        self.mood_combo = QtWidgets.QComboBox(self)
        self.mood_combo.addItems(['RELAXED', 'CONCENTRATED'])
        self.mood_combo.setCurrentText('RELAXED')

        # Create dropdown for eeg devices
        devices = retrieve_eeg_devices()
        self.device_combo = QtWidgets.QComboBox(self)
        self.device_combo.addItems([device[1] for device in devices])

        # Create dropdown for MIDI ports selection
        input_ports, output_ports = retrieve_midi_ports()

        self.instrument_combo = QtWidgets.QComboBox(self)
        self.instrument_combo.addItems([port for port in input_ports])
        self.instrument_combo.addItems([SIMULATE_INSTRUMENT])
        self.instrument_combo.setCurrentText(SIMULATE_INSTRUMENT)

        self.rhythm_combo = QtWidgets.QComboBox(self)
        self.rhythm_combo.addItems([port for port in output_ports])
        self.rhythm_combo.setCurrentText(output_ports[2])

        self.melody_combo = QtWidgets.QComboBox(self)
        self.melody_combo.addItems([port for port in output_ports])
        self.melody_combo.setCurrentText(output_ports[3])

        # Add OK and Cancel buttons
        self.button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel, self)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        # Add widgets to layout
        layout.addWidget(QtWidgets.QLabel('Select model'))
        layout.addWidget(self.model_combo)
        layout.addWidget(QtWidgets.QLabel('Select starting mood'))
        layout.addWidget(self.mood_combo)
        layout.addWidget(QtWidgets.QLabel('Select EEG device'))
        layout.addWidget(self.device_combo)
        layout.addWidget(QtWidgets.QLabel('Select MIDI INPUT port for the INSTRUMENT'))
        layout.addWidget(self.instrument_combo)
        layout.addWidget(QtWidgets.QLabel('Select MIDI OUTPUT port for the RHYTHM'))
        layout.addWidget(self.rhythm_combo)
        layout.addWidget(QtWidgets.QLabel('Select MIDI OUTPUT port for the MELODY'))
        layout.addWidget(self.melody_combo)
        layout.addWidget(self.button_box)


    def get_data(self):
        return {
            'MODEL' : self.model_combo.currentText(),
            'STARTING_MOOD' : self.mood_combo.currentText(),
            'EEG_DEVICE_SERIAL_NUMBER' : self.device_combo.currentText(),
            'INSTRUMENT_IN_PORT_NAME' : self.instrument_combo.currentText(),
            'RHYTHM_OUT_PORT_NAME' : self.rhythm_combo.currentText(),
            'MELODY_OUT_PORT_NAME' : self.melody_combo.currentText()
        }



class CustomDialog(QMessageBox):
    def __init__(self, answer, buttons=['Yes', 'No']):
        super().__init__()

        # Create a message box (QMessageBox)
        self.setWindowTitle('Custom Dialog')

        # Set the text of the message box in the center
        self.setText(answer)

        # Add custom buttons
        for button in buttons:
            if button == 'Yes':
                self.addButton(button, QMessageBox.YesRole)
            elif button == 'No':
                self.addButton(button, QMessageBox.NoRole)
            else:
                self.addButton(button, QMessageBox.ActionRole)
    


if __name__ == '__main__':
    app = QApplication([])
    window = CustomDialog('ciao')
    window.exec_()
    
