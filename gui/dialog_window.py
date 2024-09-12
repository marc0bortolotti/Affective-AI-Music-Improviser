from PyQt5 import QtWidgets, QtCore
import re
import bluetooth
import rtmidi
from PyQt5.QtWidgets import QApplication, QMessageBox


def retrieve_eeg_devices():
    saved_devices = bluetooth.discover_devices(duration=1, lookup_names=True, lookup_class=True)
    unicorn_devices = list(filter(lambda x: re.search(r'UN-\d{4}.\d{2}.\d{2}', x[1]), saved_devices))
    enophone_devices = list(filter(lambda x: re.search(r'enophone', x[1]), saved_devices))
    synthetic_devices = [('00:00:00:00:00:00', 'Synthetic Board', '0000')]
    ant_neuro_devices = [('ANT_NEURO_225', 'ANT Neuro 225', '0000'), ('ANT_NEURO_411', 'ANT Neuro 411', '0000')]
    lsl_device = [('LSL', 'LSL', '0000')]
    all_devices = synthetic_devices + unicorn_devices + enophone_devices + ant_neuro_devices + lsl_device
    return all_devices


def retrieve_midi_ports():
    available_input_ports = []    
    for port in rtmidi.MidiIn().get_ports():
        available_input_ports.append(port)
    available_output_ports = []
    for port in rtmidi.MidiOut().get_ports():
        available_output_ports.append(port)
    return available_input_ports, available_output_ports



class SetupDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(SetupDialog, self).__init__(parent)

        self.setWindowTitle('Setup')

        # Create layout
        layout = QtWidgets.QVBoxLayout(self)

        # Create dropdown for midi input selection
        devices = retrieve_eeg_devices()
        self.device_combo = QtWidgets.QComboBox(self)
        self.device_combo.addItems([device[1] for device in devices])
        layout.addWidget(QtWidgets.QLabel('Select EEG device'))
        layout.addWidget(self.device_combo)

        # Create dropdown for MIDI ports selection
        input_ports, output_ports = retrieve_midi_ports()

        self.instrument_input_combo = QtWidgets.QComboBox(self)
        self.instrument_input_combo.addItems([port for port in input_ports])
        self.instrument_input_combo.setCurrentText('Drum In Port 3')
        layout.addWidget(QtWidgets.QLabel('Select MIDI INPUT port for the INSTRUMENT'))
        layout.addWidget(self.instrument_input_combo)

        self.instrument_output_combo = QtWidgets.QComboBox(self)
        self.instrument_output_combo.addItems([port for port in output_ports])
        self.instrument_output_combo.setCurrentText('Drum Out Port 1')
        layout.addWidget(QtWidgets.QLabel('Select MIDI OUTPUT port for the INSTRUMENT'))
        layout.addWidget(self.instrument_output_combo)

        self.melody_rec_combo = QtWidgets.QComboBox(self)
        self.melody_rec_combo.addItems([port for port in output_ports])
        self.melody_rec_combo.setCurrentText('Bass Out Port Recording 2')
        layout.addWidget(QtWidgets.QLabel('Select MIDI RECORD port for the GENERATED MELODY'))
        layout.addWidget(self.melody_rec_combo)

        self.melody_play_combo = QtWidgets.QComboBox(self)
        self.melody_play_combo.addItems([port for port in output_ports])
        self.melody_play_combo.setCurrentText('Bass Out Port Playing 3')
        layout.addWidget(QtWidgets.QLabel('Select MIDI PLAY port for the GENERATED MELODY'))
        layout.addWidget(self.melody_play_combo)

        self.simulation_combo = QtWidgets.QComboBox(self)
        self.simulation_combo.addItems([port for port in output_ports])
        self.simulation_combo.setCurrentText('Simulation Port 5')
        layout.addWidget(QtWidgets.QLabel('Select MIDI SIMULATION port for the GENERATED MELODY'))
        layout.addWidget(self.simulation_combo)


        # Create slider for window size
        self.window_size_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.window_size_slider.setMinimum(1)
        self.window_size_slider.setMaximum(20)
        self.window_size_slider.setValue(4)
        self.window_size_slider.valueChanged.connect(self.update_window_size_label)
        self.window_size_label = QtWidgets.QLabel(f'Window size: {self.window_size_slider.value()} seconds', self)

        layout.addWidget(self.window_size_label)
        layout.addWidget(self.window_size_slider)

        # Add OK and Cancel buttons
        self.button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel, self)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def update_window_size_label(self, value):
        self.window_size_label.setText(f'Window size: {value} seconds')


    def get_data(self):
        return {
            'EEG_DEVICE_SERIAL_NUMBER' : self.device_combo.currentText(),
            'INSTRUMENT_MIDI_IN_PORT_NAME' : self.instrument_input_combo.currentText(),
            'INSTRUMENT_MIDI_OUT_PORT_NAME' : self.instrument_output_combo.currentText(),
            'MELODY_MIDI_REC_PORT_NAME' : self.melody_rec_combo.currentText(),
            'MELODY_MIDI_PLAY_PORT_NAME' : self.melody_play_combo.currentText(),
            'SIMULATION_MIDI_OUT_PORT_NAME' : self.simulation_combo.currentText(),
            'WINDOW_DURATION' : self.window_size_slider.value()
        }



class CustomDialog(QMessageBox):
    def __init__(self, answer, buttons=['Yes', 'No']):
        super().__init__()

        # Create a message box (QMessageBox)
        self.setWindowTitle('Custom Dialog')
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
    
