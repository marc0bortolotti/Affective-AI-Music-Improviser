import reapy

def set_midi_input():

    with reapy.inside_reaper():

        midi_port_names = reapy.core.reaper.midi.get_input_names()
        device_id = midi_port_names.index('Drum Out Port')
        
        # Get the project and track
        project = reapy.Project()

        # Create two new tracks
        track1 = project.add_track()
        track2 = project.add_track()

        # Name the tracks
        track1.name = "Track 1"
        track2.name = "Track 2"

        # Calculate input mode
        # 4096 is the constant for MIDI inputs
        # midi_device_id is the ID of the MIDI input device
        # midi_channel is shifted left by 5 positions
        channel = 1  # All channels
        input_mode = input_mode = 4096 + (device_id * 32) + channel

        # Set track input mode
        track1.set_info_value("I_RECINPUT", input_mode)
        track2.set_info_value("I_RECINPUT", input_mode)

        track1.set_info_value("I_RECARM", 1)  # Arm track 1
        track2.set_info_value("I_RECARM", 1)  # Arm track 2

        track1.set_info_value("I_RECMON", 1)


set_midi_input()
