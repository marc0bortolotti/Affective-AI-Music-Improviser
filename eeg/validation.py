import os
import time
import simpleaudio
import logging
from eeg.processing import generate_samples

file_path = os.path.dirname(__file__)
relax_music = simpleaudio.WaveObject.from_wave_file(file_path + '/music/The_Scientist.wav')
excited_music = simpleaudio.WaveObject.from_wave_file(file_path + '/music/Blitzkrieg_Bop.wav')
white_noise = simpleaudio.WaveObject.from_wave_file(file_path + '/music/White_Noise.wav')

def validation(eeg_device, window_size, window_overlap, rec_time=60):
    logging.info("Validation: Start Validation")

    # start recording eeg
    eeg_device.start_recording()
    time.sleep(7)  # wait for signal to stabilize

    eeg_samples_classes = []

    # Relaxed (1 minute)
    logging.info(f"Validation: Play a relaxed rythm for {rec_time} seconds")
    play = relax_music.play()
    start = time.time() 
    while True:
        if time.time() - start < rec_time:
            time.sleep(0.2)
        else:
            break
    play.stop()
    eeg = eeg_device.get_eeg_data(recording_time=rec_time)
    eeg_samples_classes.append(generate_samples(eeg, window_size, window_overlap))

    time.sleep(5)  # wait for signal to stabilize

    # Excited (1 minute)
    logging.info(f"Validation: Play an excited rythm for {rec_time} seconds")
    play = excited_music.play()
    start = time.time() 
    while True:
        if time.time() - start < rec_time:
            time.sleep(0.2)
        else:
            break
    play.stop()
    eeg = eeg_device.get_eeg_data(recording_time=rec_time)
    eeg_samples_classes.append(generate_samples(eeg, window_size, window_overlap))

    # stop recording eeg
    eeg_device.stop_recording()

    # classification
    accuracy, f1 = eeg_device.get_metrics(eeg_samples_classes)

    logging.info(f"Validation fineshed: Accuracy: {accuracy}, F1: {f1}")

    return accuracy, f1
