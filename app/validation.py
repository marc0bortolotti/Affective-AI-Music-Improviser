import os
import time
import logging
from EEG.processing import generate_samples
import pygame

pygame.mixer.init()
file_path = os.path.dirname(__file__)
white_noise = pygame.mixer.Sound(file_path + '/music/White_Noise.wav')

def validation(eeg_device, window_size, window_overlap, rec_time=60):
    
    logging.info("Validation: Start Validation")

    # start recording eeg
    eeg_device.start_recording()
    eeg_device.insert_marker('V')
    time.sleep(5)  # wait for signal to stabilize

    eeg_samples_classes = []

    # Baseline (max 20 seconds)
    baseline_time = min(rec_time/2, 30)
    logging.info(f"Validation: Pause for {baseline_time} seconds.")
    eeg_device.insert_marker('WN')
    white_noise.play()
    start = time.time() 
    while True:
        if time.time() - start < baseline_time:
            time.sleep(0.2)
        else:
            break
    white_noise.stop()
    
    # Relaxed (1 minute)
    logging.info(f"Validation: Playing relaxed for {rec_time} seconds")
    eeg_device.insert_marker('VR')
    start = time.time() 
    while True:
        if time.time() - start < rec_time:
            time.sleep(0.2)
        else:
            break
    eeg = eeg_device.get_eeg_data(recording_time=rec_time)
    eeg_samples_classes.append(generate_samples(eeg, window_size, window_overlap))

    # Baseline (max 20 seconds)
    baseline_time = min(rec_time/2, 20)
    logging.info(f"Validation: Pause for {baseline_time} seconds.")
    eeg_device.insert_marker('WN')
    white_noise.play()
    start = time.time() 
    while True:
        if time.time() - start < baseline_time:
            time.sleep(0.2)
        else:
            break
    white_noise.stop()

    # Excited (1 minute)
    logging.info(f"Validation: Playing excited for {rec_time} seconds.")
    eeg_device.insert_marker('VE')
    start = time.time() 
    while True:
        if time.time() - start < rec_time:
            time.sleep(0.2)
        else:
            break
    eeg = eeg_device.get_eeg_data(recording_time=rec_time)
    eeg_samples_classes.append(generate_samples(eeg, window_size, window_overlap))

    # stop recording eeg
    eeg_device.stop_recording()

    # classification
    accuracy, f1 = eeg_device.get_metrics(eeg_samples_classes)

    logging.info(f"Validation fineshed: Accuracy: {accuracy}, F1: {f1}")

    return accuracy, f1
