import time
import numpy as np
import logging
from eeg.processing import generate_samples
import simpleaudio
import os

file_path = os.path.dirname(__file__)
relax_music = simpleaudio.WaveObject.from_wave_file(file_path + '/music/The_Scientist.wav')
excited_music = simpleaudio.WaveObject.from_wave_file(file_path + '/music/Blitzkrieg_Bop.wav')
white_noise = simpleaudio.WaveObject.from_wave_file(file_path + '/music/White_Noise.wav')


def pretraining(eeg_device, WINDOW_SIZE, WINDOW_OVERLAP):
    logging.info("Pretraining: Start Training")

    steps = 1

    # start recording eeg
    eeg_device.start_recording()
    time.sleep(7)  # wait for signal to stabilize

    eeg_samples_baseline = []
    eeg_samples_relax = []
    eeg_samples_excited = []

    for i in range(steps):
        # Baseline (30 seconds)
        logging.info("Pretraining: Pause for 30 seconds. Please, do not move or think about anything. Just relax.")
        play = white_noise.play()
        play.wait_done()
        eeg = eeg_device.get_eeg_data(recording_time=30)
        eeg_samples_baseline.append(generate_samples(eeg, WINDOW_SIZE, WINDOW_OVERLAP))

        # Relax (1 minute)
        logging.info("Pretraining: Play a relaxed rythm for 60 seconds")
        play = relax_music.play()
        play.wait_done()
        eeg = eeg_device.get_eeg_data(recording_time=60)
        eeg_samples_relax.append(generate_samples(eeg, WINDOW_SIZE, WINDOW_OVERLAP))

        # Baseline (30 seconds)
        logging.info("Pretraining: Pause for 30 seconds. Please, do not move or think about anything. Just relax.")
        play = white_noise.play()
        play.wait_done()
        eeg = eeg_device.get_eeg_data(recording_time=30)
        eeg_samples_baseline.append(generate_samples(eeg, WINDOW_SIZE, WINDOW_OVERLAP))

        # Excited (1 minute)
        logging.info("Pretraining: Play an excited rythm for 60 seconds")
        play = excited_music.play()
        play.wait_done()
        eeg = eeg_device.get_eeg_data(recording_time=60)
        eeg_samples_excited.append(generate_samples(eeg, WINDOW_SIZE, WINDOW_OVERLAP))

    eeg_samples_baseline = np.concatenate(eeg_samples_baseline)
    eeg_samples_relax = np.concatenate(eeg_samples_relax)
    eeg_samples_excited = np.concatenate(eeg_samples_excited)
    eeg_samples_classes = [eeg_samples_relax, eeg_samples_excited]

    # stop recording eeg
    eeg_device.stop_recording()

    #------------CLASSIFICATION----------------
    scaler, svm_model, lda_model, baseline = eeg_device.fit_classifier(eeg_samples_baseline, eeg_samples_classes)
    logging.info("Pretraining: Training Finished")
    return scaler, svm_model, lda_model, baseline


def validation(eeg_device, WINDOW_SIZE, WINDOW_OVERLAP):
    logging.info("Validation: Start Validation")

    # start recording eeg
    eeg_device.start_recording()
    time.sleep(7)  # wait for signal to stabilize

    eeg_samples_classes = []

    # Relax (1 minute)
    logging.info("Validation: Play a relaxed rythm for 60 seconds")
    play = relax_music.play()
    play.wait_done()
    eeg = eeg_device.get_eeg_data(recording_time=60)
    eeg_samples_classes.append(generate_samples(eeg, WINDOW_SIZE, WINDOW_OVERLAP))

    # Excited (1 minute)
    logging.info("Validation: Play an excited rythm for 60 seconds")
    play = excited_music.play()
    play.wait_done()
    eeg = eeg_device.get_eeg_data(recording_time=60)
    eeg_samples_classes.append(generate_samples(eeg, WINDOW_SIZE, WINDOW_OVERLAP))

    # stop recording eeg
    eeg_device.stop_recording()

    #------------CLASSIFICATION----------------
    accuracy, f1 = eeg_device.get_metrics(eeg_samples_classes)

    logging.info(f"Validation: Accuracy: {accuracy:.2f} F1 Score: {f1:.2f}")
    logging.info("Validation: Validation Finished")
