import time
import numpy as np
import logging
from EEG.processing import generate_samples
import simpleaudio
import os

file_path = os.path.dirname(__file__)
relax_music = simpleaudio.WaveObject.from_wave_file(file_path + '/music/The_Scientist.wav')
excited_music = simpleaudio.WaveObject.from_wave_file(file_path + '/music/Blitzkrieg_Bop.wav')
white_noise = simpleaudio.WaveObject.from_wave_file(file_path + '/music/White_Noise.wav')

session_types = ['listening', 'playing']


def pretraining(eeg_device, WINDOW_SIZE, WINDOW_OVERLAP, steps = 1, rec_time=60):
    logging.info("Pretraining: Start Training")

    # start recording eeg
    eeg_device.start_recording()
    time.sleep(7)  # wait for signal to stabilize

    eeg_samples_baseline = []
    eeg_samples_relax = []
    eeg_samples_excited = []

    for step in range(steps):

        logging.info(f"Pretraining: Step {step+1}/{steps}")

        for session_type in session_types:

            # Baseline (max 20 seconds)
            baseline_time = min(rec_time/2, 20)
            logging.info(f"Pretraining: Pause for {baseline_time} seconds. Please, do not move or think about anything. Just relax.")
            play = white_noise.play()
            start = time.time() 
            while True:
                if time.time() - start < baseline_time:
                    time.sleep(0.2)
                else:
                    break
            play.stop()
            eeg = eeg_device.get_eeg_data(recording_time=baseline_time)
            eeg_samples_baseline.append(generate_samples(eeg, WINDOW_SIZE, WINDOW_OVERLAP))

            # Relax (1 minute)
            logging.info(f"Pretraining: Play a relaxed rythm for {rec_time} seconds")
            if session_type == 'listening':
                play = relax_music.play()
            start = time.time() 
            while True:
                if time.time() - start < rec_time:
                    time.sleep(0.2)
                else:
                    break
            if session_type == 'listening':
                play.stop()
            eeg = eeg_device.get_eeg_data(recording_time=rec_time)
            eeg_samples_relax.append(generate_samples(eeg, WINDOW_SIZE, WINDOW_OVERLAP))

            # Baseline (max 20 seconds)
            logging.info(f"Pretraining: Pause for {baseline_time} seconds. Please, do not move or think about anything. Just relax.")
            play = white_noise.play()
            start = time.time() 
            while True:
                if time.time() - start < baseline_time:
                    time.sleep(0.2)
                else:
                    break
            play.stop()
            eeg = eeg_device.get_eeg_data(recording_time=baseline_time)
            eeg_samples_baseline.append(generate_samples(eeg, WINDOW_SIZE, WINDOW_OVERLAP))

            # Excited (1 minute)
            logging.info(f"Pretraining: Play an excited rythm for {rec_time} seconds")
            if session_type == 'listening':
                play = excited_music.play()
            start = time.time() 
            while True:
                if time.time() - start < rec_time:
                    time.sleep(0.2)
                else:
                    break
            if session_type == 'listening':
                play.stop()
            eeg = eeg_device.get_eeg_data(recording_time=rec_time)
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


