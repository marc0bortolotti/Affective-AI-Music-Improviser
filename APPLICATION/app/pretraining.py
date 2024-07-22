import time
import numpy as np
import logging
from eeg.processing import generate_samples
from eeg.classification import get_eeg_classifiers
import simpleaudio

relax_music = simpleaudio.WaveObject.from_wave_file('APPLICATION/app/music/The_Scientist.wav')
excited_music = simpleaudio.WaveObject.from_wave_file('APPLICATION/app/music/Blitzkrieg_Bop.wav')
white_noise = simpleaudio.WaveObject.from_wave_file('APPLICATION/app/music/White_Noise.wav')



def pretraining(unicorn, WINDOW_SIZE, WINDOW_OVERLAP):
    logging.info("Pretraining: Start Training")

    steps = 2

    # start recording eeg
    unicorn.start_unicorn_recording()
    time.sleep(7) # wait for signal to stabilize

    eeg_samples_baseline = []
    eeg_samples_relax = []
    eeg_samples_excited = []

    for i in range(steps):
        # Baseline (30 seconds)
        logging.info("Pretraining: Pause for 20 seconds. Please, do not move or think about anything. Just relax.")
        white_noise.play()
        eeg = unicorn.get_eeg_data(recording_time = 30)
        eeg_samples_baseline.append(generate_samples(eeg, WINDOW_SIZE, WINDOW_OVERLAP))

        # Relax (1 minute)
        logging.info("Pretraining: Play a relaxed rythm on the metronome for 30 seconds")
        relax_music.play()
        eeg = unicorn.get_eeg_data(recording_time = 60)
        eeg_samples_relax.append(generate_samples(eeg, WINDOW_SIZE, WINDOW_OVERLAP))

        # Baseline (30 seconds)
        logging.info("Pretraining: Pause for 20 seconds. Please, do not move or think about anything. Just relax.")
        white_noise.play()
        eeg = unicorn.get_eeg_data(recording_time = 30)
        eeg_samples_baseline.append(generate_samples(eeg, WINDOW_SIZE, WINDOW_OVERLAP))

        # Excited (1 minute)
        logging.info("Pretraining: Play an excited rythm on the metronome")
        excited_music.play()
        eeg = unicorn.get_eeg_data(recording_time = 60)
        eeg_samples_excited.append(generate_samples(eeg, WINDOW_SIZE, WINDOW_OVERLAP))

    eeg_samples_baseline = np.concatenate(eeg_samples_baseline)
    eeg_samples_relax = np.concatenate(eeg_samples_relax)
    eeg_samples_excited = np.concatenate(eeg_samples_excited)

    # stop recording eeg
    unicorn.stop_unicorn_recording()

    #------------CLASSIFICATION----------------
    scaler, svm_model, lda_model, baseline = get_eeg_classifiers(eeg_samples_baseline, [eeg_samples_relax, eeg_samples_excited])

    logging.info("Pretraining: Training Finished")
    return scaler, svm_model, lda_model, baseline



def validation(unicorn, WINDOW_DURATION):
    logging.info("Validation: Start Validation")

    # start recording eeg
    unicorn.start_unicorn_recording()
    time.sleep(7) # wait for signal to stabilize

    eeg_samples_classes = []

    # Relax (1 minute)
    logging.info("Validation: Play a relaxed rythm on the metronome for 30 seconds")
    relax_music.play()
    eeg = unicorn.get_eeg_data(recording_time = 60)
    eeg_samples_classes.append(generate_samples(eeg, WINDOW_DURATION))

    # Excited (1 minute)
    logging.info("Validation: Play an excited rythm on the metronome")
    excited_music.play()
    eeg = unicorn.get_eeg_data(recording_time = 60)
    eeg_samples_classes.append(generate_samples(eeg, WINDOW_DURATION))

    # stop recording eeg
    unicorn.stop_unicorn_recording()

    #------------CLASSIFICATION----------------
    accuracy, f1 = unicorn.get_metrics(eeg_samples_classes)

    logging.info(f"Validation: Accuracy: {accuracy:.2f} F1 Score: {f1:.2f}")
    logging.info("Validation: Validation Finished")
