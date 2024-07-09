import time
import numpy as np
import logging
from eeg.processing import generate_samples


def countdown(duration):
    for i in range(0, duration):
        print('Countdown: {:2d} seconds left'.format(duration - i), end='\r')
        time.sleep(1)
    print('Countdown:  0 seconds left\n')


def pretraining(unicorn, WINDOW_SIZE, WINDOW_OVERLAP):
    logging.info("Pretraining: Start Training")

    # start recording eeg
    unicorn.start_unicorn_recording()

    # Baseline
    logging.info("Pretraining: Pause for 20 seconds. Please, do not move or think about anything. Just relax.")
    countdown(20+5) # Wait 5 seconds for the unicorn signal to stabilize
    eeg = unicorn.get_eeg_data(recording_time = 20)
    eeg_samples_baseline_1 = generate_samples(eeg, WINDOW_SIZE, WINDOW_OVERLAP)

    # Relax 
    logging.info("Pretraining: Play a relaxed rythm on the metronome for 30 seconds")
    countdown(30)
    eeg = unicorn.get_eeg_data(recording_time = 30)
    eeg_samples_relax = generate_samples(eeg, WINDOW_SIZE, WINDOW_OVERLAP)

    # Baseline
    logging.info("Pretraining: Pause for 20 seconds. Please, do not move or think about anything. Just relax.")
    countdown(20)
    eeg = unicorn.get_eeg_data(recording_time = 20)
    eeg_samples_baseline_2 = generate_samples(eeg, WINDOW_SIZE, WINDOW_OVERLAP)

    # Excited
    logging.info("Pretraining: Play an excited rythm on the metronome")
    countdown(30)
    eeg = unicorn.get_eeg_data(recording_time = 30)
    eeg_samples_excited = generate_samples(eeg, WINDOW_SIZE, WINDOW_OVERLAP)


    logging.info("Pretraining: Training Finished")
    eeg_samples_baseline = np.concatenate((eeg_samples_baseline_1, eeg_samples_baseline_2))

    # stop recording eeg
    unicorn.stop_unicorn_recording()

    #------------CLASSIFICATION----------------
    scaler, svm_model, lda_model, baseline = unicorn.eeg_classification(eeg_samples_baseline, [eeg_samples_relax, eeg_samples_excited])

    return scaler, svm_model, lda_model, baseline