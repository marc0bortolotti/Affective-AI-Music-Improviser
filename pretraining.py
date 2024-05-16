import logging
import threading
import numpy as np
from BCI.utils.loader import generate_samples
from BCI.utils.eeg_classification import eeg_classification
import time


def countdown(duration):
    for i in range(0, duration):
        print('Countdown: {:2d} seconds left'.format(duration - i), end='\r')
        time.sleep(1)
    print('Countdown:  0 seconds left\n')


def thread_function_metronome(name, click):
    logging.info("Thread %s: starting", name)
    click.start()


def pretraining(unicorn, click, window_size, window_overlap):
    logging.info("Pretraining: Start Training")

    # Baseline
    logging.info("Pretraining: Pause for 20 seconds. Please, do not move or think about anything. Just relax.")
    unicorn.start_unicorn_recording()
    countdown(20+5) # Wait 5 seconds for the unicorn signal to stabilize
    eeg = unicorn.get_eeg_data(recording_time = 20)
    eeg_samples_baseline_1 = generate_samples(eeg, window_size, window_overlap)

    # Relax 
    logging.info("Pretraining: Play a relaxed rythm on the metronome for 30 seconds")
    thread_click = threading.Thread(target=thread_function_metronome, args=('Click', click))
    thread_click.start()
    countdown(30)
    eeg = unicorn.get_eeg_data(recording_time = 30)
    eeg_samples_relax = generate_samples(eeg, window_size, window_overlap)
    click.stop()
    thread_click.join()
    logging.info("Thread Click: finishing")

    # Baseline
    logging.info("Pretraining: Pause for 20 seconds. Please, do not move or think about anything. Just relax.")
    countdown(20)
    eeg = unicorn.get_eeg_data(recording_time = 20)
    eeg_samples_baseline_2 = generate_samples(eeg, window_size, window_overlap)

    # Excited
    logging.info("Pretraining: Play an excited rythm on the metronome")
    thread_click = threading.Thread(target=thread_function_metronome, args=('Click', click))
    thread_click.start()
    countdown(30)
    eeg = unicorn.get_eeg_data(recording_time = 30)
    eeg_samples_excited = generate_samples(eeg, window_size, window_overlap)
    click.stop()
    thread_click.join()
    logging.info("Thread Click: finishing")
    logging.info("Pretraining: Training Finished")
    eeg_samples_baseline = np.concatenate((eeg_samples_baseline_1, eeg_samples_baseline_2))

    #------------CLASSIFICATION----------------
    scaler, svm_model, lda_model, baseline = eeg_classification(eeg_samples_baseline, [eeg_samples_relax, eeg_samples_excited])

    return scaler, svm_model, lda_model, baseline