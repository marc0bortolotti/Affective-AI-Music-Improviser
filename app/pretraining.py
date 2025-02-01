import time
import numpy as np
import logging
from EEG.processing import generate_samples, convert_to_mne
from EEG.classifier import fit_eeg_classifier, save_eeg_classifier
import os
import asrpy
import pygame

pygame.mixer.init()
relax_music = pygame.mixer.Sound('app/music/The_Scientist.wav')
excited_music = pygame.mixer.Sound('app/music/Blitzkrieg_Bop.wav')
white_noise = pygame.mixer.Sound('app/music/White_Noise.wav')

session_types = ['Listening', 'Playing']

def pretraining(training_path, metrics_path, eeg_device, WINDOW_SIZE, WINDOW_OVERLAP, steps = 1, rec_time=60):
    
    logging.info("Pretraining: Start Training")

    # start recording eeg
    eeg_device.start_recording()
    eeg_device.insert_marker('T')
    time.sleep(5)  # wait for signal to stabilize

    # Rest
    logging.info(f"Pretraining: Resting for {rec_time} seconds")
    eeg_device.insert_marker('RS')
    start = time.time() 
    while True:
        if time.time() - start < rec_time:
            time.sleep(0.2)
        else:
            break    

    eeg_rest = eeg_device.get_eeg_data(recording_time=rec_time)
    trigger = np.zeros(len(eeg_rest))
    eeg_rest = convert_to_mne(eeg_rest, trigger, eeg_device.sample_frequency, eeg_device.ch_names)
    print(eeg_rest)
    asr = asrpy.ASR(sfreq=eeg_device.sample_frequency, cutoff=15)
    asr.fit(eeg_rest)
    eeg_device.set_asr(asr)

    eeg_samples_baseline = []
    eeg_samples_relax = []
    eeg_samples_excited = []

    for step in range(steps):

        logging.info(f"Pretraining: Step {step+1}/{steps}")

        for session_type in session_types:

            # Baseline
            baseline_time = min(rec_time/2, 30)
            logging.info(f"Pretraining: Pause for {baseline_time} seconds.")
            eeg_device.insert_marker('WN')
            white_noise.play()
            start = time.time() 
            while True:
                if time.time() - start < baseline_time:
                    time.sleep(0.2)
                else:
                    break
            white_noise.stop()
            eeg = eeg_device.get_eeg_data(recording_time=baseline_time)
            eeg_samples_baseline.append(generate_samples(eeg, WINDOW_SIZE, WINDOW_OVERLAP))

            # Relax
            logging.info(f"Pretraining: {session_type} relaxed for {rec_time} seconds.")
            eeg_device.insert_marker('PTR')
            if session_type == 'Listening':
                relax_music.play()
            start = time.time() 
            while True:
                if time.time() - start < rec_time:
                    time.sleep(0.2)
                else:
                    break
            if session_type == 'Listening':
                relax_music.stop()
            eeg = eeg_device.get_eeg_data(recording_time=rec_time)
            eeg_samples_relax.append(generate_samples(eeg, WINDOW_SIZE, WINDOW_OVERLAP))

            # Baseline
            logging.info(f"Pretraining: Pause for {baseline_time} seconds.")
            eeg_device.insert_marker('WN')
            white_noise.play()
            start = time.time() 
            while True:
                if time.time() - start < baseline_time:
                    time.sleep(0.2)
                else:
                    break
            white_noise.stop()
            eeg = eeg_device.get_eeg_data(recording_time=baseline_time)
            eeg_samples_baseline.append(generate_samples(eeg, WINDOW_SIZE, WINDOW_OVERLAP))

            # Excited 
            logging.info(f"Pretraining: {session_type} excited for {rec_time} seconds")
            eeg_device.insert_marker('PTE')
            if session_type == 'Listening':
                excited_music.play()
            start = time.time() 
            while True:
                if time.time() - start < rec_time:
                    time.sleep(0.2)
                else:
                    break
            if session_type == 'Listening':
                excited_music.stop()
            eeg = eeg_device.get_eeg_data(recording_time=rec_time)
            eeg_samples_excited.append(generate_samples(eeg, WINDOW_SIZE, WINDOW_OVERLAP))

    # stop recording eeg
    eeg_device.stop_recording()

    eeg_samples_baseline = np.concatenate(eeg_samples_baseline)
    eeg_samples_relax = np.concatenate(eeg_samples_relax)
    eeg_samples_excited = np.concatenate(eeg_samples_excited)
    eeg_samples_classes = [eeg_samples_relax, eeg_samples_excited]

    #------------CLASSIFICATION----------------
    scaler, svm_model, lda_model, baseline, accuracy_lda, f1_lda, accuracy_svm, f1_svm = fit_eeg_classifier(eeg_samples_baseline, 
                                                                                                            eeg_samples_classes, 
                                                                                                            eeg_device.sample_frequency, 
                                                                                                            eeg_device.ch_names, 
                                                                                                            asr=asr)
                                                                                                        
    logging.info("Pretraining: Training Finished")

    # Save the EEG classifier and the EEG raw data
    
    save_eeg_classifier(training_path, scaler=scaler, svm_model=svm_model, lda_model=lda_model, baseline=baseline)

    eeg_device.save_session(os.path.join(training_path, 'training.csv'))


    with open(metrics_path, 'w') as f:
        f.write('TRAINING\n')
        f.write(f'LDA-Accuracy: {accuracy_lda}\nLDA-F1 Score: {f1_lda}\nSVM-Accuracy: {accuracy_svm}\nSVM-F1 Score: {f1_svm}')
    


