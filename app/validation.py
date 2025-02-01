import os
import time
import logging
from EEG.processing import generate_samples
from EEG.classifier import load_eeg_classifier
import pygame

pygame.mixer.init()
white_noise = pygame.mixer.Sound('app/music/White_Noise.wav')

def validation(training_path, metrics_path, eeg_device, window_size, window_overlap, rec_time=60):
    
    logging.info("Validation: Start Validation")

    try:
        # Load the EEG classifier from the file
        scaler, lda_model, svm_model, baseline = load_eeg_classifier(training_path)
    except:
        logging.error("No classifier found. Please, train the classifier first.")
        exit()


    for classifier in ['LDA', 'SVM']: 

        print(f'\n{classifier}')

        if classifier == 'LDA':
            eeg_device.set_classifier(baseline=baseline, classifier=lda_model, scaler=scaler)
        else:
            eeg_device.set_classifier(baseline=baseline, classifier=svm_model, scaler=scaler)

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

        eeg_device.save_session(os.path.join(training_path, 'validation_lda.csv'))

        with open(metrics_path, 'a') as f:
            f.write('\n\nVALIDATION\n')
            f.write(f'{classifier}-Accuracy: {accuracy}\n{classifier}-F1 Score: {f1}')


