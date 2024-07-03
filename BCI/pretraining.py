import logging
import threading
import numpy as np
from BCI.utils.loader import generate_samples
from BCI.utils.feature_extraction import calculate_baseline, extract_features, baseline_correction
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score
from sklearn import svm
import time


def preprocessing(eeg_samples_baseline, eeg_samples_list):
    # Calculate baseline
    baseline = calculate_baseline(eeg_samples_baseline)

    eeg_features_list = []
    for eeg_samples in eeg_samples_list:
        eeg_features = extract_features(eeg_samples)
        # Apply baseline correction
        eeg_features_corrected = baseline_correction(eeg_features, baseline)
        eeg_features_list.append(eeg_features_corrected)

    return eeg_features_list, baseline
    

def eeg_classification(eeg_samples_baseline, eeg_samples_list):

    # Preprocessing (Feature extraction and Baseline correction)
    eeg_features_list, baseline = preprocessing(eeg_samples_baseline, eeg_samples_list)

    # Prepare the data for classification
    X = np.concatenate(eeg_features_list)
    y = np.concatenate([np.ones(eeg_features.shape[0]) * i for i, eeg_features in enumerate(eeg_features_list)])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33)

    # Normalization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test) 

    # LDA
    lda_model = LinearDiscriminantAnalysis()
    lda_model.fit(X_train, y_train)
    y_lda_pred = lda_model.predict(X_test)
    y_lda_pred_proba = lda_model.predict_proba(X_test)
    accuracy_lda = accuracy_score(y_test, y_lda_pred)
    f1_lda = f1_score(y_test, y_lda_pred, average='macro')
    logging.info(f'LDA\tAccuracy: {accuracy_lda:.2f} F1 Score: {f1_lda:.2f}')

    # SVM
    svm_model = svm.SVC(kernel = 'linear')
    svm_model.fit(X_train, y_train) 
    y_svm_pred = svm_model.predict(X_test)
    accuracy_svm = accuracy_score(y_test, y_svm_pred)
    f1_svm = f1_score(y_test, y_svm_pred, average='macro')
    logging.info(f'SVM\tAccuracy: {accuracy_svm:.2f} F1 Score: {f1_svm:.2f}')

    return scaler, svm_model, lda_model , baseline

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