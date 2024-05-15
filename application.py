import UDP.udp_connection as udp
import threading
import MIDI.metronome as metronome
import MIDI.midi_in as midi_input
import logging
import threading
import time
import BCI.unicorn_brainflow as unicorn_brainflow
from BCI.utils.loader import generate_samples

format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO,datefmt="%H:%M:%S")

BPM = 120
SERVER_IP = '127.0.0.1'
SERVER_PORT = 1001



def thread_function_metronome(name):
    logging.info("Thread %s: starting", name)
    global click
    click = metronome.Metronome(SERVER_IP, SERVER_PORT, BPM)
    click.start()



def thread_function_unicorn(name):
    logging.info("Thread %s: starting", name)
    eeg = unicorn_brainflow.collect_unicorn_data()
    samples = generate_samples(eeg)


def thread_function_midi(name):
    logging.info("Thread %s: starting", name)
    global midi
    midi = midi_input.MIDI_Input(server_ip = SERVER_IP, server_port = SERVER_PORT)
    midi.run()



if __name__ == "__main__":

    logging.info("Main: before creating threads")
    thread_click = threading.Thread(target=thread_function_metronome, args=('Click',))
    thread_midi = threading.Thread(target=thread_function_midi, args=('MIDI Input',))
    # thread_unicorn = threading.Thread(target=thread_function_unicorn, args=('unicorn',))
    logging.info("Main: before running thread")
    thread_click.start()
    thread_midi.start()
    # thread_unicorn.start()
    logging.info("Main: wait for the thread to finish")

    time.sleep(60)
    click.stop()
    midi.close()

    thread_click.join()
    thread_midi.join()
    # thread_unicorn.join()
    logging.info("Thread MIDI Input: finishing")
    logging.info("Thread Click: finishing")
    # logging.info("Thread unicorn: finishing")
    logging.info("Main: all done")