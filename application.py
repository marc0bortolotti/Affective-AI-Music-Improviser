import UDP.udp_connection as udp
import threading
import MIDI.metronome as metronome
import logging
import threading
import time
import BCI.unicorn_brainflow as unicorn_brainflow

format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO,datefmt="%H:%M:%S")

BPM = 120
SERVER_IP = '127.0.0.1'
SERVER_PORT = 1001

    
def thread_function_server(name):
    logging.info("Thread %s: starting", name)
    global server
    server = udp.Server(ip = SERVER_IP, port = SERVER_PORT, buffer_size = 100, window_size = 10)
    server.start()



def thread_function_metronome(name):
    logging.info("Thread %s: starting", name)
    global click
    click = metronome.Metronome(SERVER_IP, SERVER_PORT, BPM)
    click.start()



if __name__ == "__main__":

    logging.info("Main: before creating threads")
    thread_server = threading.Thread(target=thread_function_server,  args=('server',))
    thread_metronome= threading.Thread(target=thread_function_metronome, args=('metronome',))
    logging.info("Main: before running thread")
    thread_server.start()
    thread_metronome.start()
    logging.info("Main: wait for the thread to finish")
    time.sleep(20)
    click.stop()
    server.close()
    thread_server.join()
    thread_metronome.join()
    logging.info("Thread server: finishing")
    logging.info("Thread metronome: finishing")
    logging.info("Main: all done")