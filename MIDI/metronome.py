import simpleaudio
from time import perf_counter, sleep
import sys
sys.path.append('..')
import UDP.udp_connection as udp
import logging


# define class metronome
class Metronome:
    def __init__(self, server_ip, server_port, BPM=120):
        self.BPM = BPM
        self.stop_execution = False
        self.lastTime = perf_counter()
        self.delay = 4 * (60/self.BPM)
        self.click = simpleaudio.WaveObject.from_wave_file('MIDI/click_samples/click_120.wav')
        self.client = udp.Client('metronome')
        self.server_ip = server_ip
        self.server_port = server_port

    def start(self):
        while not self.stop_execution:
            sleep(self.delay / 3.0)
            currentTime = perf_counter()

            while True:
                currentTime = perf_counter()
                if (currentTime - self.lastTime >= self.delay):
                    self.click.play()
                    self.client.send("click", self.server_ip, self.server_port)
                    self.lastTime = currentTime
                    break
        logging.info("Metronome: stopped")
        self.client.close()

    def stop(self):
        self.stop_execution = True