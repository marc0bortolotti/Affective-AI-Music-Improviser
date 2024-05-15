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
        self.delay = 4 * (60/self.BPM)
        self.click = simpleaudio.WaveObject.from_wave_file('MIDI/click_samples/click_120.wav')
        self.client = udp.Client('Click')
        self.server_ip = server_ip
        self.server_port = server_port

    def start(self):
        currentTime = perf_counter()
        while not self.stop_execution:
            lastTime = currentTime
            sleep(self.delay / 3.0)
            while True:
                currentTime = perf_counter()
                if (currentTime - lastTime >= self.delay):
                    self.click.play()
                    self.client.send("CLICK: 1/4", self.server_ip, self.server_port)
                    break
        logging.info("Click: stopped")
        self.client.close()

    def stop(self):
        self.stop_execution = True