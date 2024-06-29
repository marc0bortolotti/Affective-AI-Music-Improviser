import time
from pythonosc.dispatcher import Dispatcher
from pythonosc import osc_message_builder
from pythonosc import udp_client
from pythonosc import osc_server
import os
import mido
import sys 
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from UDP.udp_connection import Server_UDP, Client_UDP
import threading
import logging


SYNCH_MSG = "Click:1"
REC_MSG = '/action/_SWS_RECTOGGLE'



class Client_OSC:
  def __init__(self, ip, port, parse_message = False):
    self.ip = ip
    self.port = port
    self.parse_message = parse_message
    self.client = udp_client.UDPClient(ip, port)

  def send(self, msg):
    decoded_msg = osc_message_builder.OscMessageBuilder(address = msg).build()
    self.client.send(decoded_msg)
    if self.parse_message:
      logging.info(f"OSC Client: sent message <<{msg}>> to Reaper on {self.ip}:{self.port}")



class Server_OSC:

  def __init__(self, self_ip, self_port, udp_ip, udp_port, bpm = 120, parse_message = False):
    self.parse_message = parse_message
    
    self.BEAT_DURATION = 60/bpm
    self.last_beat = None 
    self.record_started = False
    self.exit = False

    # to send a synchronization msg
    self.udp_ip = udp_ip 
    self.udp_port = udp_port
    self.udp_client = Client_UDP('Click', parse_message=self.parse_message)

    # OSC manager
    self.dispatcher = Dispatcher()
    # self.dispatcher.map("/*", self.print_message)
    self.dispatcher.map("/record", self.record_handler, "Record")
    self.dispatcher.map("/beat/str", self.beat_handler, "Beat")

    self.server = osc_server.ThreadingOSCUDPServer((self_ip, self_port), self.dispatcher)
  

  def run(self):
    logging.info("OSC Server: running on {}".format(self.server.server_address))
    self.server.serve_forever()


  def print_message(self, unused_addr, args):
    if self.parse_message:
      logging.info(args)

  def beat_handler(self, unused_addr, args, bar_beat_ticks):
    if not self.exit:
      start_time = time.time()
      bar, beat, ticks = bar_beat_ticks.split('.')
      ticks = int(ticks) 
      bar = int(bar)
      beat = int(beat)
      if self.last_beat != beat and bar > 1 and self.record_started:
        self.last_beat = beat # to avoid multiple messages arriving during the while loop
        delay = (self.BEAT_DURATION * ticks) / 100 # ticks are in percentage of the beat duration
        if self.parse_message:
          logging.info("{0}: {1}".format(args[0], bar_beat_ticks))
          logging.info(f'Delay: {delay} ms')
        if beat == 4:
          while True:
            if time.time() - start_time >= (self.BEAT_DURATION/2) - delay:
              self.udp_client.send(SYNCH_MSG, self.udp_ip, self.udp_port)
              self.udp_client.send(SYNCH_MSG, self.udp_ip, 1111)
              break 
            else:
              time.sleep(0.001)

  def record_handler(self, unused_addr, args, record):
    if record == 1.0:
      if self.parse_message:
        logging.info("OSC Server: <<start record>> from Reaper")
      self.record_started = True
    else:
      if self.parse_message:
        logging.info("OSC Server: <<stop record>> from Reaper")
      self.record_started = False

  def close(self):
    self.exit = True
    self.udp_client.close()
    self.server.shutdown()
    logging.info("OSC Server: closed")










if __name__ == "__main__":

  # def udp_server_thread_function(udp_server):
  #   # logging.info(mido.get_output_names())
  #   playing_port = mido.open_output('loopMIDI Port Playing 2')
  #   recording_port = mido.open_output('loopMIDI Port Recording 3')
  #   mid = mido.MidiFile(os.path.join(MIDI_FOLDER_PATH, 'example/bass_one_bar.MID'))
  #   udp_server.start()

  #   while True:
  #     if udp_server.get_message():
  #       for msg in mid.play():
  #           playing_port.send(msg)
  #     else:
  #       time.sleep(0.001)

  # format = "%(asctime)s: %(message)s"
  # logging.basicConfig(format=format, level=logging.INFO,datefmt="%H:%M:%S")

  # MIDI_FOLDER_PATH = 'C:/Users/Gianni/Desktop/MARCO/UNI/Magistrale/TESI/Code/MIDI'

  # OSC_SERVER_IP = "127.0.0.1"
  # OSC_SERVER_PORT = 9000

  REAPER_IP = "127.0.0.1"
  REAPER_PORT = 8000

  # MIDI_IP = "127.0.0.1"
  # MIDI_PORT = 7000

  # udp_server = Server_UDP(MIDI_IP, MIDI_PORT, parse_message = True)
  # udp_server_thread = threading.Thread(target = udp_server_thread_function, args = (udp_server,))
  # udp_server_thread.start()
  
  client = Client_OSC(REAPER_IP, REAPER_PORT, parse_message = True)
  client.send(REC_MSG)

  # server_osc = Server_OSC(OSC_SERVER_IP, OSC_SERVER_PORT, MIDI_IP, MIDI_PORT, bpm = 120, parse_message = True)


