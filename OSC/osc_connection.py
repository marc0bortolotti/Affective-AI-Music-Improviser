import time
from pythonosc.dispatcher import Dispatcher
from pythonosc import osc_server
from pythonosc.udp_client import SimpleUDPClient
import logging

REC_MSG = '/action/40046'
MOVE_CURSOR_TO_ITEM_END_MSG = '/action/41174'
MOVE_CURSOR_TO_NEXT_MEASURE_MSG = '/action/41040'
CONFIDENCE_MSG = '/confidence'
EMOTION_MSG = '/emotion'
EEG_DEVICES_MSG = '/eeg_devices'
MIDI_IN_PORTS_MSG = '/midi_in_ports'
MIDI_OUT_PORTS_MSG = '/midi_out_ports'
MODELS_MSG = '/models'
TEMPERATURE_MSG = '/temperature'
INSTRUMENT_MIDI_IN_PORT_MSG = "/instr_midi_in_port"
INSTRUMENT_MIDI_OUT_PORT_MSG = "/instr_midi_out_port"
GENERATED_MIDI_OUT_PORT_MSG = "/gen_midi_out_port"
GENERATION_TYPE_MSG = "/gen_type"
TX_FINISHED_MSG = '/tx_finished'
USER_NAME_MSG = '/user_name'
PROCESSING_READY_MSG = "/ready"
POPUP_MSG = "/popup"


ESTIMATED_LATENCY_FOR_RX = 0.28 # in seconds


class Client_OSC:
  def __init__(self, parse_message = False):
    self.parse_message = parse_message

  def send(self, ip, port, msg, value):
    self.client = SimpleUDPClient(ip, port)
    self.client.send_message(msg, value)
    if self.parse_message:
      logging.info(f"OSC Client: sent message <<{msg, value}>> on {ip}:{port}")



class Server_OSC:

  def __init__(self, ip, port, bpm, parse_message = False):

    # Variables for the server
    self.parse_message = parse_message
    self.ip = ip
    self.port = port
    self.exit = False

    # Variables for synchronization
    self.BEAT_DURATION = 60/bpm
    self.last_beat = None 
    self.record_started = False
    self.synch_event = None

    # Variables for the application
    self.temperature = 1.0
    self.instrument_midi_in_port = None
    self.instrument_midi_out_port = None
    self.generated_midi_out_port = None
    self.generation_type = None
    self.eeg_device = None
    self.user_name = None

    self.parameters_setted = False
    self.processing_ready = False

    # OSC manager
    self.dispatcher = Dispatcher()
    # self.dispatcher.map("/*", self.print_message)
    self.dispatcher.map("/record", self.record_handler, "Record")
    self.dispatcher.map("/beat/str", self.beat_handler, "Beat")
    self.dispatcher.map(TEMPERATURE_MSG, self.temperature_handler, "Temperature")
    self.dispatcher.map(INSTRUMENT_MIDI_IN_PORT_MSG, self.instrument_midi_in_port_handler, "Instrument MIDI IN Port")
    self.dispatcher.map(INSTRUMENT_MIDI_OUT_PORT_MSG, self.instrument_midi_out_port_handler, "Instrument MIDI OUT Port")
    self.dispatcher.map(GENERATED_MIDI_OUT_PORT_MSG, self.generated_midi_out_port_handler, "Generated MIDI OUT Port")
    self.dispatcher.map(GENERATION_TYPE_MSG, self.generation_type_handler, "Generation Type")
    self.dispatcher.map(EEG_DEVICES_MSG, self.eeg_device_handler, "EEG Device")
    self.dispatcher.map(TX_FINISHED_MSG, self.tx_finished_handler, "Transmission Finished")
    self.dispatcher.map(USER_NAME_MSG, self.user_name_handler, "User Name")
    self.dispatcher.map(PROCESSING_READY_MSG, self.processing_ready_handler, "Processing Ready")

    self.server = osc_server.ThreadingOSCUDPServer((self.ip, self.port), self.dispatcher)

  def run(self):
    logging.info("OSC Server: running on {}".format(self.server.server_address))
    self.server.serve_forever()

  def print_message(self, unused_addr, args):
    if self.parse_message:
      logging.info(args)

  def set_event(self, event):
    self.synch_event = event

  def beat_handler(self, unused_addr, args, bar_beat_ticks):
    if not self.exit:
      bar, beat, ticks = bar_beat_ticks.split('.')
      beat = int(beat)
      bar = int(bar)
      ticks = int(ticks) 
      if self.last_beat != beat and bar > 1 and self.record_started:
        self.last_beat = beat # to avoid multiple messages arriving during the while loop
        start_time = time.time()
        delay = (self.BEAT_DURATION * ticks) / 100 # ticks are in percentage of the beat duration
        if self.parse_message:
          logging.info("{0}: {1}".format(args[0], bar_beat_ticks))
          logging.info(f'Delay: {delay} ms')
        if beat == 4:
          while True:
            if time.time() - start_time >= (self.BEAT_DURATION - delay - ESTIMATED_LATENCY_FOR_RX):
              self.synch_event.set()
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

  def temperature_handler(self, unused_addr, args, temperature):
    if self.parse_message:
      logging.info(f"Temperature: {temperature}")
    self.temperature = float(temperature)

  def instrument_midi_in_port_handler(self, unused_addr, args, port):
    if self.parse_message:
      logging.info(f"Instrument MIDI IN Port: {port}")
    self.instrument_midi_in_port = port

  def instrument_midi_out_port_handler(self, unused_addr, args, port):
    if self.parse_message:
      logging.info(f"Instrument MIDI OUT Port: {port}")
    self.instrument_midi_out_port = port
  
  def generated_midi_out_port_handler(self, unused_addr, args, port):
    if self.parse_message:
      logging.info(f"Generated MIDI OUT Port: {port}")
    self.generated_midi_out_port = port
  
  def generation_type_handler(self, unused_addr, args, generation_type):
    if self.parse_message:
      logging.info(f"Generation Type: {generation_type}")
    self.generation_type = generation_type

  def eeg_device_handler(self, unused_addr, args, eeg_device):
    if self.parse_message:
      logging.info(f"EEG Device: {eeg_device}")
    self.eeg_device = eeg_device
  
  def tx_finished_handler(self, unused_addr, args, tx_finished):
    if self.parse_message:
      logging.info(f"Transmission Finished")
    self.parameters_setted =  True

  def user_name_handler(self, unused_addr, args, user_name):
    if self.parse_message:
      logging.info(f"User Name: {user_name}")
    self.user_name = user_name

  def processing_ready_handler(self, unused_addr, args, ready):
    if self.parse_message:
      logging.info(f"Processing Ready: {ready}")
    self.processing_ready = True
    
  def popup_handler(self, unused_addr, args, answer):
    if self.parse_message:
      logging.info(f"Popup: {answer}")

  def get_temperature(self):
    return self.temperature

  def close(self):
    self.exit = True
    self.server.shutdown()
    self.server.server_close()
    logging.info("OSC Server: closed")




if __name__ == "__main__":
  client = Client_OSC()
  while True:
    emotion = input("Emotion: ")
    client.send('127.0.0.1', 7000, '/emotion', float(emotion))
  