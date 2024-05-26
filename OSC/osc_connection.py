import time
from pythonosc.dispatcher import Dispatcher
from pythonosc import osc_message_builder
from pythonosc import udp_client
from pythonosc import osc_server
import rtmidi
import threading
import mido

midiout = rtmidi.MidiOut()
available_ports = midiout.get_ports()
print(available_ports)
if available_ports:
    midiout.open_port(1)


# print(mido.get_output_names())
# port = mido.open_output('loopMIDI Port 1')
# msg_on = mido.Message('note_on', channel=0, note=64, velocity=127, time=0)
# msg_off = mido.Message('note_off', channel=0, note=64, velocity=127, time=0)



SERVER_IP = "127.0.0.1"
SERVER_PORT = 9000

REAPER_IP = "192.168.1.139"
REAPER_PORT = 8000



class client_OSC:
  def __init__(self, ip, port):
    self.client = udp_client.UDPClient(ip, port)

  def send(self, msg):
    self.client.send(msg)

  def send_test(self, value):
    msg = osc_message_builder.OscMessageBuilder(address = "/test")
    msg.add_arg(value)
    msg = msg.build()
    self.client.send(msg)
    print(f"Sent: {msg}")




class server_OSC:
  def __init__(self, ip, port):
    self.last_beat = None
    self.midi_play = False
    self.dispatcher = Dispatcher()
    self.midi_thread = threading.Thread(target=self.midi_thread_function)
    self.midi_thread.start()
    # self.dispatcher.map("/*", print)
    self.dispatcher.map("/beat/str", self.print_beat_handler, "Beat")
    self.server = osc_server.ThreadingOSCUDPServer((ip, port), self.dispatcher)
    print("Serving on {}".format(self.server.server_address))
    self.server.serve_forever()

  def print_beat_handler(self, unused_addr, args, bar_beat_ticks):
    bar, beat, ticks = bar_beat_ticks.split('.') 
    if self.last_beat != beat:
      # start_time = time.time()
      # port.send(msg_on)
      # print(f"Time to send: {time.time() - start_time}")
      # time.sleep(0.3)
      # port.send(msg_off)
      if beat == '1':
        self.midi_play = True
        print("[{0}] ~ {1}".format(args[0], bar_beat_ticks))
      self.last_beat = beat

  def midi_thread_function(self):
    while True:
      if self.midi_play:
        note_on = [0x90, 60, 112] # channel 1, middle C, velocity 112
        note_off = [0x80, 60, 0]
        start_time = time.time()
        time.sleep(0.03)
        while time.time() - start_time < 0.1:
          continue
        midiout.send_message(note_on)
        start_time = time.time()
        time.sleep(0.1)
        while time.time() - start_time < 0.2:
          continue
        midiout.send_message(note_off)
        self.midi_play = False
      else:
        time.sleep(0.001)


if __name__ == "__main__":
  
  client = client_OSC(REAPER_IP, REAPER_PORT)
  client.send_test(0.5)

  server = server_OSC(SERVER_IP, SERVER_PORT)


