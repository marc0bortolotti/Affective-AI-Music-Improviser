import oscP5.*;
import netP5.*;

String PROCESSING_READY_MSG = "/ready";
String CONFIDENCE_MSG = "/confidence";
String TEMPERATURE_MSG = "/temperature";
String EMOTION_MSG = "/emotion";
String EEG_DEVICES_MSG = "/eeg_devices";
String MIDI_IN_PORTS_MSG = "/midi_in_ports";
String MIDI_OUT_PORTS_MSG = "/midi_out_ports";
String MODELS_MSG = "/models";
String INSTRUMENT_MIDI_IN_PORT_MSG = "/instr_midi_in_port";
String INSTRUMENT_MIDI_OUT_PORT_MSG = "/instr_midi_out_port";
String GENERATED_MIDI_OUT_PORT_MSG = "/gen_midi_out_port";
String GENERATION_TYPE_MSG = "/gen_type";
String TX_FINISHED_MSG = "/tx_finished";
String USER_NAME_MSG = "/user_name";

boolean displayMenu = false;

class OSCClientServer{
  OscP5 oscP5;  
  int port = 7000;  
  int python_port = 9000;
  String messageAddress;
  String msgString;
  
  OSCClientServer(){
    // Initialize the OSC server
    oscP5 = new OscP5(this, port);
    println("OSC server started on port " + port);
  }

  // This function is automatically called when an OSC message is received
  void oscEvent(OscMessage message) {
    messageAddress = message.addrPattern();
    if (message.checkAddrPattern(CONFIDENCE_MSG) == true) {
      confidence = message.get(0).floatValue();
      println("Value: " + confidence); 
    }
    
    if (message.checkAddrPattern(EMOTION_MSG) == true) {
      emotion = message.get(0).floatValue();  
      println("Emotion: " + emotion);
    }
    
    if (message.checkAddrPattern(EEG_DEVICES_MSG) == true) {
      msgString = message.get(0).toString();  
      println("Devices: " + msgString);
      eegDevices.add(msgString);
    }
    
    if (message.checkAddrPattern(MIDI_IN_PORTS_MSG) == true) {
      msgString = message.get(0).toString();  
      println("Midi Input Ports: " + msgString);
      midiInputPorts.add(msgString);
    }
    
    if (message.checkAddrPattern(MIDI_OUT_PORTS_MSG) == true) {
      msgString = message.get(0).toString();  
      println("Midi Output Ports: " + msgString);
      midiOutputPorts.add(msgString);
    }
    
    if (message.checkAddrPattern(MODELS_MSG) == true) {
      msgString = message.get(0).toString();  
      println("Models: " + msgString);
      generativeModels.add(msgString);
    }
    
    if (message.checkAddrPattern(TX_FINISHED_MSG) == true) {
      displayMenu = true;
    }
  }
  
  void sendOSCMessage(String address, float value) {
    OscMessage msg = new OscMessage(address);
    msg.add(value);
    oscP5.send(msg, new NetAddress("127.0.0.1", python_port));
    println(address + ": " + value);
  }
  
  void sendOSCMessageString(String address, String value) {
    OscMessage msg = new OscMessage(address);
    msg.add(value);
    oscP5.send(msg, new NetAddress("127.0.0.1", python_port));
    println(address + ": " + value);
  }
}
