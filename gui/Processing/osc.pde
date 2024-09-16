import oscP5.*;
import netP5.*;

class OSCClientServer{
  OscP5 oscP5;  
  int port = 7000;  
  int python_port = 9000;
  float receivedValue;
  String messageAddress;
  
  OSCClientServer(){
    // Initialize the OSC server
    oscP5 = new OscP5(this, port);
    println("OSC started on port " + port);
  }

  // This function is automatically called when an OSC message is received
  void oscEvent(OscMessage message) {
    messageAddress = message.addrPattern();
    if (message.checkAddrPattern("/confidence") == true) {
      receivedValue = message.get(0).floatValue();  
      println("Value: " + receivedValue);
      confidence = receivedValue;
    }
  }
  
  void sendOSCMessage(String address, float value) {
    OscMessage msg = new OscMessage(address);
    msg.add(value);
    oscP5.send(msg, new NetAddress("127.0.0.1", python_port));
    println(address + ": " + value);
  }
}
