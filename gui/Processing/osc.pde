import oscP5.*;
import netP5.*;

class OSCClientServer{
  OscP5 server_oscP5;  // OSC object
  OscP5 client_oscP5;  // OSC object
  int serverPort = 7000;  // Port number to listen for incoming messages
  int clientPort = 9000;  // Port number to send messagges (temperature value)
  float receivedValue;
  String messageAddress;
  
  OSCClientServer(){
    // Initialize the OSC server
    server_oscP5 = new OscP5(this, serverPort);
    println("OSC server started on port " + serverPort);
    
    // Initialize the OSC client
    client_oscP5 = new OscP5(this, clientPort);
    println("OSC server started on port " + clientPort);
  }

  // This function is automatically called when an OSC message is received
  void oscEvent(OscMessage message) {
    // Store the address of the OSC message
    messageAddress = message.addrPattern();
    
    if (message.checkAddrPattern("/confidence") == true) {
      receivedValue = message.get(0).floatValue();  // Get the float value
      println("Value: " + receivedValue);
      confidence = receivedValue;
    }
  }
  
  void sendOSCMessage(String address, float value) {
    OscMessage msg = new OscMessage(address);
    // Add values to the OSC message
    msg.add(value);
    // Send the message
    client_oscP5.send(msg, new NetAddress("127.0.0.1", clientPort));
    println(address + ": " + value);
  }
}
