import oscP5.*;
import netP5.*;

OscP5 oscP5;  // OSC object
int port = 12000;  // Port number to listen for incoming messages

// Variables to store incoming data
float receivedValue = 0;
String messageAddress = "";

// This function is automatically called when an OSC message is received
void oscEvent(OscMessage message) {
  // Print the address pattern of the received message
  println("Received an OSC message with address: " + message.addrPattern());
  
  // Store the address of the OSC message
  messageAddress = message.addrPattern();
  
  // Check if the message has at least one argument
  if (message.checkAddrPattern("/test") == true) {
    // Extract the first argument (assuming it's a float)
    if (message.checkTypetag("f")) {  // "f" stands for float
      receivedValue = message.get(0).floatValue();  // Get the float value
      println("Value: " + receivedValue);
    }
  }
}
