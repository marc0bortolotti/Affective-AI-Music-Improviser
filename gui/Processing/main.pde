float confidence = 0.1;   // Confidence variable (0 to 1)
float temperature = 1.0;  // Temperature value (0 to infinite)
float newTemperature, newConfidence;
OSCClientServer oscClientServer;
Emoticon emoticon;
CustomSlider temperatureSlider, confidenceSlider;
ControlP5 cp5_temperature, cp5_confidence;  
boolean confidenceSliderOn = false;
boolean temperatureSliderOn = false;
float emotion = 0.0;

void setup() {  
  
  frameRate(50);     // Set frame rate to 50 Hertz
  size(700, 600); 
  
  oscClientServer = new OSCClientServer();
  
  emoticon = new Emoticon();
  
  cp5_temperature = new ControlP5(this);
  cp5_confidence = new ControlP5(this);
  
  if (temperatureSliderOn){
    temperatureSlider = new CustomSlider(cp5_temperature, "temperature", "TEMPERATURE", width * 7/8, height * 1/8, -8.0, 20.0, 0.0);
    temperatureSlider.updateValueLabel(1.0);
  }
  
  if (confidenceSliderOn){
    confidenceSlider = new CustomSlider(cp5_confidence, "confidence", "CONFIDENCE", width * 7/8, height * 5/8, 0.0, 1.0, 0.1);
  }
  
}

void draw() {  
  
  if (emotion == 0.0) {background(0, 255, 255);}
  else {background(255, 70, 70);}
  
  
  emoticon.drawEmoticon();  
  emoticon.updatePosition();
  
  if (temperatureSliderOn){
    newTemperature = exp(temperatureSlider.getValue());
    if (newTemperature != temperature) {
      temperature = newTemperature;
      oscClientServer.sendOSCMessage("/temperature", temperature);
    }
  }
  
  if (confidenceSliderOn){
    confidenceSlider.setValue(confidence);
    confidenceSlider.updateValueLabel(confidence);  
  }
  
}

void mousePressed() {
  // Check if the mouse is inside the label's rectangle
  if (mouseX > temperatureSlider.labelX && mouseX < temperatureSlider.labelX + temperatureSlider.labelWidth && 
      mouseY > temperatureSlider.labelY && mouseY < temperatureSlider.labelY + temperatureSlider.labelHeight) {
     temperatureSlider.setValue(0.0);
  }
}
