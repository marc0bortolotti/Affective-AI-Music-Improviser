float confidence = 0.8;   // Confidence variable (0 to 1)
float temperature = 1.0;  // Temperature value (0 to infinite)
float emotion = 0.0; // 0 relaxed, excited otherwise
float newTemperature, newConfidence;
OSCClientServer oscClientServer;
StartMenu startMenu;
Emoticon bouncingEmoticon;
CustomSlider temperatureSlider, confidenceSlider;
ControlP5 cp5_temperature, cp5_confidence;  
boolean confidenceSliderOn = true;
boolean temperatureSliderOn = true;

void setup() {  
  
  frameRate(50);     // Set frame rate to 50 Hertz
  size(700, 600); 
  
  oscClientServer = new OSCClientServer();
  
  startMenu = new StartMenu();
  
  bouncingEmoticon = new Emoticon();
  
  cp5_temperature = new ControlP5(this);
  cp5_confidence = new ControlP5(this);
  
  if (temperatureSliderOn){
    temperatureSlider = new CustomSlider(cp5_temperature, "temperature", "TEMPERATURE", width * 7/8, height * 1/8, -8.0, 20.0, 0.0);
    temperatureSlider.updateValueLabel(1.0);
    temperatureSlider.setVisible(false);
  }
  
  if (confidenceSliderOn){
    confidenceSlider = new CustomSlider(cp5_confidence, "confidence", "CONFIDENCE", width * 7/8, height * 5/8, 0.0, 1.0, 0.1);
    confidenceSlider.setVisible(false);
  }
  
}

void draw() {  
  
  if (emotion == 0.0) {
    background(0, 255, 255);
  }
  else {
    background(255, 70, 70);
  }
  
  if (isNameEntered) {
    // Display the second page
    startMenu.displaySecondPage();
    
    if(isParameterSelected){
      bouncingEmoticon.drawEmoticon();  
      bouncingEmoticon.updatePosition();
    
      if (temperatureSliderOn){
        temperatureSlider.setVisible(true);
        newTemperature = exp(temperatureSlider.getValue());
        if (newTemperature != temperature) {
          temperature = newTemperature;
          oscClientServer.sendOSCMessage("/temperature", temperature);
        }
      }
      
      if (confidenceSliderOn){
        confidenceSlider.setVisible(true);
        confidenceSlider.setValue(confidence);
        confidenceSlider.updateValueLabel(confidence);  
      }
    } else {
      // Display the second page
      startMenu.displaySecondPage();
    }
    
  } else {
    // Display the first page for name input
    startMenu.displayFirstPage();
  }
}

void mousePressed() {
  // Check if the mouse is inside the label's rectangle
  if (mouseX > temperatureSlider.labelX && mouseX < temperatureSlider.labelX + temperatureSlider.labelWidth && 
      mouseY > temperatureSlider.labelY && mouseY < temperatureSlider.labelY + temperatureSlider.labelHeight) {
     temperatureSlider.setValue(0.0);
  }
  
  if (isNameEntered) {
    startMenu.midiInputDropdown.mousePressed();
    startMenu.midiOutputDropdown.mousePressed();
    startMenu.generativeModelDropdown.mousePressed();
    startMenu.generationTypeDropdown.mousePressed();
    startMenu.eegDeviceDropdown.mousePressed();
  }
}

void keyPressed() {
    if (!isNameEntered) {
      if (keyCode == BACKSPACE) {
        if (userName.length() > 0) {
          userName = userName.substring(0, userName.length() - 1);
        }
      } else if (keyCode == ENTER && userName.length() > 0) {
        isNameEntered = true;
      } else if (key != CODED) {
        userName += key;
      }
    }
  }
