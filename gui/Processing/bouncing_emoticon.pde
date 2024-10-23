
class Emoticon{
  
  int ySpeed = 2;              // Speed of the emoticon in the y direction
  int glowDuration = 5000;  // Minimum duration for high confidence to activate glow (5 seconds)
  int emoticonSize;   // Diameter of the emoticon
  boolean sustainedHigh = false; // Tracks whether confidence has been high for 5 seconds
  int firstHighTime = 0;         // The last time high confidence was registered
  int currentTime = 0;      // Tracks elapsed time
  boolean start_count = true;
  float pulseRadius = 0;    // Pulse size for glow effect
  float pulseSpeed = 120 / 60; // Pulsating speed (120 beats per minute)
  float pulseDirection = 1; // Pulse direction for glow
  int emoticonRadius;       // Radius of the emoticon
  int x, y;                // Position of the emoticon
  

  Emoticon(){
    emoticonSize = width * 3/5;
    emoticonRadius = emoticonSize / 2; // Calculate the radius
    int offset = 0;
    if(temperatureSliderOn || confidenceSliderOn){
      offset = - 50;
    }
    x = width / 2 + offset;     // Keep the emoticon horizontally centered 
    y = height / 2;     // Start in the middle vertically
  }
  
  // Draws the emoticon based on the current confidence level
  void drawEmoticon(){
    
    // Track elapsed time to check for sustained high confidence
    currentTime = millis();
  
    // Check if confidence has been high for more than 5 seconds
    if (confidence > 0.8) {
      
      if (start_count){
        firstHighTime = millis();
        start_count = false;
      }
      
      if (currentTime - firstHighTime >= glowDuration) {
        sustainedHigh = true;  // Glow effect starts
      }
      else {
      sustainedHigh = false;   // Reset glow if confidence drops
      }
      
    }
    else{
      start_count = true;
      sustainedHigh = false;
    }
    
    // If glow is active for sustained high confidence
    if (sustainedHigh) {
      drawGlow();
    }
       
    fill(255, 204, 0);  // Yellow for the face
    noStroke();
    ellipse(x, y, emoticonSize, emoticonSize);  // Draw the face
  
    // Draw the eyes relative to the radius
    drawEyes();
  
    // Draw the mouth based on confidence
    drawMouth();
  }
  
  // Draws the glow behind the face for high confidence
  void drawGlow() {
    int glowAlpha = 150;      // Glow transparency
    noStroke();
    fill(255, 255, 0, glowAlpha);  // Semi-transparent yellow glow
    ellipse(x, y, emoticonSize + pulseRadius, emoticonSize + pulseRadius);  // Pulsating circle
  }
  
  // Function to draw the eyes with pupils and iris based on confidence
  void drawEyes() {
    int eyeHeight = emoticonRadius / 2;  // Eye width is half the radius
    int eyeWidth = eyeHeight * 3 / 4;   // Eye height is slightly smaller to create the "egg" shape
    int eyeOffsetX = emoticonRadius / 3;  // Horizontal offset of eyes from the center
    int eyeOffsetY = emoticonRadius / 3;  // Vertical offset of eyes from the center
    int pupilHeigth = eyeHeight / 2;  // Pupil size relative to eye size
    int pupilWidth = eyeWidth / 2;  // Pupil size relative to eye size
    float irisSize = eyeWidth / 1.5; // Iris size relative to eye size
  
    fill(255);  // White for the eyes
  
    // Draw left eye
    ellipse(x - eyeOffsetX, y - eyeOffsetY, eyeWidth, eyeHeight);
    // Draw right eye
    ellipse(x + eyeOffsetX, y - eyeOffsetY, eyeWidth, eyeHeight);
  
    // Draw pupils and irises based on confidence
    if (confidence < 0.6) {
      // Low confidence - eyes look away
      fill(0);  // Black for the pupils
      ellipse(x - eyeOffsetX - eyeWidth / 4, y - eyeOffsetY, pupilWidth, pupilHeigth);  // Left pupil
      ellipse(x + eyeOffsetX - eyeWidth / 4, y - eyeOffsetY, pupilWidth, pupilHeigth);  // Right pupil
    } else {
      fill(0);  // Black for the pupils
      ellipse(x - eyeOffsetX, y - eyeOffsetY, pupilWidth, pupilHeigth);  // Left pupil
      ellipse(x + eyeOffsetX, y - eyeOffsetY, pupilWidth, pupilHeigth);  // Right pupil
    }
  }
  
  // Function to draw the mouth based on confidence
  void drawMouth() {
    noFill(); 
    stroke(0); //set the color of the outline of shapes, such as lines, points, rectangles, ellipses, arcs, etc
    strokeWeight(3);
  
    int smileWidth = emoticonRadius;    // Smile width is proportional to the radius
    int smileHeight = emoticonRadius / 2; // Smile height is half the radius
    int smileOffsetY = emoticonRadius / 4; // Vertical offset of the smile from the center
  
    // Draw mouth expression based on confidence level
    if (confidence < 0.2) {
      // Low confidence - frown
      arc(x, y + smileOffsetY + 20, smileWidth, smileHeight, PI, TWO_PI);  // Frown arc  
      // NB: The starting angle of the arc, in radians. Angles in Processing start at 3 o’clock and increase clockwise. stop: The ending angle of the arc, in radians.
    } else if (confidence <= 0.6) {
      // Neutral or slight smile
      arc(x, y + smileOffsetY + 20, smileWidth, smileHeight - 30, PI, TWO_PI);  // Neutral mouth
    } else if (confidence > 0.6 && confidence <= 0.8) {
      // Slight smile
      arc(x, y + smileOffsetY, smileWidth, smileHeight, 0, PI);  // Smiling arc
    } else if (confidence>0.8){
      // Full smile
      arc(x, y + smileOffsetY, smileWidth, smileHeight + 30, 0, PI);  // Full smile
    }
  }
  
  void updatePosition(){
    // Update the vertical position of the emoticon  
    y += ySpeed;
    
    // Update speed base on emotion
    if (emotion == 0.0) {
      if (ySpeed < 0){
        ySpeed = -2;
      }
      else{
        ySpeed = 2;
      }
    }
    else {
      if (ySpeed < 0){
        ySpeed = -4;
      }
      else{
        ySpeed = 4;
      }
    }
  
    // Reverse direction when hitting the top or bottom of the screen
    if (this.y > height - this.emoticonRadius || this.y < this.emoticonRadius) {
      ySpeed *= -1;
    }
  
    // Manage pulsating glow for sustained high confidence
    pulseRadius += pulseDirection * pulseSpeed;
    if (pulseRadius > 50 || pulseRadius < 0) {
      pulseDirection *= -1;  // Reverse the pulsating effect
    }
  }
}
