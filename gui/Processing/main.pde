void setup() {
  // Initialize the OSC server
  oscP5 = new OscP5(this, port);
  println("OSC server started on port " + port);
  
  size(500, 400);         // Increased width to accommodate the slider
  emoticonRadius = emoticonSize / 2; // Calculate the radius
  x = width / 2 - 50;     // Keep the emoticon horizontally centered (offset for the slider)
  y = height / 2;         // Start in the middle vertically
  ySpeed = 3;             // Speed of the vertical movement
  frameRate(50);          // Set frame rate to 50 Hertz

  // Initialize ControlP5 slider for confidence
  cp5 = new ControlP5(this);
  cp5.addSlider("confidence")         // Attach the confidence variable to the slider
     .setPosition(width - 80, 100)    // Position of the slider (right of the screen)
     .setSize(30, 200)                // Size of the slider
     .setRange(0, 1)                  // Set range from 0 to 1 (for confidence)
     .setValue(0.1)                   // Initial value for confidence
     .setLabel("Confidence");         // Label for the slider
}

void draw() {
  background(255);        // Clear the screen with white background

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

  // Draw the emoticon and update expressions
  drawEmoticon();

  // Update the vertical position of the emoticon
  y += ySpeed;

  // Reverse direction when hitting the top or bottom of the screen
  if (y > height - emoticonRadius || y < emoticonRadius) {
    ySpeed *= -1;
  }

  // Manage pulsating glow for sustained high confidence
  pulseRadius += pulseDirection * pulseSpeed;
  if (pulseRadius > 50 || pulseRadius < 0) {
    pulseDirection *= -1;  // Reverse the pulsating effect
  }
}
