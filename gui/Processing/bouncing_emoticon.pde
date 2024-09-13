import controlP5.*;

ControlP5 cp5;           // ControlP5 object for the slider
float confidence = 0.1;   // Confidence variable (0 to 1)
int emoticonSize = 200;   // Diameter of the emoticon
int emoticonRadius;       // Radius of the emoticon
int x, y;                // Position of the emoticon
int ySpeed;              // Speed of the emoticon in the y direction
int lowConfidenceTime = 0;
boolean sustainedHigh = false; // Tracks whether confidence has been high for 5 seconds
float pulseRadius = 0;    // Pulse size for glow effect
float pulseSpeed = 120 / 60; // Pulsating speed (120 beats per minute)
float pulseDirection = 1; // Pulse direction for glow
int glowDuration = 5000;  // Minimum duration for high confidence to activate glow (5 seconds)
int glowAlpha = 150;      // Glow transparency
int firstHighTime = 0;         // The last time high confidence was registered
int currentTime = 0;      // Tracks elapsed time
boolean start_count = true;


// Draws the emoticon based on the current confidence level
void drawEmoticon() {
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
  } else if (confidence < 0.6) {
    // Neutral or slight smile
    arc(x, y + smileOffsetY + 20, smileWidth, smileHeight - 30, PI, TWO_PI);  // Neutral mouth
  } else if (confidence > 0.6 && confidence < 0.8) {
    // Slight smile
    arc(x, y + smileOffsetY, smileWidth, smileHeight, 0, PI);  // Smiling arc
  } else {
    // Full smile
    arc(x, y + smileOffsetY, smileWidth, smileHeight + 30, 0, PI);  // Full smile
  }
}
