import controlP5.*;

class CustomSlider{
 
  Textlabel valueLabel;
  float labelX, labelY, labelHeight, labelWidth;
  ControlP5 cp5;
  Slider slider;
  int x, y;
  String name, caption;
  float max, min, startingValue;
  

  CustomSlider(ControlP5 cp5, String name, String caption, int x, int y, float min, float max, float startingValue){
    
    this.cp5 = cp5;
    this.name = name;
    this.caption = caption;
    this.x = x;
    this.y =y;
    this.max = max;
    this.min = min;
    this.startingValue = startingValue;

    slider = cp5.addSlider(name)
                   .setPosition(x, y)
                   .setSize(30, 200)
                   .setRange(min, max)  // Linear range for the slider
                   .setValue(startingValue); // Set label color to transparent
                 
    slider.getCaptionLabel() 
          .setColor(color(0))
          .setFont(createFont("Arial", 16))  // Set the font and size of the label
          .align(ControlP5.CENTER, ControlP5.TOP_OUTSIDE);   
                             
    // Add a label for the slider value
    valueLabel = cp5.addTextlabel("sliderValueLabel")
                    .setColor(color(0))
                    .setFont(createFont("Arial", 14)); 
                    
    labelHeight = valueLabel.getHeight();
    labelWidth = valueLabel.getWidth();
    
    slider.getValueLabel().setVisible(false); // Hide the default value label
  }
  
  void setValue(float value){
    slider.setValue(value);
  }
   
  float getValue(){
    return slider.getValue();
  }
  
  void setVisible(boolean value){
    slider.setVisible(value);
    slider.getValueLabel().setVisible(false);
    valueLabel.setVisible(false);
  }
  
  void updateValueLabel(float value){
    labelY = map(slider.getValue(), slider.getMin(), slider.getMax(), slider.getPosition()[1] + slider.getHeight(), slider.getPosition()[1]);
    // Update the label's position to follow the slider handle
    valueLabel.setPosition(slider.getPosition()[0] + slider.getWidth(), labelY - 6);  // Adjust the x-position as needed
    // Display the logarithmic value
    valueLabel.setText(nf(value, 1, 2));
    labelX = valueLabel.getPosition()[0];
    labelY = valueLabel.getPosition()[1];
  }
  
}
