// Define global variables for the state of the app
String userName = "";
boolean isNameEntered = false;

ArrayList<String> eegDevices = new ArrayList<String>();
ArrayList<String> midiInputPorts = new ArrayList<String>();
ArrayList<String> midiOutputPorts = new ArrayList<String>();
ArrayList<String> generativeModels = new ArrayList<String>();
ArrayList<String> generationTypes = new ArrayList<String>();

ArrayList<Dropdown> dropdownList = new ArrayList<Dropdown>();

class StartMenu{
  
  StartMenu(){
    
    generationTypes.add("Melody");
    generationTypes.add("Rhythm");
    
    
    dropdownList.add(new Dropdown(0, midiInputPorts, "MIDI INSTRUMENT INPUT PORT", INSTRUMENT_MIDI_IN_PORT_MSG));
    dropdownList.add(new Dropdown(40, midiOutputPorts, "MIDI GENERATED OUTPUT PORT", GENERATED_MIDI_OUT_PORT_MSG));
    dropdownList.add(new Dropdown(80, midiOutputPorts, "MIDI INSTRUMENT OUTPUT PORT", INSTRUMENT_MIDI_OUT_PORT_MSG));
    dropdownList.add(new Dropdown(120, generativeModels, "MODELS", MODELS_MSG));
    dropdownList.add(new Dropdown(160, generationTypes, "GENERATION TYPE", GENERATION_TYPE_MSG));
    dropdownList.add(new Dropdown(200, eegDevices, "EEG DEVICE", EEG_DEVICES_MSG));
  }

  void display() {
    
    textSize(24);
    text("User Name", width / 2 - 50, height / 2 - 220);
    textSize(18);
    
    fill(0);
    rect(width / 2 - 100, height / 2 - 200, 200, 30);  // Input box
    fill(255);
    text(userName, width / 2 - 30, height / 2 - 200 + 15);
    
    for (int i = 1; i <= dropdownList.size(); i++) {
      dropdownList.get(dropdownList.size()-i).display();
    }
  }
}




class Dropdown {
  float x, y, w, h, offset;
  ArrayList<String>  options;
  int selectedIndex = 0;
  boolean isExpanded = false;  // Whether the dropdown is expanded
  boolean isHovered = false;
  int arrowSize = 10;
  String label, address;  // Label for the dropdown

  // Constructor to initialize the dropdown with a label
  Dropdown(int offset, ArrayList<String> options, String label, String address) {
    this.label = label;
    this.options = options;
    this.h = 30;  // Height of each option/button
    this.offset = offset;
    this.address = address;
  }

  // Display the dropdown menu
  void display() {
    // Dynamically calculate the dropdown position and width based on window size
    this.w = width / 2;  // Set the width to half of the window width
    this.x = (width - w) / 2 + 100;  // Center horizontally
    this.y = offset + height / 2 - h;  // Center vertically

    isHovered = (mouseX > x && mouseX < x + w && mouseY > y && mouseY < y + h);
    
    // Display the label to the left of the dropdown
    fill(0);
    textSize(16);
    textAlign(RIGHT, CENTER);
    text(label, x - 10, y + h / 2);

    // Display the main button with the selected option
    fill(255);
    rect(x, y, w, h, 5);  // Main button area
    fill(0);
    textAlign(LEFT, CENTER);
    text(options.get(selectedIndex), x + 10, y + h / 2);  // Selected option

    // Draw the arrow
    if (isExpanded) {
      fill(0);
      triangle(x + w - arrowSize, y + h / 2 - 5, x + w - arrowSize - 5, y + h / 2 + 5, x + w - arrowSize + 5, y + h / 2 + 5);
    } else {
      fill(0);
      triangle(x + w - arrowSize, y + h / 2 + 5, x + w - arrowSize - 5, y + h / 2 - 5, x + w - arrowSize + 5, y + h / 2 - 5);
    }

    // Show the list of options if expanded
    if (isExpanded) {
      for (int i = 0; i < options.size(); i++) {
        fill(255);
        rect(x, y + (i + 1) * h, w, h);
        fill(0);
        textSize(14);
        text(options.get(i), x + 10, y + (i + 1) * h + h / 2);
      }
    }
  }
  
  String getSelectedItem(){
    return options.get(selectedIndex);
  }

  // Check if the mouse is pressed on the dropdown
  void mousePressed() {
    if (isHovered && mouseY < y + h) {
      // Toggle the dropdown when the main button is clicked
      isExpanded = !isExpanded;
    } else if (isExpanded) {
      // Select an option if the dropdown is expanded
      for (int i = 0; i < options.size(); i++) {
        if (mouseY > y + (i + 1) * h && mouseY < y + (i + 2) * h) {
          selectedIndex = i;
          isExpanded = false;  // Close the dropdown after selection
          break;
        }
      }
    }
  }
}
