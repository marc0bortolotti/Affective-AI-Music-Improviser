// Define global variables for the state of the app
String userName = "";
boolean isNameEntered = false;
boolean isParameterSelected = false;

class StartMenu{
  
  String[] midiInputPorts = {"Input 1", "Input 2", "Input 3"};
  String[] midiOutputPorts = {"Output 1", "Output 2", "Output 3"};
  String[] generativeModels = {"Model A", "Model B", "Model C"};
  String[] generationTypes = {"Type 1", "Type 2", "Type 3"};
  String[] eegDevices = {"Device 1", "Device 2", "Device 3"};
  Dropdown midiInputDropdown;
  Dropdown midiOutputDropdown;
  Dropdown generativeModelDropdown;
  Dropdown generationTypeDropdown;
  Dropdown eegDeviceDropdown;
  
  StartMenu(){
    println("MENU");
    midiInputDropdown = new Dropdown(0, midiInputPorts, "MIDI IN");
    midiOutputDropdown = new Dropdown(20, midiOutputPorts, "MIDI IN");
    generativeModelDropdown = new Dropdown(40, generativeModels, "MIDI IN");
    generationTypeDropdown = new Dropdown(60, generationTypes, "MIDI IN");
    eegDeviceDropdown = new Dropdown(80, eegDevices, "MIDI IN");
  }

  void displayFirstPage() {
    textSize(24);
    text("Enter Your Name", width / 2, height / 4);
    textSize(18);
    text("Press Enter to submit", width / 2, height / 1.5);
    
    fill(0);
    rect(width / 2 - 100, height / 2, 200, 30);  // Input box
    fill(255);
    text(userName, width / 2, height / 2 + 15);
  }
  
  void displaySecondPage() {
    textSize(24);
    text("Settings", width / 2, height / 4 - 50);
  
    textSize(16);
    text("MIDI Input Port:", width / 2, height / 4);
    midiInputDropdown.display();
  
    text("MIDI Output Port:", width / 2, height / 4 + 60);
    midiOutputDropdown.display();
  
    text("Generative Model:", width / 2, height / 4 + 120);
    generativeModelDropdown.display();
  
    text("Generation Type:", width / 2, height / 4 + 180);
    generationTypeDropdown.display();
  
    text("EEG Device:", width / 2, height / 4 + 240);
    eegDeviceDropdown.display();
  }
}




class Dropdown {
  float x, y, w, h, offset;
  String[] options;
  int selectedIndex = -1;
  boolean isExpanded = false;  // Whether the dropdown is expanded
  boolean isHovered = false;
  int arrowSize = 10;
  String label;  // Label for the dropdown

  // Constructor to initialize the dropdown with a label
  Dropdown(int offset, String[] options, String label) {
    this.label = label;
    this.options = options;
    this.h = 30;  // Height of each option/button
    this.offset = offset;
  }

  // Display the dropdown menu
  void display() {
    // Dynamically calculate the dropdown position and width based on window size
    this.w = width / 2;  // Set the width to half of the window width
    this.x = (width - w) / 2;  // Center horizontally
    this.y = offset + height / 2 - h / 2;  // Center vertically

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
    if (selectedIndex == -1) {
      text("Select an option", x + 10, y + h / 2);  // Default text
    } else {
      text(options[selectedIndex], x + 10, y + h / 2);  // Selected option
    }

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
      for (int i = 0; i < options.length; i++) {
        fill(255);
        rect(x, y + (i + 1) * h, w, h);
        fill(0);
        textSize(14);
        text(options[i], x + 10, y + (i + 1) * h + h / 2);
      }
    }
  }

  // Check if the mouse is pressed on the dropdown
  void mousePressed() {
    if (isHovered && mouseY < y + h) {
      // Toggle the dropdown when the main button is clicked
      isExpanded = !isExpanded;
    } else if (isExpanded) {
      // Select an option if the dropdown is expanded
      for (int i = 0; i < options.length; i++) {
        if (mouseY > y + (i + 1) * h && mouseY < y + (i + 2) * h) {
          selectedIndex = i;
          isExpanded = false;  // Close the dropdown after selection
          break;
        }
      }
    }
  }
}
