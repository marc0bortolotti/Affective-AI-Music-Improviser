# AI-Affective Music Improviser
AI-Affective Music Improviser is a deep neural network-based system that establishes a direct connection with the minds of musichans to create music in real-time. The Imporviser acquires data from both the musician's instrument and a BCI device. 
The instrument sound must be in MIDI format while the BCI measures the EEG signal coming from the user.
Data are processed in parallel, in real-time and fed into a [TCN] (https://github.com/marc0bortolotti/Affective-AI-Music-Improviser/blob/main/generative_model/model.py) architecture that produce the affective music.
Depending on the application, the generated music pattern could be a melody or rhytmic drum pattern.



[loopMIDI](https://www.tobias-erichsen.de/software/loopmidi.html)

This project will follow the following procedure.
1. Revisit [setting up arduino IDE](https://github.com/billiyz/nano-33-ble-sense) without going through the programming part of sensor data collection.
2. Look into accelerometer and gyroscope operation
3. Write a sketch to collect accelerometer and gyroscope data
4. Upload the data to the google colaboratory platform
5. Train a neural network
6. Convert the trained model into tensorflow lite model
7. Encode the model in arduino header file

## Application
* Mobile Smart Phones
* Drones
* Aeroplanes
* Mobile IoT Devices
* Anti-theft/Asset Tracking/Security Devices

## Operation

![photo 1](images/3d-accelerometer.png)



 
















