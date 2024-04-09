# open a udp socket and listen for incoming data
# when data is received, parse the data 

import socket
import sys
from BCI.utils.loader import unicorn_fs

# Global Variables
UDP_IP = "127.0.0.1" # Localhost
UDP_PORT = 5005
BUFFER_SIZE = 4 * unicorn_fs
WINDOW_SIZE = 1 * unicorn_fs

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

# Function to receive data
def receive_data():
    while True:
        data, addr = sock.recvfrom(1024)
        print("Received message: ", data.decode())
        print("From: ", addr)
        return data.decode()
    
if __name__ == "__main__":
    print("Server is running...")
    data_from_unicorn = receive_data()

    # create a buffer to store the data
    buffer = []
    buffer.append(data_from_unicorn)
    if len(buffer) == BUFFER_SIZE:
        # make a prediction
        print("Make a prediction")
        # remove first n elements from the buffer
        buffer = buffer[WINDOW_SIZE:]

    # Close the socket
    sock.close()
    sys.exit()





