# open a udp socket and listen for incoming data
# when data is received, parse the data 

import socket
import sys
from BCI.utils.loader import unicorn_fs

# Global Variables
UDP_IP = "127.0.0.1" # Localhost
UDP_PORT = 1002

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

# Function to send data
def send_data(data):
    sock.sendto(data.encode(), (UDP_IP, 1001))
    print("Sent message: ", data)

if __name__ == "__main__":

    send_data("Hello, World!")






