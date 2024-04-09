# open a udp socket and listen for incoming data
# when data is received, parse the data 

import socket
import sys
from BCI.utils.loader import unicorn_fs

# Global Variables
UDP_IP = "127.0.0.1" # Localhost
UDP_PORT = 1001
BUFFER_SIZE = 4 * unicorn_fs
WINDOW_SIZE = 1 * unicorn_fs

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

    
if __name__ == "__main__":

    print("Server is running...")
    exit_server = False
    buffer = []

    while not exit_server:

        print("Waiting for data...")

        data, addr = sock.recvfrom(10000)
        msg = data.decode()
        print("Received message: ", msg)
        print("From: ", addr)

        data_from_unicorn = msg
        # create a buffer to store the data
        buffer.append(data_from_unicorn)
        if len(buffer) == BUFFER_SIZE:
            # make a prediction
            print("Make a prediction")
            # remove first n elements from the buffer
            buffer = buffer[WINDOW_SIZE:]

        # # exit on keyboard interrupt (Ctrl + C)
        # try:
        #     pass
        # except KeyboardInterrupt:
        #     print("Exiting server...")
        #     exit_server = True 
        #     sock.close()




