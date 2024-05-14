import socket
import logging



class Server:

    def __init__(self, ip, port, buffer_size, window_size):
        self.ip = ip
        self.port = port
        self.buffer_size = buffer_size
        self.window_size = window_size
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.ip, self.port))
        self.buffer = []
        self.exit_server = False
        logging.info(f"Server {self.ip}:{self.port}: started")

    def start(self):

        while not self.exit_server:
            data, addr = self.sock.recvfrom(10000)
            if data != None:
                msg = data.decode()
                logging.info(f"Server {self.ip}:{self.port}: received message <<{msg}>> from {addr}")

                data_from_unicorn = msg
                # create a buffer to store the data
                self.buffer.append(data_from_unicorn)
                if len(self.buffer) == self.buffer_size:
                    # make a prediction
                    logging.info("Make a prediction")
                    # remove first n elements from the buffer
                    self.buffer = self.buffer[self.window_size:]

        self.sock.close()
        logging.info(f"Server {self.ip}:{self.port}: closed")

    def close(self):
        self.exit_server = True





class Client:
    
    def __init__(self, name):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.name = name
        logging.info(f"Client {name}: started")

    def send(self, msg, ip, port):
        self.sock.sendto(msg.encode(), (ip, port))
        logging.info(f"Client {self.name}: sent message <<{msg}>> to {ip}:{port}")

    def close(self):
        self.sock.close()
        logging.info(f"Client {self.name}: closed")