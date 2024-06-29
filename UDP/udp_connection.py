import socket
import logging
import time


class Server_UDP:

    def __init__(self, ip, port, buffer_size = 100, window_size = 1024, parse_message = False):
        self.ip = ip
        self.port = port
        self.buffer_size = buffer_size
        self.window_size = window_size
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.buffer = []
        self.parse_message = parse_message
        self.exit = False

    def run(self):
        self.sock.bind((self.ip, self.port))
        logging.info(f"UDP Server {self.ip}:{self.port}: started")


    def get_message(self):
        data, addr = self.sock.recvfrom(self.window_size)
        if data != None:
            msg = data.decode()
            self.update_buffer(msg)
            if self.parse_message:
                logging.info(f"UDP Server {self.ip}:{self.port}: received message <<{msg}>> from {addr}")
            return msg, addr
        return None
    
    def update_buffer(self, msg):
        self.buffer.append(msg)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def get_buffer(self):
        return self.buffer

    def close(self):
        self.exit = True
        self.sock.close()
        logging.info(f"UDP Server {self.ip}:{self.port}: closed")





class Client_UDP:
    
    def __init__(self, name, parse_message = False):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.name = name
        self.parse_message = parse_message
        logging.info(f"UDP Client {name}: started")

    def send(self, msg, ip, port):
        self.sock.sendto(msg.encode(), (ip, port))
        if self.parse_message:
            logging.info(f"UDP Client {self.name}: sent message <<{msg}>> to {ip}:{port}")

    def close(self):
        self.sock.close()
        logging.info(f"UDP Client {self.name}: closed")