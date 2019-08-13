import socket

# server configuration
HOST = '127.0.0.1'
PORT = 8080
VERSION = socket.AF_INET
PROTOCOL = socket.SOCK_DGRAM

# start server
server = socket.socket(VERSION, PROTOCOL)

# bind server and wait for connections
server.bind((HOST, PORT))


# accept incoming packets
while True:
    print("Waiting to recv")
    data, addr = server.recvfrom(1024)
    print("Waiting...")
    print(data)

print("Complete")
