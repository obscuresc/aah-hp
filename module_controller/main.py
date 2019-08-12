import socket

# server configuration
HOST = '127.0.0.1'
PORT = 8080
VERSION = socket.AF_INET
PROTOCOL = socket.SOCK_DGRAM

# start server
with socket.socket(VERSION, PROTOCOL) as server:

    # bind server and wait for connections
    server.bind((HOST, PORT))
    s.listen()

    # accept incoming request
    conn, addr = s.accept()
    with conn:
        print('Connected by', addr)
        while True:
            data = conn.recv(1024)
            if not data:
                break
            conn.sendall(data)
