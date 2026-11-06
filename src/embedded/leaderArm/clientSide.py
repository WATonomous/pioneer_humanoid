# THIS IS AN EXAMPLE OF READING THE ESP32 PACKETS, THANKS TO CLAUDE!!

import socket
import struct

HOST = "0.0.0.0"  # Listens on all available network cards
PORT = 8080

# '<' forces little-endian (matches ESP32), 'I' = 32-bit uint, '6h' = six 16-bit shorts
PACKET_FORMAT = "<I6h" 
PACKET_SIZE = struct.calcsize(PACKET_FORMAT) # Struct size is exactly 16 bytes

def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        # Allow immediate port reuse after restarting the script
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((HOST, PORT))
        server_socket.listen(1)
        
        print(f"Listening for SO-100 ESP32 stream on port {PORT}...")
        conn, addr = server_socket.accept()
        
        with conn:
            print(f"Connected by ESP32 at {addr}")
            # Match ESP32 low-latency socket performance
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            
            buffer = bytearray()
            while True:
                data = conn.recv(1024)
                if not data:
                    print("ESP32 disconnected.")
                    break
                buffer.extend(data)
                
                # FIX: Process all fully arrived packets (use >= instead of ==)
                while len(buffer) >= PACKET_SIZE:
                    # FIX: Correctly slice the first 16 bytes out of the buffer
                    packet_bytes = buffer[:PACKET_SIZE]
                    del buffer[:PACKET_SIZE]
                    
                    # FIX: Added * to capture all 6 servo short values into a list
                    timestamp, *positions = struct.unpack(PACKET_FORMAT, packet_bytes)
                    
                    # Ready to pipe directly into your Python control loops or LeRobot logic!
                    print(f"[{timestamp}ms] Servo Positions: {positions}")

if __name__ == "__main__":
    main()