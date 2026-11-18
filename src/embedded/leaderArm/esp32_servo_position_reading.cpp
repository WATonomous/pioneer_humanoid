#include <Arduino.h>
#include <driver/uart.h>
#include <WiFi.h>
#include <lwip/sockets.h>

const char* ssid     = "esp32_network";
const char* password = "password";
const char* host     = "192.168.137.1"; // Replace with your computer's IP address
const uint16_t port  = 8080;

// Non-padded data struct for packets sent to computer from arm
struct __attribute__( (__packed__) ) armTelem {
  uint32_t time;
  int16_t positions[6];
};

int client_fd = -1; // default

/**
* PACKET_STRUCTURE:
* 0xFF, 
* 0xFF, 
* Servo ID, 
* Length, 
* Instruction, 
* Parameter1 (Register to look at), 
* Parameter2 (Length of data to read), 
* Checksum Byte 
*/

/**
* CHECKSUM_FORMULA:
* CheckSUM = ~(ID + Length + Parameter1 + Parameter2) & 0xF]
*/

// SERVO_INSTRUCTIONS:
const uint8_t STS_HEADER      = 0xFF;
const uint8_t STS_PING        = 0x01;
const uint8_t STS_READ        = 0x02;
const uint8_t STS_WRITE       = 0x03;
const uint8_t STS_REGWRITE    = 0x04;
const uint8_t STS_ACTION      = 0x05;
const uint8_t STS_RESET       = 0x06;
const uint8_t STS_SYNCWRITE   = 0x08;
const uint8_t STS_SYNCREAD    = 0x82;

// SERVO_REGISTERS:
const uint8_t INSTR_POSITION  = 0x38;
const uint8_t INSTR_SPEED     = 0x3A;
const uint8_t INSTR_LOAD      = 0x3C;
const uint8_t INSTR_VOLTAGE   = 0x3E;
const uint8_t INSTR_TEMP      = 0x3F;
const uint8_t INSTR_CURRENT   = 0x41;

// SERVO IDs
const uint8_t ID1             = 1;
const uint8_t ID2             = 2;
const uint8_t ID3             = 3;
const uint8_t ID4             = 4;
const uint8_t ID5             = 5;
const uint8_t ID6             = 6;

const uint8_t NUM_SERVOS      = 6;

// ESP CONFIGURATION
const uint8_t CUTOFF_TIME_MILLIS  = 2;
const uint8_t packetLengthRead    = 8;
const uint8_t packetLengthPing    = 6;
const int SERVO_TX_PIN = 17;
const int SERVO_RX_PIN = 16;
HardwareSerial servoSerial(1);

uint8_t txPacket[packetLengthRead] = {0};
uint8_t rxPacket[packetLengthRead] = {0};

// ANGLE CONVERSIONS
float raw_to_degrees(uint16_t raw) {
  return raw * (360.0f / 4096.0f);
}

float raw_to_centered_degrees(uint16_t raw) {
  return (raw - 2048.0f) * (360.0f / 4096.0f);
}

float raw_to_radians(uint16_t raw) {
  return (raw - 2048.0f) * (2.0f * PI / 4096.0f);
}

uint8_t calculate_checksum(uint8_t packet[], int packetLength){
  unsigned int checkSum = 0;
  for (int i = 2; i < packetLength - 1; i++){
    checkSum += packet[i];
  }
  return (byte)(~checkSum);
}

bool connectToHost() {
  client_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (client_fd < 0) return false;

  int flag = 1;
  setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, (char*)&flag, sizeof(int));

  struct sockaddr_in server_addr;
  server_addr.sin_family = AF_INET;
  server_addr.sin_port = htons(port);
  inet_pton(AF_INET, host, &server_addr.sin_addr);

  if (connect(client_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
    close(client_fd);
    client_fd = -1;
    return false;
  }
  Serial.println("Connected");
  return true;
}


void build_reading_packet(uint8_t* packet, uint8_t header1, uint8_t header2, uint8_t ID, uint8_t param1, uint8_t param2) {
  packet[0] = header1;
  packet[1] = header2;
  packet[2] = ID;
  packet[3] = 4;
  packet[4] = STS_READ;
  packet[5] = param1;
  packet[6] = param2;
  packet[7] = 0;
  packet[7] = calculate_checksum( packet, packetLengthRead );
}

void servo_position( uint8_t* packet, uint8_t servoID ) {
  build_reading_packet( packet, STS_HEADER, STS_HEADER, servoID, INSTR_POSITION, 2 );
}

bool read_packet_decode(uint8_t *rx_packet, int timeout){ 
  uint8_t buf[8];
  
  servoSerial.setTimeout(timeout);
  int length = servoSerial.readBytes(buf, 8);

  if (length != 8          ||
      buf[0] != STS_HEADER || 
      buf[1] != STS_HEADER ||
      calculate_checksum(buf, packetLengthRead) != buf[7]
      ) {
    return false;
  }

  for (int i = 0; i < packetLengthRead; ++i){
    rx_packet[i] = buf[i];
  }

  return true;
}

uint16_t position_from_rx(uint8_t *rx_packet){
  uint8_t lowBit = rx_packet[5];
  uint8_t highBit = rx_packet[6];
  return (uint16_t)lowBit | ((uint16_t)highBit << 8);
}

unsigned long startTime = 0;
const unsigned long timeoutDuration = 30000;
bool waiting = false;
bool connected = false;

void setup() {
  servoSerial.begin(1000000, SERIAL_8N1, SERVO_RX_PIN, SERVO_TX_PIN);
  Serial.begin(115200);
  delay(200);

  WiFi.begin( ssid, password );

  if( !waiting ) {
    waiting = true;
    while( waiting ) {
        if( WiFi.status() != WL_CONNECTED ) {
          delay( 500 );
          continue;
        }

        Serial.println( "Connected to network :)" );
        waiting = false;
        connected = true;
    }

    if( !connected ) Serial.println( "Failed to connect to network :(" );
  }

  WiFi.setSleep(false);
  
}

armTelem telemetry;

void loop() {

  if (client_fd < 0) {
    Serial.println("Attempting connection...");
    if (!connectToHost()) {
      delay(2000);
      return;
    }
  }

  memset(&telemetry, 0, sizeof(armTelem));
  telemetry.time = millis();

  uint8_t idArray[NUM_SERVOS] = {ID1, ID2, ID3, ID4, ID5, ID6};

  for( int i = 0; i < NUM_SERVOS; ++i ) {

    while (servoSerial.available()) { servoSerial.read(); }

    servo_position(txPacket, idArray[i]);
    servoSerial.write(txPacket, packetLengthRead);
    servoSerial.flush();
    
    if (read_packet_decode( rxPacket, CUTOFF_TIME_MILLIS )) {
      uint16_t rawPos = position_from_rx( rxPacket );
      telemetry.positions[i] = rawPos;
      
      float absDeg = raw_to_degrees(rawPos);
      float relDeg = raw_to_centered_degrees(rawPos);
      float rad    = raw_to_radians(rawPos);

      Serial.printf("SERVO %d | Raw: %4d | Abs: %6.1f deg | Rel: %6.1f deg | Rad: %5.2f rad\n", 
                    idArray[i], rawPos, absDeg, relDeg, rad);
    } else {
      Serial.printf("SERVO %d: READ ERROR\n", idArray[i]);
      telemetry.positions[i] = -1;
    }
  }

  int bytesSent = send(client_fd, (char*)&telemetry, sizeof(armTelem), 0);
  
  if (bytesSent < 0) {
    Serial.println("Connection lost.");
    close(client_fd);
    client_fd = -1;
  }

  delay(20); //500
}