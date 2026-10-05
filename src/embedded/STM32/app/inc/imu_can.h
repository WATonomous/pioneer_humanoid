#ifndef IMU_CAN_H
#define IMU_CAN_H

#include <stdbool.h>
#include <stdint.h>

#define IMU_CAN_PAYLOAD_SIZE 20U
#define IMU_CAN_EXTENDED_ID 0x2B00U
typedef struct {
  int16_t qx;
  int16_t qy;
  int16_t qz;
  int16_t qw;

  int16_t angular_velocity_x;
  int16_t angular_velocity_y;
  int16_t angular_velocity_z;

  int16_t gravity_x;
  int16_t gravity_y;
  int16_t gravity_z;
} ImuCanData;

void imu_can_pack(uint8_t payload[IMU_CAN_PAYLOAD_SIZE], const ImuCanData* imu);
bool imu_can_init(void);
bool imu_can_send(const ImuCanData* imu);

#endif
