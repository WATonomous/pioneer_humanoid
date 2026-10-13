#ifndef IMU_TASK_H
#define IMU_TASK_H

#include "FreeRTOS.h"
#include "queue.h"
#include "task.h"
#include "stdbool.h"
#include "sh2.h"
#include "sh2_SensorValue.h"
#include "drivers/bno085/bno085_hal.h"

typedef struct {
    float qw;
    float qx;
    float qy;
    float qz;
} bno085_quaterinion_data;

typedef struct {
    float gyro_x;
    float gyro_y;
    float gyro_z;
} bno085_gyro_data;

typedef struct {
    float gravity_x;
    float gravity_y;
    float gravity_z;
} bno085_gravity_data;

typedef struct {
    bno085_quaterinion_data quat;
    bno085_gyro_data gyro;
    bno085_gravity_data grav;
} bno085_data;

typedef struct {
    TickType_t quat_updated;
    TickType_t gyro_updated;
    TickType_t grav_updated;
    bno085_data data;
} bno085_sample;

extern QueueHandle_t bno085SampleQueue;
extern TaskHandle_t bno085TaskHandle;

void BNO085_Task(void *pvParameters);

#endif // IMU_TASK_H
