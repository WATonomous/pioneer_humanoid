#include "app/can_task.h"
#include "app/bno085_task.h"
#include "stm32/fdcan.h"

#include "FreeRTOS.h"
#include "queue.h"
#include "task.h"

#include <stdbool.h>
#include <stdint.h>

#define IMU_CAN_PAYLOAD_SIZE        20U
#define IMU_CAN_EXTENDED_ID         0x2B00U
#define IMU_STALE_TIMEOUT_MS        20U

#define IMU_QUAT_SCALE              16384.0f
#define IMU_GYRO_SCALE              512.0f
#define IMU_GRAVITY_SCALE           256.0f

typedef struct
{
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

static bool sample_is_fresh(const bno085_sample *sample)
{
    TickType_t now = xTaskGetTickCount();
    TickType_t timeout = pdMS_TO_TICKS(IMU_STALE_TIMEOUT_MS);

    if ((sample->quat_updated == 0) ||
        (sample->gyro_updated == 0) ||
        (sample->grav_updated == 0))
    {
        return false;
    }

    if ((now - sample->quat_updated) > timeout)
    {
        return false;
    }

    if ((now - sample->gyro_updated) > timeout)
    {
        return false;
    }

    if ((now - sample->grav_updated) > timeout)
    {
        return false;
    }

    return true;
}

static ImuCanData sample_to_can_data(const bno085_sample *sample)
{
    ImuCanData data;

    data.qx = (int16_t)(sample->data.quat.qx * IMU_QUAT_SCALE);
    data.qy = (int16_t)(sample->data.quat.qy * IMU_QUAT_SCALE);
    data.qz = (int16_t)(sample->data.quat.qz * IMU_QUAT_SCALE);
    data.qw = (int16_t)(sample->data.quat.qw * IMU_QUAT_SCALE);

    data.angular_velocity_x = (int16_t)(sample->data.gyro.gyro_x * IMU_GYRO_SCALE);
    data.angular_velocity_y = (int16_t)(sample->data.gyro.gyro_y * IMU_GYRO_SCALE);
    data.angular_velocity_z = (int16_t)(sample->data.gyro.gyro_z * IMU_GYRO_SCALE);

    data.gravity_x = (int16_t)(sample->data.grav.gravity_x * IMU_GRAVITY_SCALE);
    data.gravity_y = (int16_t)(sample->data.grav.gravity_y * IMU_GRAVITY_SCALE);
    data.gravity_z = (int16_t)(sample->data.grav.gravity_z * IMU_GRAVITY_SCALE);

    return data;
}

static void pack_int16_big_endian(uint8_t *destination, int16_t value)
{
    uint16_t raw = (uint16_t)value;

    destination[0] = (uint8_t)(raw >> 8);
    destination[1] = (uint8_t)(raw & 0xFFU);
}

static void pack_imu_data(uint8_t payload[IMU_CAN_PAYLOAD_SIZE], const ImuCanData *imu)
{
    pack_int16_big_endian(&payload[0], imu->qx);
    pack_int16_big_endian(&payload[2], imu->qy);
    pack_int16_big_endian(&payload[4], imu->qz);
    pack_int16_big_endian(&payload[6], imu->qw);

    pack_int16_big_endian(&payload[8], imu->angular_velocity_x);
    pack_int16_big_endian(&payload[10], imu->angular_velocity_y);
    pack_int16_big_endian(&payload[12], imu->angular_velocity_z);

    pack_int16_big_endian(&payload[14], imu->gravity_x);
    pack_int16_big_endian(&payload[16], imu->gravity_y);
    pack_int16_big_endian(&payload[18], imu->gravity_z);
}

static bool send_imu_data(const ImuCanData *imu)
{
    uint8_t payload[IMU_CAN_PAYLOAD_SIZE];

    FDCAN_TxHeaderTypeDef header = {
        .Identifier = IMU_CAN_EXTENDED_ID,
        .IdType = FDCAN_EXTENDED_ID,
        .TxFrameType = FDCAN_DATA_FRAME,
        .DataLength = FDCAN_DLC_BYTES_20,
        .ErrorStateIndicator = FDCAN_ESI_ACTIVE,
        .BitRateSwitch = FDCAN_BRS_ON,
        .FDFormat = FDCAN_FD_CAN,
        .TxEventFifoControl = FDCAN_NO_TX_EVENTS,
        .MessageMarker = 0
    };

    pack_imu_data(payload, imu);

    return HAL_FDCAN_AddMessageToTxFifoQ(&hfdcan1, &header, payload) == HAL_OK;
}

void CanTask(void *pvParameters)
{
    (void)pvParameters;

    bno085_sample sample;

    for (;;)
    {
        if (xQueueReceive(bno085SampleQueue, &sample, pdMS_TO_TICKS(IMU_STALE_TIMEOUT_MS)) != pdTRUE)
        {
            continue;
        }

        if (!sample_is_fresh(&sample))
        {
            continue;
        }

        ImuCanData canData = sample_to_can_data(&sample);
        send_imu_data(&canData);
    }
}