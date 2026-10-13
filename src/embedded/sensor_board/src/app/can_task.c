#include "app/can_task.h"
#include "app/bno085_task.h"
#include "stm32/fdcan.h"

#include "FreeRTOS.h"
#include "queue.h"
#include "task.h"

#include <stdbool.h>
#include <stdint.h>

#define IMU_QUATERNION_PAYLOAD_SIZE         8U
#define IMU_ANGULAR_VELOCITY_PAYLOAD_SIZE   6U
#define IMU_GRAVITY_PAYLOAD_SIZE            6U

#define IMU_QUATERNION_CAN_ID                0x2B00U
#define IMU_ANGULAR_VELOCITY_CAN_ID          0x2D00U
#define IMU_GRAVITY_CAN_ID                   0x2E00U
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

static void pack_quaternion(
    uint8_t payload[IMU_QUATERNION_PAYLOAD_SIZE],
    const ImuCanData *imu)
{
    pack_int16_big_endian(&payload[0], imu->qx);
    pack_int16_big_endian(&payload[2], imu->qy);
    pack_int16_big_endian(&payload[4], imu->qz);
    pack_int16_big_endian(&payload[6], imu->qw);
}

static void pack_angular_velocity(
    uint8_t payload[IMU_ANGULAR_VELOCITY_PAYLOAD_SIZE],
    const ImuCanData *imu)
{
    pack_int16_big_endian(&payload[0], imu->angular_velocity_x);
    pack_int16_big_endian(&payload[2], imu->angular_velocity_y);
    pack_int16_big_endian(&payload[4], imu->angular_velocity_z);
}

static void pack_gravity(
    uint8_t payload[IMU_GRAVITY_PAYLOAD_SIZE],
    const ImuCanData *imu)
{
    pack_int16_big_endian(&payload[0], imu->gravity_x);
    pack_int16_big_endian(&payload[2], imu->gravity_y);
    pack_int16_big_endian(&payload[4], imu->gravity_z);
}

static bool send_classic_frame(
    uint32_t identifier,
    uint32_t dataLength,
    uint8_t *payload)
{
    FDCAN_TxHeaderTypeDef header = {
        .Identifier = identifier,
        .IdType = FDCAN_EXTENDED_ID,
        .TxFrameType = FDCAN_DATA_FRAME,
        .DataLength = dataLength,
        .ErrorStateIndicator = FDCAN_ESI_ACTIVE,
        .BitRateSwitch = FDCAN_BRS_OFF,
        .FDFormat = FDCAN_CLASSIC_CAN,
        .TxEventFifoControl = FDCAN_NO_TX_EVENTS,
        .MessageMarker = 0
    };

    return HAL_FDCAN_AddMessageToTxFifoQ(
        &hfdcan1,
        &header,
        payload
    ) == HAL_OK;
}

static bool send_imu_data(const ImuCanData *imu)
{
    uint8_t quaternionPayload[IMU_QUATERNION_PAYLOAD_SIZE] = {0};
    uint8_t angularVelocityPayload[IMU_ANGULAR_VELOCITY_PAYLOAD_SIZE] = {0};
    uint8_t gravityPayload[IMU_GRAVITY_PAYLOAD_SIZE] = {0};

    pack_quaternion(quaternionPayload, imu);
    pack_angular_velocity(angularVelocityPayload, imu);
    pack_gravity(gravityPayload, imu);

    bool quaternionSent = send_classic_frame(
        IMU_QUATERNION_CAN_ID,
        FDCAN_DLC_BYTES_8,
        quaternionPayload
    );

    bool angularVelocitySent = send_classic_frame(
        IMU_ANGULAR_VELOCITY_CAN_ID,
        FDCAN_DLC_BYTES_6,
        angularVelocityPayload
    );

    bool gravitySent = send_classic_frame(
        IMU_GRAVITY_CAN_ID,
        FDCAN_DLC_BYTES_6,
        gravityPayload
    );

    return quaternionSent && angularVelocitySent && gravitySent;
}

void CanTask(void *pvParameters)
{
    (void)pvParameters;

    bno085_sample sample;
    TickType_t nextWakeTime = xTaskGetTickCount();

    for (;;)
    {
        if ((xQueuePeek(bno085SampleQueue, &sample, 0) == pdTRUE) &&
            sample_is_fresh(&sample))
        {
            ImuCanData canData = sample_to_can_data(&sample);
            send_imu_data(&canData);
        }

        vTaskDelayUntil(&nextWakeTime, pdMS_TO_TICKS(4U));
    }
}
