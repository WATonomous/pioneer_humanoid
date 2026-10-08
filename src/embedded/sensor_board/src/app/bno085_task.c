#include "app/bno085_task.h"
#include "app/can_task.h"

QueueHandle_t bno085SampleQueue = NULL;
TaskHandle_t bno085TaskHandle = NULL;
static bno085_sample imuData = {0};

static void sensorCallback(void *cookie, sh2_SensorEvent_t *event)
{
    sh2_SensorValue_t value;

    if (sh2_decodeSensorEvent(&value, event) != SH2_OK)
    {
        return;
    }

    TickType_t now = xTaskGetTickCount();

    switch (value.sensorId)
    {
        case SH2_GAME_ROTATION_VECTOR:
            imuData.data.quat.qw = value.un.gameRotationVector.real;
            imuData.data.quat.qx = value.un.gameRotationVector.i;
            imuData.data.quat.qy = value.un.gameRotationVector.j;
            imuData.data.quat.qz = value.un.gameRotationVector.k;

            imuData.quat_updated = now;
            break;
        case SH2_GYROSCOPE_CALIBRATED:
            imuData.data.gyro.gyro_x = value.un.gyroscope.x;
            imuData.data.gyro.gyro_y = value.un.gyroscope.y;
            imuData.data.gyro.gyro_z = value.un.gyroscope.z;

            imuData.gyro_updated = now;
            break;
        case SH2_GRAVITY:
            imuData.data.grav.gravity_x = value.un.gravity.x;
            imuData.data.grav.gravity_y = value.un.gravity.y;
            imuData.data.grav.gravity_z = value.un.gravity.z;

            imuData.grav_updated = now;
            break;
        default:
            break;
    }

    xQueueOverwrite(
        bno085SampleQueue,
        &imuData
    );
}

void BNO085_Task(void *pvParameters)
{
    sh2_Hal_t hal = get_bno085_hal();
    if (sh2_open(&hal, NULL, NULL) != SH2_OK)
    {
        vTaskSuspend(NULL);
    }
    sh2_setSensorCallback(sensorCallback, NULL);

    sh2_SensorConfig_t config = {0};
    config.reportInterval_us = 4000;

    if (sh2_setSensorConfig(SH2_GAME_ROTATION_VECTOR, &config) != SH2_OK)
    {
        vTaskSuspend(NULL);
    }

    if (sh2_setSensorConfig(SH2_GYROSCOPE_CALIBRATED, &config) != SH2_OK)
    {
        vTaskSuspend(NULL);
    }

    if (sh2_setSensorConfig(SH2_GRAVITY, &config) != SH2_OK)
    {
        vTaskSuspend(NULL);
    }

    for (;;)
    {
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        sh2_service();
    }
}
