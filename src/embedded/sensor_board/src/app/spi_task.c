#include "app/spi_task.h"
#include "app/bno085_task.h"

#include <string.h>

#define SHTP_HDR_SIZE_BYTES 4U

TaskHandle_t spiTaskHandle = NULL;

typedef enum
{
    IDLE,
    TX,
    RX_HDR,
    RX_BDY
} spiState_E;

typedef struct
{
    spiState_E currentState;
    uint8_t rxBuf[SH2_HAL_MAX_TRANSFER_IN];
    uint8_t txBuf[SH2_HAL_MAX_TRANSFER_OUT];
    uint32_t rxTransferLen;
    uint32_t rxBufLen;
    uint32_t txBufLen;
} spiTaskData;

static spiTaskData taskData =
{
    .rxBuf = {0},
    .txBuf = {0},
    .rxTransferLen = 0,
    .rxBufLen = 0,
    .txBufLen = 0,
    .currentState = IDLE
};

static uint8_t txZeros[SH2_HAL_MAX_TRANSFER_IN] = {0};

static void spi_write(void)
{
    HAL_GPIO_WritePin(SPI1_CS_GPIO_Port, SPI1_CS_Pin, GPIO_PIN_RESET);

    HAL_SPI_TransmitReceive_IT(&hspi1, taskData.txBuf, taskData.rxBuf, taskData.txBufLen);

    HAL_GPIO_WritePin(BNO085_WAKE_GPIO_Port, BNO085_WAKE_Pin, GPIO_PIN_SET);
}

static void spi_read(bool cs)
{
    if (cs)
    {
        HAL_GPIO_WritePin(SPI1_CS_GPIO_Port, SPI1_CS_Pin, GPIO_PIN_RESET);

        HAL_SPI_TransmitReceive_IT(&hspi1, txZeros, taskData.rxBuf, SHTP_HDR_SIZE_BYTES);
    }
    else
    {
        HAL_SPI_TransmitReceive_IT(&hspi1, txZeros, &taskData.rxBuf[SHTP_HDR_SIZE_BYTES], taskData.rxTransferLen - SHTP_HDR_SIZE_BYTES);
    }
}

static spiState_E getDesiredState(uint32_t events)
{
    spiState_E ret = taskData.currentState;

    switch (taskData.currentState)
    {
        case IDLE:
        {
            bool interrupt = (HAL_GPIO_ReadPin(BNO085_INT_GPIO_Port, BNO085_INT_Pin) == GPIO_PIN_RESET);
            if (interrupt && (taskData.rxBufLen == 0))
            {
                if (taskData.txBufLen > 0)
                {
                    ret = TX;
                }
                else
                {
                    ret = RX_HDR;
                }
            }
            break;
        }
        case TX:
            if (events & SPI_EVENT_COMPLETE)
            {
                ret = IDLE;
            }
            break;
        case RX_HDR:
            if (events & SPI_EVENT_COMPLETE)
            {
                taskData.rxTransferLen = ((uint32_t)taskData.rxBuf[0]) | ((uint32_t)taskData.rxBuf[1] << 8);
                taskData.rxTransferLen &= 0x7FFF;

                if (taskData.rxTransferLen > SHTP_HDR_SIZE_BYTES &&
                    taskData.rxTransferLen <= sizeof(taskData.rxBuf))
                {
                    ret = RX_BDY;
                }
                else
                {
                    ret = IDLE;
                }
            }
            break;
        case RX_BDY:
            if (events & SPI_EVENT_COMPLETE)
            {
                ret = IDLE;
            }
            break;
    }
    return ret;
}

static void runAction(void)
{
    switch (taskData.currentState)
    {
        case IDLE:
            break;
        case TX:
            spi_write();
            break;
        case RX_HDR:
            spi_read(true);
            break;
        case RX_BDY:
            spi_read(false);
            break;
    }
}

static void exitAction(void)
{
    switch (taskData.currentState)
    {
        case IDLE:
            break;
        case TX:
            HAL_GPIO_WritePin(SPI1_CS_GPIO_Port, SPI1_CS_Pin, GPIO_PIN_SET);

            uint32_t recvLen = ((uint32_t)taskData.rxBuf[0]) | ((uint32_t)taskData.rxBuf[1] << 8);
            recvLen &= 0x7FFF;

            taskData.rxBufLen = (recvLen < taskData.txBufLen) ? recvLen : taskData.txBufLen;
            taskData.txBufLen = 0;

            if (taskData.rxBufLen > 0)
            {
                xTaskNotifyGive(bno085TaskHandle);
            }
            break;
        case RX_HDR:
            /* Keep CS asserted only when a bounded body transfer follows. */
            if (taskData.rxTransferLen <= SHTP_HDR_SIZE_BYTES ||
                taskData.rxTransferLen > sizeof(taskData.rxBuf))
            {
                HAL_GPIO_WritePin(SPI1_CS_GPIO_Port, SPI1_CS_Pin, GPIO_PIN_SET);
            }
            if (taskData.rxTransferLen == SHTP_HDR_SIZE_BYTES)
            {
                taskData.rxBufLen = taskData.rxTransferLen;
                xTaskNotifyGive(bno085TaskHandle);
            }
            break;
        case RX_BDY:
            HAL_GPIO_WritePin(SPI1_CS_GPIO_Port, SPI1_CS_Pin, GPIO_PIN_SET);
            taskData.rxBufLen = taskData.rxTransferLen;
            xTaskNotifyGive(bno085TaskHandle);
            break;
    }
}

void SpiTask(void *pvParameters)
{
    for (;;)
    {
        uint32_t events = 0;
        xTaskNotifyWait(0, UINT32_MAX, &events, portMAX_DELAY);

        if (events & SPI_EVENT_RX_FREE)
        {
            taskData.rxBufLen = 0;
        }

        spiState_E desiredState = getDesiredState(events);
        if (taskData.currentState != desiredState)
        {
            exitAction();
            taskData.currentState = desiredState;
            runAction();
        }
    }
}

const uint8_t *SpiTask_read(uint32_t *len)
{
    if (len == NULL)
    {
        return NULL;
    }

    *len = taskData.rxBufLen;
    if (*len == 0)
    {
        return NULL;
    }

    return taskData.rxBuf;
}

bool SpiTask_write(const uint8_t *data, uint32_t len)
{
    if (len == 0 || len > SH2_HAL_MAX_PAYLOAD_OUT || data == NULL)
    {
        return false;
    }

    if (taskData.txBufLen > 0)
    {
        return false;
    }

    memcpy(taskData.txBuf, data, len);
    taskData.txBufLen = len;

    return true;
}
