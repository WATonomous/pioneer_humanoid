#include "drivers/bno085/bno085_hal.h"

#include <string.h>

static int bno085_hal_open(sh2_Hal_t *self);
static void bno085_hal_close(sh2_Hal_t * self);
static int bno085_hal_read(sh2_Hal_t *self, uint8_t *pBuffer, unsigned len, uint32_t *t_us);
static int bno085_hal_write(sh2_Hal_t *self, uint8_t *pBuffer, unsigned len);
static uint32_t get_time_us(sh2_Hal_t *self);

static sh2_Hal_t bno085_hal =
{
    .open = bno085_hal_open,
    .close = bno085_hal_close,
    .read = bno085_hal_read,
    .write = bno085_hal_write,
    .getTimeUs = get_time_us
};

#define RESET_DELAY_MS 10
#define BOOT_TIMEOUT_MS 2000

static bool isOpen = false;

static int bno085_hal_open(sh2_Hal_t *self)
{
    if (isOpen)
    {
        return SH2_ERR;
    }

    HAL_GPIO_WritePin(SPI1_RST_GPIO_Port, SPI1_RST_Pin, GPIO_PIN_RESET);
    HAL_GPIO_WritePin(SPI1_CS_GPIO_Port, SPI1_CS_Pin, GPIO_PIN_SET);

    HAL_Delay(RESET_DELAY_MS);

    HAL_GPIO_WritePin(SPI1_RST_GPIO_Port, SPI1_RST_Pin, GPIO_PIN_SET);

    uint32_t startTime = HAL_GetTick();

    while (HAL_GPIO_ReadPin(BNO085_INT_GPIO_Port, BNO085_INT_Pin) == GPIO_PIN_SET)
    {
        if ((HAL_GetTick() - startTime) >= BOOT_TIMEOUT_MS)
        {
            return SH2_ERR_TIMEOUT;
        }
    }

    isOpen = true;
    return SH2_OK;
}

static void bno085_hal_close(sh2_Hal_t * self)
{
    HAL_GPIO_WritePin(SPI1_RST_GPIO_Port, SPI1_RST_Pin, GPIO_PIN_RESET);
    HAL_GPIO_WritePin(SPI1_CS_GPIO_Port, SPI1_CS_Pin, GPIO_PIN_SET);

    isOpen = false;
}

static int bno085_hal_read(sh2_Hal_t *self, uint8_t *pBuffer, unsigned len, uint32_t *t_us)
{
    uint32_t ret = 0;
    uint32_t rxLen = 0;
    const uint8_t *rxBuf = SpiTask_read(&rxLen);

    if (rxLen > 0)
    {
        if (len >= rxLen)
        {
            memcpy(pBuffer, rxBuf, rxLen);
            ret = rxLen;
            *t_us = get_time_us(self);
        }
        else
        {
            ret = SH2_ERR_BAD_PARAM;
        }

        xTaskNotify(spiTaskHandle, SPI_EVENT_RX_FREE, eSetBits);
    }

    return ret;
}

static int bno085_hal_write(sh2_Hal_t *self, uint8_t *pBuffer, unsigned len)
{
    if ((self == 0) || (len > SH2_HAL_MAX_TRANSFER_OUT) ||
        ((len > 0) && (pBuffer == 0)))
    {
        return SH2_ERR_BAD_PARAM;
    }

    if (!SpiTask_write(pBuffer, len))
    {
        return 0;
    }

    HAL_GPIO_WritePin(BNO085_WAKE_GPIO_Port, BNO085_WAKE_Pin, GPIO_PIN_RESET);

    return len;
}

static uint32_t get_time_us(sh2_Hal_t *self)
{
    uint32_t time_ms = HAL_GetTick();

    return time_ms * 1000;
}

sh2_Hal_t get_bno085_hal (void)
{
    return bno085_hal;
}
