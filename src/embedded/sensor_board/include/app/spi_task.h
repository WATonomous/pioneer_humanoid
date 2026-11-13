#ifndef SPI_TASK_H
#define SPI_TASK_H

#include "FreeRTOS.h"
#include "task.h"
#include "sh2_hal.h"
#include "stdbool.h"
#include "stm32/spi.h"
#include "stm32/gpio.h"

#define SPI_EVENT_INT       (1U << 0)
#define SPI_EVENT_COMPLETE  (1U << 1)
#define SPI_EVENT_RX_FREE   (1U << 2)

extern TaskHandle_t spiTaskHandle;

void SpiTask(void *pvParameters);

const uint8_t *SpiTask_read(uint32_t *len);
bool SpiTask_write(const uint8_t *data, uint32_t len);

#endif // SPI_TASK_H
