#ifndef CAN_TASK
#define CAN_TASK

#include "FreeRTOS.h"
#include "stm32/fdcan.h"
#include "task.h"
#include <stdbool.h>
#include <stdint.h>


void CanTask(void* pvParameters);

#endif // CAN_TASK