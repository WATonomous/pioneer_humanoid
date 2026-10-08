#ifndef BNO085_HAL
#define BNO085_HAL

#include "sh2_hal.h"
#include "sh2_err.h"
#include "stdbool.h"
#include "stm32/gpio.h"
#include "stm32/spi.h"
#include "FreeRTOS.h"
#include "task.h"

sh2_Hal_t get_bno085_hal (void);

#endif // BNO085_HAL
