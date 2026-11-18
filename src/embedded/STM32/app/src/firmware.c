
#include "../inc/common_defines.h"
#include "../inc/imu_can.h"
#include "../inc/system.h"
#include "FreeRTOS.h"
#include "stm32g4xx.h"
#include "task.h"

#define BOOTLOADER_SIZE (0x08008000U)

// set the vector table offset of app firmware image
static void vector_setup(void) {
  SCB->VTOR = BOOTLOADER_SIZE;
}

// thread for blinking led
void blink_led(void* pvParams) {
  while (1) {
    HAL_GPIO_TogglePin(GPIOA, GPIO_PIN_5);
    // vTaskDelay(pdMS_TO_TICKS(500));
    HAL_Delay(500);
  }
}
static volatile uint32_t imu_can_send_failures = 0;

static void imu_can_publish_task(void* pvParams) {
  (void)pvParams;

  /*
   * Temporary data used until the BNO085 reader is connected.
   *
   * Quaternion: identity rotation, W = 1.0 in Q14.
   * Gravity: approximately -9.81 m/s^2 in Q8.
   */
  ImuCanData imu = {
      .qx = 0,
      .qy = 0,
      .qz = 0,
      .qw = 16384,

      .angular_velocity_x = 0,
      .angular_velocity_y = 0,
      .angular_velocity_z = 0,

      .gravity_x = 0,
      .gravity_y = 0,
      .gravity_z = -2511,
  };

  TickType_t next_wake_time = xTaskGetTickCount();

  while (1) {
    if (!imu_can_send(&imu)) {
      imu_can_send_failures++;
    }

    vTaskDelayUntil(&next_wake_time, pdMS_TO_TICKS(4));
  }
}

int main() {
  vector_setup();
  HAL_Init();
  system_setup();
  if (!imu_can_init()) {
    while (1) {
    }
  }

  __HAL_RCC_GPIOA_CLK_ENABLE();

  // Configure GPIO PIN5;
  GPIO_InitTypeDef GPIO_InitStruct;
  GPIO_InitStruct.Pin = GPIO_PIN_5;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  // create blinking led task
  xTaskCreate(blink_led, "BLINK_LED", 128, NULL, 1, NULL);
  if (xTaskCreate(imu_can_publish_task, "IMU_CAN", 256, NULL, 2, NULL) != pdPASS) {
    while (1) {
    }
  }
  // start FreeRTOS Scheduler
  vTaskStartScheduler();

  while (1)
    ;
}