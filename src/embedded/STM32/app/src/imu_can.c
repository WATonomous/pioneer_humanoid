#include "../inc/imu_can.h"
#include "stm32g4xx_hal.h"

static FDCAN_HandleTypeDef hfdcan1;

static void pack_int16_big_endian(uint8_t* destination, int16_t value) {
  uint16_t raw = (uint16_t)value;

  destination[0] = (uint8_t)(raw >> 8);
  destination[1] = (uint8_t)(raw & 0xFFU);
}

void imu_can_pack(uint8_t payload[IMU_CAN_PAYLOAD_SIZE], const ImuCanData* imu) {
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

bool imu_can_init(void) {
  RCC_PeriphCLKInitTypeDef peripheral_clock = {0};

  /*
   * The repository configures PCLK1 to 42 MHz.
   * Use it as the FDCAN peripheral clock.
   */
  peripheral_clock.PeriphClockSelection = RCC_PERIPHCLK_FDCAN;
  peripheral_clock.FdcanClockSelection = RCC_FDCANCLKSOURCE_PCLK1;

  if (HAL_RCCEx_PeriphCLKConfig(&peripheral_clock) != HAL_OK) {
    return false;
  }

  /* Enable the FDCAN peripheral and GPIO port clocks. */
  __HAL_RCC_FDCAN_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();

  /*
   * PA11 receives the STM32 logic-level CAN signal.
   * PA12 transmits the STM32 logic-level CAN signal.
   * A physical CAN transceiver sits between these pins and CANH/CANL.
   */
  GPIO_InitTypeDef gpio = {0};
  gpio.Pin = GPIO_PIN_11 | GPIO_PIN_12;
  gpio.Mode = GPIO_MODE_AF_PP;
  gpio.Pull = GPIO_NOPULL;
  gpio.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  gpio.Alternate = GPIO_AF9_FDCAN1;
  HAL_GPIO_Init(GPIOA, &gpio);

  hfdcan1.Instance = FDCAN1;
  hfdcan1.Init.ClockDivider = FDCAN_CLOCK_DIV1;
  hfdcan1.Init.FrameFormat = FDCAN_FRAME_FD_BRS;
  hfdcan1.Init.Mode = FDCAN_MODE_NORMAL;
  hfdcan1.Init.AutoRetransmission = ENABLE;
  hfdcan1.Init.TransmitPause = DISABLE;
  hfdcan1.Init.ProtocolException = DISABLE;

  /*
   * Nominal CAN rate:
   * 42 MHz / [4 × (1 + 16 + 4)] = 500 kbit/s
   */
  hfdcan1.Init.NominalPrescaler = 4;
  hfdcan1.Init.NominalSyncJumpWidth = 4;
  hfdcan1.Init.NominalTimeSeg1 = 16;
  hfdcan1.Init.NominalTimeSeg2 = 4;

  /*
   * CAN-FD data rate:
   * 42 MHz / [1 × (1 + 16 + 4)] = 2 Mbit/s
   */
  hfdcan1.Init.DataPrescaler = 1;
  hfdcan1.Init.DataSyncJumpWidth = 4;
  hfdcan1.Init.DataTimeSeg1 = 16;
  hfdcan1.Init.DataTimeSeg2 = 4;

  /*
   * This bridge only transmits for now, so receive filters are unnecessary.
   */
  hfdcan1.Init.StdFiltersNbr = 0;
  hfdcan1.Init.ExtFiltersNbr = 0;
  hfdcan1.Init.TxFifoQueueMode = FDCAN_TX_FIFO_OPERATION;

  if (HAL_FDCAN_Init(&hfdcan1) != HAL_OK) {
    return false;
  }

  if (HAL_FDCAN_ConfigTxDelayCompensation(&hfdcan1, 16, 0) != HAL_OK) {
    return false;
  }

  if (HAL_FDCAN_EnableTxDelayCompensation(&hfdcan1) != HAL_OK) {
    return false;
  }

  if (HAL_FDCAN_Start(&hfdcan1) != HAL_OK) {
    return false;
  }

  return true;
}

bool imu_can_send(const ImuCanData* imu) {
  FDCAN_TxHeaderTypeDef header = {0};
  uint8_t payload[IMU_CAN_PAYLOAD_SIZE] = {0};

  if (imu == NULL) {
    return false;
  }

  imu_can_pack(payload, imu);

  header.Identifier = IMU_CAN_EXTENDED_ID;
  header.IdType = FDCAN_EXTENDED_ID;
  header.TxFrameType = FDCAN_DATA_FRAME;
  header.DataLength = FDCAN_DLC_BYTES_20;
  header.ErrorStateIndicator = FDCAN_ESI_ACTIVE;
  header.BitRateSwitch = FDCAN_BRS_ON;
  header.FDFormat = FDCAN_FD_CAN;
  header.TxEventFifoControl = FDCAN_NO_TX_EVENTS;
  header.MessageMarker = 0;

  return HAL_FDCAN_AddMessageToTxFifoQ(&hfdcan1, &header, payload) == HAL_OK;
}
