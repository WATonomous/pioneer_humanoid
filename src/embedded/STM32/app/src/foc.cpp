
/** For setup
 * - initialize PWM
 * - initialize angle/position encoder (MT6835)
 * - initialize current sensing
 * - initialize motor driver
 * - initialize motor configurations (PID, mode) + motor.initFOC()
 * - initialize CANFD and enable RX ISR
 *
 */

#include "SimpleFOCDrivers.h"
#include "encoders/mt6835/MagneticSensorMT6835.h"
#include "stm32g4xx_hal.h"
#include <Arduino.h>
#include <SimpleFOC.h>

BLDCMotor motor = BLDCMotor(7);
bool setup_success = true;

float target_voltage = 0.0f;

#define M0_IN1_PIN PA_8
#define M0_IN2_PIN PA_9
#define M0_IN3_PIN PA_10
#define M0_EN_PIN PB_5
#define SENSOR_CS_PIN PB_6

BLDCDriver3PWM driver = BLDCDriver3PWM(M0_IN1_PIN, M0_IN2_PIN, M0_IN3_PIN, M0_EN_PIN);

SPISettings mt6835_spi_settings(1000000, MT6835_BITORDER, SPI_MODE3);

MagneticSensorMT6835 sensor(SENSOR_CS_PIN, mt6835_spi_settings);

FDCAN_HandleTypeDef hfdcan1;

constexpr uint8_t CAN_RX_QUEUE_SIZE = 8;
constexpr uint8_t CAN_RX_QUEUE_MASK = CAN_RX_QUEUE_SIZE - 1;

struct can_rx_message {

  FDCAN_RxHeaderTypeDef header;
  uint8_t data[64];
};

can_rx_message can_rx_queue[CAN_RX_QUEUE_SIZE];

volatile uint8_t can_rx_head = 0;
volatile uint8_t can_rx_tail = 0;
volatile uint32_t can_rx_dropped = 0;

bool initCANFD() {
  /*
   * Select PCLK1 as the FDCAN clock.
   *
   * The bit timings below assume PCLK1/FDCAN clock = 170 MHz.
   */
  RCC_PeriphCLKInitTypeDef peripheral_clock = {};

  peripheral_clock.PeriphClockSelection = RCC_PERIPHCLK_FDCAN;
  peripheral_clock.FdcanClockSelection = RCC_FDCANCLKSOURCE_PCLK1;

  if (HAL_RCCEx_PeriphCLKConfig(&peripheral_clock) != HAL_OK) {
    return false;
  }

  /* Enable peripheral and GPIO clocks. */
  __HAL_RCC_FDCAN_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();

  /*
   * PA11: FDCAN1_RX
   * PA12: FDCAN1_TX
   */
  GPIO_InitTypeDef gpio = {};

  gpio.Pin = GPIO_PIN_11 | GPIO_PIN_12;
  gpio.Mode = GPIO_MODE_AF_PP;
  gpio.Pull = GPIO_NOPULL;
  gpio.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  gpio.Alternate = GPIO_AF9_FDCAN1;

  HAL_GPIO_Init(GPIOA, &gpio);

  /* Configure the FDCAN peripheral. */
  hfdcan1.Instance = FDCAN1;

  hfdcan1.Init.ClockDivider = FDCAN_CLOCK_DIV1;
  hfdcan1.Init.FrameFormat = FDCAN_FRAME_FD_BRS;
  hfdcan1.Init.Mode = FDCAN_MODE_NORMAL;

  hfdcan1.Init.AutoRetransmission = ENABLE;
  hfdcan1.Init.TransmitPause = DISABLE;
  hfdcan1.Init.ProtocolException = DISABLE;

  /*
   * Nominal/arbitration rate:
   *
   * 170 MHz / [10 × (1 + 27 + 6)]
   * = 500 kbit/s
   */
  hfdcan1.Init.NominalPrescaler = 10;
  hfdcan1.Init.NominalSyncJumpWidth = 6;
  hfdcan1.Init.NominalTimeSeg1 = 27;
  hfdcan1.Init.NominalTimeSeg2 = 6;

  /*
   * Data-phase rate:
   *
   * 170 MHz / [5 × (1 + 13 + 3)]
   * = 2 Mbit/s
   */
  hfdcan1.Init.DataPrescaler = 5;
  hfdcan1.Init.DataSyncJumpWidth = 3;
  hfdcan1.Init.DataTimeSeg1 = 13;
  hfdcan1.Init.DataTimeSeg2 = 3;

  hfdcan1.Init.StdFiltersNbr = 1;
  hfdcan1.Init.ExtFiltersNbr = 0;
  hfdcan1.Init.TxFifoQueueMode = FDCAN_TX_FIFO_OPERATION;

  if (HAL_FDCAN_Init(&hfdcan1) != HAL_OK) {
    return false;
  }

  /*
   * Accept all 11-bit standard identifiers.
   *
   * Mask = 0 means no identifier bits need to match.
   */
  FDCAN_FilterTypeDef filter = {};

  filter.IdType = FDCAN_STANDARD_ID;
  filter.FilterIndex = 0;
  filter.FilterType = FDCAN_FILTER_MASK;
  filter.FilterConfig = FDCAN_FILTER_TO_RXFIFO0;
  filter.FilterID1 = 0x000;
  filter.FilterID2 = 0x000;

  if (HAL_FDCAN_ConfigFilter(&hfdcan1, &filter) != HAL_OK) {
    return false;
  }

  /*
   * Reject nonmatching messages and remote frames.
   * All standard data frames match the filter above.
   */
  if (HAL_FDCAN_ConfigGlobalFilter(&hfdcan1, FDCAN_REJECT, FDCAN_REJECT, FDCAN_REJECT_REMOTE,
                                   FDCAN_REJECT_REMOTE) != HAL_OK) {
    return false;
  }

  /* Enable RX FIFO 0 new-message notification. */
  if (HAL_FDCAN_ActivateNotification(&hfdcan1, FDCAN_IT_RX_FIFO0_NEW_MESSAGE, 0) != HAL_OK) {
    return false;
  }

  /*
   * Recommended when transmitting FD frames using BRS.
   * TDC offset = DataPrescaler × DataTimeSeg1 = 5 × 13.
   */
  if (HAL_FDCAN_ConfigTxDelayCompensation(&hfdcan1, 65, 0) != HAL_OK) {
    return false;
  }

  if (HAL_FDCAN_EnableTxDelayCompensation(&hfdcan1) != HAL_OK) {
    return false;
  }

  /*
   * Enable the interrupt in the ARM NVIC.
   * Priority 5 keeps it below very high-priority control interrupts.
   */
  HAL_NVIC_SetPriority(FDCAN1_IT0_IRQn, 5, 0);
  HAL_NVIC_EnableIRQ(FDCAN1_IT0_IRQn);

  if (HAL_FDCAN_Start(&hfdcan1) != HAL_OK) {
    return false;
  }

  return true;
}

extern "C" void FDCAN1_IT0_IRQHandler() {
  HAL_FDCAN_IRQHandler(&hfdcan1);
}

extern "C" void HAL_FDCAN_RxFifo0Callback(FDCAN_HandleTypeDef* hfdcan,
                                          uint32_t rx_fifo0_interrupts) {
  if (hfdcan->Instance != FDCAN1) {
    return;
  }

  if ((rx_fifo0_interrupts & FDCAN_IT_RX_FIFO0_NEW_MESSAGE) == 0U) {
    return;
  }

  /*
   * Drain all currently available messages from hardware FIFO 0.
   */
  while (HAL_FDCAN_GetRxFifoFillLevel(hfdcan, FDCAN_RX_FIFO0) > 0U) {
    const uint8_t head = can_rx_head;
    const uint8_t next = (head + 1U) & CAN_RX_QUEUE_MASK;

    /*
     * Read into the current empty ring-buffer entry.
     */
    if (HAL_FDCAN_GetRxMessage(hfdcan, FDCAN_RX_FIFO0, &can_rx_queue[head].header,
                               can_rx_queue[head].data) != HAL_OK) {
      break;
    }

    /*
     * If next == tail, the software queue is full.
     * The hardware message has still been drained, but it is discarded.
     */
    if (next == can_rx_tail) {
      can_rx_dropped++;
      continue;
    }

    /*
     * Ensure message data is written before publishing the new head.
     */
    __DMB();
    can_rx_head = next;
  }
}

bool popCANMessage(can_rx_message& message) {
  const uint8_t tail = can_rx_tail;

  if (tail == can_rx_head) {
    return false;
  }

  message = can_rx_queue[tail];

  __DMB();

  can_rx_tail = (tail + 1U) & CAN_RX_QUEUE_MASK;

  return true;
}

/*
 * Example protocol:
 *
 * Standard ID 0x100
 * Byte 0: target voltage low byte
 * Byte 1: target voltage high byte
 *
 * Value is a signed int16 in millivolts.
 *
 * Examples:
 *   1000  -> +1.000 V
 *   -500  -> -0.500 V
 */
void processCANMessage(const can_rx_message& message) {
  if (message.header.IdType != FDCAN_STANDARD_ID) {
    return;
  }

  if (message.header.Identifier != 0x100) {
    return;
  }

  if (message.header.DataLength != FDCAN_DLC_BYTES_2) {
    return;
  }

  int16_t target_millivolts = static_cast<int16_t>(static_cast<uint16_t>(message.data[0]) |
                                                   (static_cast<uint16_t>(message.data[1]) << 8U));

  float requested_voltage = static_cast<float>(target_millivolts) / 1000.0f;

  /* Clamp the command to the configured safety limit. */
  if (requested_voltage > motor.voltage_limit) {
    requested_voltage = motor.voltage_limit;
  } else if (requested_voltage < -motor.voltage_limit) {
    requested_voltage = -motor.voltage_limit;
  }

  target_voltage = requested_voltage;
}

void setup() {

  Serial.begin(115200);

  SimpleFOCDebug::enable(&Serial);

  // need to check with motor driver specs
  driver.voltage_power_supply = 12.0;
  // used for initial testing
  driver.voltage_limit = 2.0;
  // might need to be changed
  driver.pwm_frequency = 20000;

  if (!driver.init()) {
    Serial.println("Driver init failed!");
    setup_success = false;
    return;
  }

  // enable driver
  driver.enable();
  motor.linkDriver(&driver);
  sensor.init();
  motor.linkSensor(&sensor);

  motor.voltage_limit = 2.0;
  motor.controller = MotionControlType::torque;
  motor.torque_controller = TorqueControlType::voltage;

  motor.init();
  if (!motor.initFOC()) {
    Serial.println("Motor init failed!");
    setup_success = false;
    return;
  }

  if (!initCANFD()) {
    Serial.println("CAN FD initialization failed!");
    motor.disable();
    setup_success = false;
    return;
  }

  Serial.println("PWM, driver, and motor initialized.");
  delay(1000);
}

/** For loop
 *  -call loopFOC, move
 *  -check ringbuffer for messages
 *  -send telemetry data back
 *
 */
void loop() {
  if (!setup_success) {
    return;
  }
  can_rx_message message;

  if (popCANMessage(message)) {
    processCANMessage(message);
  }

  motor.loopFOC();
  motor.move(target_voltage);
}