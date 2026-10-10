#ifndef FREERTOS_CONFIG_H
#define FREERTOS_CONFIG_H

#include "stm32g4xx.h"

#define configUSE_PREEMPTION                         1
#define configUSE_IDLE_HOOK                          0
#define configUSE_TICK_HOOK                          0

#define configCPU_CLOCK_HZ                           (SystemCoreClock)
#define configTICK_RATE_HZ                           1000

#define configMAX_PRIORITIES                         5
#define configMINIMAL_STACK_SIZE                     128
#define configUSE_16_BIT_TICKS                       0

#define configUSE_MUTEXES                            1

#define configSUPPORT_DYNAMIC_ALLOCATION             1
#define configTOTAL_HEAP_SIZE                        (8 * 1024)

#define configPRIO_BITS                              4
#define configLIBRARY_LOWEST_INTERRUPT_PRIORITY      15
#define configLIBRARY_MAX_SYSCALL_INTERRUPT_PRIORITY 5

#define configKERNEL_INTERRUPT_PRIORITY \
    (configLIBRARY_LOWEST_INTERRUPT_PRIORITY << (8 - configPRIO_BITS))

#define configMAX_SYSCALL_INTERRUPT_PRIORITY \
    (configLIBRARY_MAX_SYSCALL_INTERRUPT_PRIORITY << (8 - configPRIO_BITS))

#define INCLUDE_vTaskSuspend                         1
#define INCLUDE_vTaskDelay                           1
#define INCLUDE_vTaskDelayUntil                      1
#define INCLUDE_vTaskPrioritySet                     1

#define vPortSVCHandler                              SVC_Handler
#define xPortPendSVHandler                           PendSV_Handler
#define xPortSysTickHandler                          SysTick_Handler

#endif // FREERTOS_CONFIG_H
