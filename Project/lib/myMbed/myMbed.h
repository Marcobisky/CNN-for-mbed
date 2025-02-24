#include <drivers/SPI.h>
#include <drivers/I2C.h>
#include <drivers/BusOut.h>
#include "targets/TARGET_STM/TARGET_STM32L4/TARGET_STM32L432xC/TARGET_NUCLEO_L432KC/PinNames.h"
#include <cstdio>
#include "rtos/ThisThread.h"
#include "platform/mbed_wait_api.h"
#include "platform/Callback.h"
#include "platform/mbed_thread.h"
#include <cstdint>
#include "drivers/DigitalIn.h"
#include "drivers/DigitalOut.h"
#include "drivers/PwmOut.h"
#include "drivers/InterruptIn.h"

#include <cmath> // For sqrt() function

using namespace mbed;
using namespace rtos;