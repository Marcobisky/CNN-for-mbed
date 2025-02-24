#pragma once

#define INPUT_SIZE 15
#define CONV_OUT_CHANNELS 2
#define CONV_KERNEL_SIZE 3
#define CONV_STRIDE 1
#define CONV_PADDING 1
#define POOL_SIZE 2
#define FC1_SIZE 4
#define OUTPUT_SIZE 2

void forward(float input[INPUT_SIZE], float output[OUTPUT_SIZE]);
