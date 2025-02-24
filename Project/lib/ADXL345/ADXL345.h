// #ifndef __ADXL345_H__
// #define __ADXL345_H__
#pragma once
#include <cstdint>

// ADXL345 Register Addresses
#define ADXL345_REG_DEVID        0x00
#define ADXL345_REG_POWER_CTL    0x2D
#define ADXL345_REG_DATA_FORMAT  0x31
#define ADXL345_REG_DATAX0       0x32
#define ADXL345_REG_DATAX1       0x33
#define ADXL345_REG_DATAY0       0x34
#define ADXL345_REG_DATAY1       0x35
#define ADXL345_REG_DATAZ0       0x36
#define ADXL345_REG_DATAZ1       0x37
#define DATA_REQUEST             0xFF


void ADXL345_SPI_Initialise(void);
int16_t readAxis(uint8_t regAddress);
void get_acceleration_xyz(float *x_acc, float *y_acc, float *z_acc);
float get_accelaration_3d(void);


// #endif // __ADXL345_H__