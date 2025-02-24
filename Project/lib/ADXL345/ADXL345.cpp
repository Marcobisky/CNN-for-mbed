#include <myMbed.h>
#include <ADXL345.h>

SPI spi(A6, A5, A4); // MOSI, MISO, SCLK
DigitalOut chipSelect(D12); // Chip select


void ADXL345_SPI_Initialise()
{
    // Set data format to full resolution, +/-16g
    spi.format(8, 3); // 8 bits per frame, mode 3
    spi.frequency(2000000); // 2 MHz clock rate

    chipSelect = 1; // Deselect the accelerometer

    // Configure ADXL345
    chipSelect = 0; // Select the accelerometer

    // Write to ADXL345 registers for configuration
    spi.write(ADXL345_REG_POWER_CTL); // Power_CTL register
    spi.write(0x08); // Enable Measurement mode, 8Hz sleep mode

    spi.write(ADXL345_REG_DATA_FORMAT); // Data_Format register
    spi.write(0x0B); // Full resolution, +/-16g

    chipSelect = 1; // Deselect the accelerometer
}

int16_t readAxis(uint8_t regAddress)
{
    int8_t data[2];
    int16_t result;
    chipSelect = 0; // Select the accelerometer

    // Read two bytes of data from the specified register address
    spi.write(regAddress | 0x80 | 0x40); // Set MSB to 1 for reading, multiple bytes read
    data[0] = spi.write(DATA_REQUEST);
    data[1] = spi.write(DATA_REQUEST);

    chipSelect = 1; // Deselect the accelerometer

    // Combine the two bytes to form a 16-bit signed integer
    result = (data[1] << 8) | data[0];
    return result;
}

void get_acceleration_xyz(float *x_acc, float *y_acc, float *z_acc)
{
    // Read accelerometer data
    int16_t x_raw = readAxis(ADXL345_REG_DATAX0);
    int16_t y_raw = readAxis(ADXL345_REG_DATAY0);
    int16_t z_raw = readAxis(ADXL345_REG_DATAZ0);

    // Convert raw data to floating point values
    *x_acc = x_raw * 0.0039; // Sensitivity: 3.9mg/LSB for +/-16g
    *y_acc = y_raw * 0.0039;
    *z_acc = z_raw * 0.0039;
}

float get_accelaration_3d(void)
{
    // Read accelerometer data
    int16_t x_raw = readAxis(ADXL345_REG_DATAX0);
    int16_t y_raw = readAxis(ADXL345_REG_DATAY0);
    int16_t z_raw = readAxis(ADXL345_REG_DATAZ0);

    // Convert raw data to floating point values
    float x_acc = x_raw * 0.0039; // Sensitivity: 3.9mg/LSB for +/-16g
    float y_acc = y_raw * 0.0039;
    float z_acc = z_raw * 0.0039;

    // Convert to 3D acceleration
    float acc_3d = sqrt(x_acc*x_acc + y_acc*y_acc + z_acc*z_acc);

    return acc_3d;
}


