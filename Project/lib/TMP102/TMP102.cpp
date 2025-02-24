#include <myMbed.h>

I2C i2c(D0, D1); // D0:SDA, D1:SCL
BusOut LEDs(A3, A2, A1, A0, D6, D9, D10, D11); // Control LEDs, LSB-MSB: (LXR1, LXR2, LXG2, LXG1)

DigitalOut Not_wearing(D2); // Not wearing detection

// TMP102 I2C slave address when ADD0 to GND
const int addr = 0x90;

// config for store configuration information
char config[3];


// Continuous Conversion Mode Configuration
void CCM_init(void)
{
    config[0] = 0x01;  // Pointer to configuration register
    config[1] = 0x60;  // MSB, 12-bit resolution, 0110 0000
    config[2] = 0xA0;  // LSB, Continuous conversion, 4Hz, 1010 0000
    i2c.write(addr, config, 3); 
}

// Shutdown Mode Configuration, usually followed by One_Shot_Meas()
void SD_init(void)
{
    config[0] = 0x01; // Point to the configuration register
    config[1] = 0x61; // MSB, 12-bit resolution, SD Mode, 0110 0001
    config[2] = 0xA0; // LSB, no change from default
    i2c.write(addr, config, 3);
}

// Perform just one measurement (in SD mode), should be followed by Read_temp_Reg_C()
void One_Shot_Meas(void)
{
    // Perform a one-shot measurement
    config[0] = 0x01;    // Point to the configuration register
    config[1] = 0x81;    // MSB of the config register, OS bit set to 1 (one-shot) D1???
    config[2] = 0x00;    // LSB of the config register
    i2c.write(addr, config, 3);

    // Wait for the measurement to complete (conversion time)
    // ThisThread::sleep_for(30); // 26ms on the datasheet
    wait_us(30000);
}

// Read the temperature register and returns the temperature in Celsius
float Read_temp_Reg_C(void)
{
    char temp_read[2]; // Array to store temperature bytes (12 bits) from sensor
    float temperature; // Store the temperature in celcius
    
    // Set pointer to temperature register
    config[0] = 0x00;
    i2c.write(addr, config, 1);

    // Read temperature data
    i2c.read(addr, temp_read, 2);
    
    // Convert the temperature data
    int16_t raw = (temp_read[0] << 4) | (temp_read[1] >> 4);
    if (raw & 0x800) { // Check if negative
        raw |= 0xF000; // Extend sign bits
    }
    temperature = 0.0625 * raw; // Calculate the temperature in Celsius
    return temperature;
}

void LEDs_Init(void)
{
    LEDs = 0x00; //Initialize the LEDs = 0000 0000
    Not_wearing = 1; // Initialize the Not_wearing = 1
}

void LEDs_Ctrl(float temp)
{
    // For easier demonstration
    float temp_test = temp + 10;

    if (temp_test <= 34 || temp_test >= 50) {
        LEDs = 0x00;
        Not_wearing = 1;
        return;
    }else{
        Not_wearing = 0;
    }
    if(temp_test > 35.5 && temp_test < 36.5)  LEDs = 0x01;
    else if(temp_test > 35.4 && temp_test < 37)   LEDs = 0x03;
    else if(temp_test > 35.3 && temp_test < 37.4)   LEDs = 0x07;
    else if(temp_test > 35.2 && temp_test < 37.8)   LEDs = 0x0F;
    else if(temp_test > 35.1 && temp_test < 38)   LEDs = 0x1F;
    else if(temp_test > 35 && temp_test < 38.3)   LEDs = 0x3F;
    else if(temp_test > 34.8 && temp_test < 38.5)   LEDs = 0x7F;
    else if (temp_test > 34 && temp_test < 50) LEDs = 0xFF;
}
// {
//     if(x_acc>-0.5 && x_acc<-0.05)       LEDs = (LEDs&0xF0)|0x01; // LEDs |= 1000 0000
//     else if(x_acc<-0.5)                 LEDs = (LEDs&0xF0)|0x02; // LEDs |= 0100 0000
//     else if(x_acc>0.05 && x_acc<0.5)    LEDs = (LEDs&0xF0)|0x04; // LEDs |= 0010 0000
//     else if(x_acc>0.5)                  LEDs = (LEDs&0xF0)|0x08; // LEDs |= 0001 0000
//     else                                LEDs = 0x00;

//     if(y_acc>-0.5 && y_acc<-0.05)       LEDs = (LEDs&0x0F)|0x10; // LEDs |= 0000 1000
//     else if(y_acc<-0.5)                 LEDs = (LEDs&0x0F)|0x20; // LEDs |= 0000 0100
//     else if(y_acc>0.05 && y_acc<0.5)    LEDs = (LEDs&0x0F)|0x40; // LEDs |= 0000 0010
//     else if(y_acc>0.5)                  LEDs = (LEDs&0x0F)|0x80; // LEDs |= 0000 0001
//     else                                LEDs = 0x00;
// }