#include <myMbed.h>
#include <ADXL345.h>
#include <TMP102.h>
#include <CNN.h>

// PwmOut buzzer(D3); // Buzzer


////////// Version 4 //////////
int main()
{
    // Define variables
    float temperature; // For storing temperature value now
    float acc_3d = 0, acc_3d_max = 0; // For storing 3D acceleration value
    float acc_x = 0, acc_y = 0, acc_z = 0, acc_x_max = 0, acc_y_max = 0, acc_z_max = 0; // Initializing 3D acceleration value

    int ctr = 0; // Record times of measurement

    // For CNN
    float acc_buf[15]; // Buffer for 5 measurements of x, y, z acceleration
    int buf_index = 0; // Buffer index

    // Initialization
    ADXL345_SPI_Initialise(); // Initialize the ADXL345
    SD_init(); // Initialize the TMP102 to shutdown mode
    LEDs_Init(); // Initialize LEDs

    // Measuring temperature and accelerometer data
    while(true)
    {
        // Sampling x, y, z accelerometer data, T = 10ms
        get_acceleration_xyz(&acc_x, &acc_y, &acc_z);
        // Extract the maximum x, y, z acceleration
        if (abs(acc_x) > abs(acc_x_max))  acc_x_max = acc_x;
        if (abs(acc_y) > abs(acc_y_max))  acc_y_max = acc_y;
        if (abs(acc_z) > abs(acc_z_max))  acc_z_max = acc_z;

        wait_us(10000); // Delay for 10ms
        ctr++;
        // Every 0.25s, print accelerometer max data in the past 0.25s
        if (ctr % 25 == 0)
        {
            printf("%.4f, %.4f, %.4f\n", acc_x_max, acc_y_max, acc_z_max);
            // Calculate the max 3D acceleration
            acc_3d_max = sqrt(acc_x_max * acc_x_max + acc_y_max * acc_y_max + acc_z_max * acc_z_max);

            // Store measurements in the buffer
            acc_buf[buf_index++] = acc_x_max;
            acc_buf[buf_index++] = acc_y_max;
            acc_buf[buf_index++] = acc_z_max;

            // If buffer is full, perform forward propagation and print result
            if (buf_index == 15)
            {
                float output[OUTPUT_SIZE];
                forward(acc_buf, output);
                printf("Probability: %f\n", output[1]); // Print the probability of falling
                buf_index = 0; // Clear the buffer for next measurements

                // buzzer.period_ms(0); // Stop the buzzer
                // // If the probability of falling is greater than 0.7, buzz the buzzer
                // if (output[1] > 0.7)
                // {
                //     buzzer.period_ms(1);
                //     buzzer.write(0.5);
                // }else{
                //     buzzer.write(0);
                // }            
            }

            // Reset the maximum x, y, z acceleration
            acc_x_max = 0;
            acc_y_max = 0;
            acc_z_max = 0;
        }
        // Every 2.5s, measure and print temperature
        if (ctr == 250)
        {
            One_Shot_Meas(); // Perform one-shot measurement
            // Read temperature
            temperature = Read_temp_Reg_C();
            printf("Temperature: %.6f C\n", temperature);
            LEDs_Ctrl(28); // Control LEDs based on temperature
            ctr = 0;

            // buzzer.period_ms(0); // Stop the buzzer
            // // If the probability of falling is greater than 0.7, buzz the buzzer
            // if (temperature > 29)
            // {
            //     buzzer.period_us(500);
            //     buzzer.write(0.5);
            // }else{
            //     buzzer.write(0);
            // }    
        }
    }
}




////////// Version 3 //////////
// int main()
// {
//     // Define variables
//     float temperature; // For storing temperature value now
//     float acc_3d = 0, acc_3d_max = 0; // For storing 3D acceleration value
//     float acc_x = 0, acc_y = 0, acc_z = 0, acc_x_max = -20, acc_y_max = -20, acc_z_max = -20; // Initializing 3D acceleration value

//     int ctr = 0; // Record times of measurement

//     // For CNN
//     float acc_buf[15]; // Buffer for 5 measurements of x, y, z acceleration
//     int buf_index = 0; // Buffer index

//     // Initialization
//     ADXL345_SPI_Initialise(); // Initialize the ADXL345
//     LEDs_Init();
//     SD_init(); // Initialize the TMP102 to shutdown mode
//     LEDs_Init(); // Initialize LEDs

//     // Measuring temperature and accelerometer data
//     while(true)
//     {
//         // Sampling x, y, z accelerometer data, T = 10ms
//         get_acceleration_xyz(&acc_x, &acc_y, &acc_z);
//         // Extract the maximum x, y, z acceleration
//         if (acc_x > acc_x_max)  acc_x_max = acc_x;
//         if (acc_y > acc_y_max)  acc_y_max = acc_y;
//         if (acc_z > acc_z_max)  acc_z_max = acc_z;

//         wait_us(10000); // Delay for 10ms
//         ctr++;
//         // Every 0.25s, print accelerometer max data in the past 0.25s
//         if (ctr % 25 == 0)
//         {
//             printf("%.4f, %.4f, %.4f\n", acc_x_max, acc_y_max, acc_z_max);
//             // Calculate the max 3D acceleration
//             acc_3d_max = sqrt(acc_x_max * acc_x_max + acc_y_max * acc_y_max + acc_z_max * acc_z_max);
//             // If acc_3d_max is greater than 6 g, buzz the buzzer
//             if (acc_3d_max > 1.5)
//             {
//                 buzzer.period_ms(1);
//                 buzzer.write(0.5);
//             }else{
//                 buzzer.write(0);
//             }

//             // Store measurements in the buffer
//             acc_buf[buf_index++] = acc_x_max;
//             acc_buf[buf_index++] = acc_y_max;
//             acc_buf[buf_index++] = acc_z_max;

//             // If buffer is full, perform forward propagation and print result
//             if (buf_index == 15)
//             {
//                 float output[OUTPUT_SIZE];
//                 forward(acc_buf, output);
//                 printf("Probability: %f\n", output[1]); // Print the probability of falling
//                 buf_index = 0; // Clear the buffer for next measurements
//             }

            
//             // Reset the maximum x, y, z acceleration
//             acc_x_max = -20;
//             acc_y_max = -20;
//             acc_z_max = -20;
//         }
//         // Every 2.5s, measure and print temperature
//         if (ctr == 250)
//         {
//             One_Shot_Meas(); // Perform one-shot measurement
//             // Read temperature
//             temperature = Read_temp_Reg_C();
//             printf("Temperature: %.6f C\n", temperature);
//             LEDs_Ctrl(temperature); // Control LEDs based on temperature
//             ctr = 0;
//         }
//     }
// }


////////// Version 2: For getting training data //////////
// int main()
// {
//     // Define variables
//     float temperature; // For storing temperature value now
//     float acc_3d = 0, acc_3d_max = 0; // For storing 3D acceleration value
//     float acc_x = 0, acc_y = 0, acc_z = 0, acc_x_max = 0, acc_y_max = 0, acc_z_max = 0; // Initializing 3D acceleration value

//     int ctr = 0; // Record times of measurement

//     // Initialization
//     ADXL345_SPI_Initialise(); // Initialize the ADXL345
//     LEDs_Init();
//     SD_init(); // Initialize the TMP102 to shutdown mode
//     LEDs_Init(); // Initialize LEDs

//     // Measuring temperature and accelerometer data
//     while(true)
//     {
//         // Sampling x, y, z accelerometer data, T = 5ms
//         get_acceleration_xyz(&acc_x, &acc_y, &acc_z);
//         // Extract the maximum x, y, z acceleration
//         if (abs(acc_x) > abs(acc_x_max))  acc_x_max = acc_x;
//         if (abs(acc_y) > abs(acc_y_max))  acc_y_max = acc_y;
//         if (abs(acc_z) > abs(acc_z_max))  acc_z_max = acc_z;

//         wait_us(10000); // Delay for 10ms
//         ctr++;
//         // Every 0.25s, print accelerometer data
//         if (ctr % 25 == 0)
//         {
//             printf("%.4f, %.4f, %.4f\n", acc_x_max, acc_y_max, acc_z_max);
//             // Calculate the max 3D acceleration
//             acc_3d_max = sqrt(acc_x_max * acc_x_max + acc_y_max * acc_y_max + acc_z_max * acc_z_max);
//             // If acc_3d_max is greater than 6 g, buzz the buzzer
//             // if (acc_3d_max > 1.5)
//             // {
//             //     buzzer.period_ms(1);
//             //     buzzer.write(0.5);
//             // }else{
//             //     buzzer.write(0);
//             // }

//             // Reset the maximum x, y, z acceleration
//             acc_x_max = 0;
//             acc_y_max = 0;
//             acc_z_max = 0;
//         }
//         // Every 2s, measure and print temperature
//         // if (ctr == 200)
//         // {
//         //     One_Shot_Meas(); // Perform one-shot measurement
//         //     // Read temperature
//         //     temperature = Read_temp_Reg_C();
//         //     printf("Temperature: %.6f C\n", temperature);
//         //     LEDs_Ctrl(temperature); // Control LEDs based on temperature
//         //     ctr = 0;
//         // }
//     }
// }


////////// Version 1 //////////
// int main()
// {
//     // Define variables
//     float temperature; // For storing temperature value now
//     int ctr = 0; // Record times of measurement

//     // Initialization
//     ADXL345_SPI_Initialise(); // Initialize the ADXL345
//     printf("Starting ADXL345 accelerometer...\n");
//     LEDs_Init();
//     CCM_init(); // Initialize the TMP102 to continuous conversion mode
//     // SD_init(); // Initialize the TMP102 to shutdown mode
//     printf("Starting TMP102 temperature sensor...\n");

//     // Measuring temperature and accelerometer data
//     while(true)
//     {
//         // Read temperature
//         temperature = Read_temp_Reg_C();
//         printf("Temperature: %.6f C\n", temperature);

//         // Read accelerometer data
//         int16_t x_raw = readAxis(ADXL345_REG_DATAX0);
//         int16_t y_raw = readAxis(ADXL345_REG_DATAY0);
//         int16_t z_raw = readAxis(ADXL345_REG_DATAZ0);

//         // Convert raw data to floating point values
//         float x_acc = x_raw * 0.0039; // Sensitivity: 3.9mg/LSB for +/-16g
//         float y_acc = y_raw * 0.0039;
//         float z_acc = z_raw * 0.0039;

//         // Print the accelerometer data
//         printf("X-axis: %.2f g, Y-axis: %.2f g, Z-axis: %.2f g\n", x_acc, y_acc, z_acc);

//         // Control LEDs based on accelerometer data
//         LED_Ctrl(x_acc, y_acc);

//         // Delay for 1 second
//         // ThisThread::sleep_for(1000);
//         wait_us(500000);
//     }
// }
