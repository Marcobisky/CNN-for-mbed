// #ifndef __TMP102_H__
// #define __TMP102_H__
#pragma once

void CCM_init(void);
void SD_init(void);
void One_Shot_Meas(void);
float Read_temp_Reg_C(void);
void LEDs_Init(void);
void LEDs_Ctrl(float temp);

// #endif // __TMP102_H__