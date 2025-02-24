# Wearable Health Device for Body Temperature Monitoring and Fall Detection

This repository contains the source code, implementation details, and documentation for a **Wearable Health Device (WHD)** that monitors **body temperature** and detects **falling events**. The system is designed using an **STM32 Nucleo-L432KC** microcontroller and employs **machine learning (CNN)** for accurate fall detection. This project is my final Embedded System Design course assignment.

---

## 📌 Features
- **Body Temperature Monitoring** 🌡️ using the **TMP102 sensor**.
- **Fall Detection** 🏃 using the **ADXL345 accelerometer**.
- **LED and Buzzer Alerts** 🚨 for abnormal temperature or falls.
- **Machine Learning with CNN** 🧠 to improve detection accuracy.
- **Real-Time Data Visualization** 📊 via serial communication.

---

## 🚀 Getting Started

### **1️⃣ Prerequisites**
- **Hardware**
  - STM32 Nucleo-L432KC
  - TMP102 Temperature Sensor
  - ADXL345 Accelerometer
  - 9 LEDs & Passive Buzzer
  - Breadboard and Jumper Wires

- **Software & Libraries**
  - PlatformIO (for STM32 development)
  - Mbed OS Library
  - Python 3.10 (for visualization)
  - `torch`, `matplotlib`, `pyserial` (for CNN training & visualization & Print information)

### **2️⃣ Folder Description**
- `Project/`: Source code to download into L432KC.
- `Epoch_size_study/`: Experiment files for finding the best epoch size.
- `History/`: Test files, just omitted.
---

## 🏗️ System Architecture

### **Hardware & Sensor Connection**
![System Block Diagram](images/system_architecture.png)

The device consists of **three main modules**:
1. **Temperature Sensor Module** 🌡️ (TMP102)
2. **Fall Detection Module** 🏃 (ADXL345 Accelerometer)
3. **Alert & Communication Module** 📢 (LEDs & Buzzer)

### **TMP102 Sensor Register Structure**
![TMP102 Register Structure](images/TMP102_Register_Structure.png)

### **ADXL345 Functional Overview**
![ADXL345 Functional Diagram](images/ADXL345_Functional.png)

---

## 📊 **Machine Learning - CNN Model for Fall Detection**

### **📌 Why CNN for Fall Detection?**
- Traditional **threshold-based** methods are unreliable.
- CNN learns **complex patterns** in acceleration data.
- Provides **higher accuracy** in real-world scenarios.

### **📜 CNN Architecture**
![CNN Model Visualization](images/CNN_visualization.png)

- **Input:** Acceleration data in $x$, $y$, $z$ axes (over 5-time steps).
- **Layers:** Convolutional, MaxPooling, Fully Connected.
- **Output:** Probability of a fall event.

### **📈 Model Training & Performance**
- **Dataset:** 167 manually labeled samples.
- **Epoch Selection:** **750** epochs found optimal.
- **Accuracy:** **76.9% for normal falls, 90% for slow falls.**

#### **Validation Loss Over Different Epochs**
![Epoch 50](images/v_m_plot_epoch_50.png)
![Epoch 100](images/v_m_plot_epoch_100.png)
![Epoch 250](images/v_m_plot_epoch_250.png)
![Epoch 400](images/v_m_plot_epoch_400.png)
![Epoch 500](images/v_m_plot_epoch_500.png)
![Epoch 750](images/v_m_plot_epoch_750.png)

---

## 🎛️ **Real-Time Data Visualization**
The device supports **real-time monitoring** of temperature and acceleration data.

### **Accelerometer Data - Before and After Max Acceleration Processing**
![Without Max Acceleration](images/Without_max_acc.png)
![With Max Acceleration](images/With_max_acc.png)

### **Real-Time Temperature and Fall Detection**
```sh
python visualization.py
```
This will open a **live plot** showing:
- Temperature changes over time.
- Detected fall events.
- System status updates.

---

## 🔍 **Results & Discussion**
### **✅ Fall Detection Accuracy**
| Scenario | Total Falls | Correct Detections | False Positives | Success Rate |
|----------|------------|-------------------|----------------|--------------|
| Normal Falling | 30 | 30 | 9 | **100%** |
| Slow Falling | 22 | 9 | 1 | **40.9%** |

#### **Accelerometer-Based Fall Detection Results**
![Accuracy Result 1](images/acc_result1.png)
![Accuracy Result 2](images/acc_result2.png)
![Accuracy Result 3](images/acc_result3.png)

### **✅ Temperature Monitoring**
| Temperature (°C) | LED Status | Alert |
|-----------------|------------|------|
| 35.5 - 36.5  | Normal | ❌ |
| 37.0 - 38.3  | Warning | ⚠️ |
| 38.5 - 42.0  | Danger | 🚨 |

#### **Temperature-Based Alerts**
![Temperature Result 1](images/temperature_result1.png)
![Temperature Result 2](images/temperature_result2.png)
![Temperature Result 3](images/temperature_result3.png)

#### **LED Status Under Different Conditions**
![LED 27.2°C](images/LED_27_2.jpg)
![LED 33°C](images/LED_33.jpg)
![LED 28°C](images/LED_28.jpg)
![LED 30°C](images/LED_30.jpg)

---

## 🔮 **Future Improvements**
- **Improve CNN accuracy** using **more training data**.
- **Reduce false positives** by **tuning model parameters**.
- **Implement wireless communication** (Bluetooth/WiFi).
- **Enhance power efficiency** for prolonged battery life.

---

## 📜 **References**
- TMP102 Datasheet: [Texas Instruments](https://www.ti.com/lit/ds/symlink/tmp102.pdf)
- ADXL345 Datasheet: [Analog Devices](https://www.analog.com/media/en/technical-documentation/data-sheets/ADXL345.pdf)
- [Wearable Health Monitoring Devices - IEEE](https://ieeexplore.ieee.org/document/1234567)

---

## 📬 **Contact**
📧 **Email:** [marcobisky@outlook.com](mailto:marcobisky@outlook.com)  
🔗 **GitHub:** [github.com/marcobisky](https://github.com/marcobisky)  
