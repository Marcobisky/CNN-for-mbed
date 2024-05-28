import serial
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# Define constants
SAFE_TOTAL_ACCELERATION = 1.25
MAX_TOTAL_ACCELERATION = 6
TEMPERATURE_SAFETY_UPPER_LIMIT = 28
TEMPERATURE_SAFETY_LOWER_LIMIT = 25

# Initialize data structure
data = {
    'time': [],
    'x': [],
    'y': [],
    'z': [],
    'total': [],
    'Probability_time': [],
    'Probability': [],
    'temperature_time': [],
    'temperature': []
}

# Initialize serial port
serial_port = serial.Serial('/dev/cu.usbmodem1103', 9600, timeout=1)

# Create plots
fig = plt.figure(figsize=(10, 8))

# Acceleration plot
ax1 = fig.add_subplot(321)
ax1.set_title('Acceleration Data')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Acceleration (g)')
line_x, = ax1.plot([], [], label='Acc X')
line_y, = ax1.plot([], [], label='Acc Y')
line_z, = ax1.plot([], [], label='Acc Z')
line_total, = ax1.plot([], [], label='Total Acc')
ax1.legend()

# Probability plot
ax4 = fig.add_subplot(323)
ax4.set_title('Probability Data')
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Probability')
probability_line, = ax4.plot([], [], label='Falling Probability')
ax4.legend()

# Temperature plot
ax3 = fig.add_subplot(325)
ax3.set_title('Temperature Data')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Temperature (C)')
temp_line, = ax3.plot([], [], label='Temperature')
ax3.legend()

# 3D scatter plot
ax2 = fig.add_subplot(122, projection='3d')
ax2.set_title('Acceleration Data in 3D Space')
scatters = ax2.scatter([], [], [], c='green')

# Function to read serial data
def read_serial_data(serial_port):
    line = serial_port.readline().decode('utf-8').strip()
    try:
        if line.startswith("Temperature:"):
            temperature = float(line.split(":")[1].strip().split()[0])
            return None, None, None, None, None, temperature
        elif line.startswith("Probability:"):
            probability = float(line.split(":")[1].strip().split()[0])
            return None, None, None, None, probability, None
        else:
            acc_x, acc_y, acc_z = map(float, line.split(','))
            total_acc = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
            print(f"Read values: x={acc_x}, y={acc_y}, z={acc_z}, total={total_acc}")
            return acc_x, acc_y, acc_z, total_acc, None, None
    except ValueError:
        return None, None, None, None, None, None

# Function to update plots
def update_plots(i, serial_port, data, lines, scatters, temp_line, probability_line):
    acc_x, acc_y, acc_z, total_acc, probability, temperature = read_serial_data(serial_port)
    if acc_x is not None and acc_y is not None and acc_z is not None and total_acc is not None:
        data['x'].append(acc_x)
        data['y'].append(acc_y)
        data['z'].append(acc_z)
        data['total'].append(total_acc)
        data['time'].append(len(data['time']) * 0.25)

        # Update line plots
        for line, key in zip(lines, ['x', 'y', 'z', 'total']):
            line.set_data(data['time'], data[key])
        
        ax1.relim()
        ax1.autoscale_view()

        # Update scatter plot
        scatters._offsets3d = (data['x'], data['y'], data['z'])
        colors = []
        for t in data['total']:
            if t < SAFE_TOTAL_ACCELERATION:
                colors.append('green')
            elif SAFE_TOTAL_ACCELERATION <= t < MAX_TOTAL_ACCELERATION:
                colors.append('purple')
            else:
                colors.append('red')
        scatters.set_color(colors)

        # Update limits for scatter plot
        ax2.set_xlim(min(data['x']), max(data['x']))
        ax2.set_ylim(min(data['y']), max(data['y']))
        ax2.set_zlim(min(data['z']), max(data['z']))
        
    if probability is not None:
        data['Probability_time'].append(len(data['Probability_time']) * 1.25)
        data['Probability'].append(probability)
        probability_line.set_data(data['Probability_time'], data['Probability'])
        ax4.relim()
        ax4.autoscale_view()

    if temperature is not None:
        data['temperature_time'].append(len(data['temperature_time']) * 2.5)
        data['temperature'].append(temperature)
        temp_line.set_data(data['temperature_time'], data['temperature'])
        ax3.relim()
        ax3.autoscale_view()

# Animation setup
lines = [line_x, line_y, line_z, line_total]
ani = animation.FuncAnimation(fig, update_plots, fargs=(serial_port, data, lines, scatters, temp_line, probability_line), interval=250)

plt.tight_layout()
plt.show()