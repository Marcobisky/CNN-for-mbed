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
        # Set y limit to 1
        ax4.set_ylim(0, 1)
        

    if temperature is not None:
        data['temperature_time'].append(len(data['temperature_time']) * 2.5)
        data['temperature'].append(temperature)
        temp_line.set_data(data['temperature_time'], data['temperature'])
        ax3.relim()
        ax3.autoscale_view()

    return lines + [scatters, temp_line, probability_line]

# Main function
def main():
    serial_port_path = '/dev/cu.usbmodem1103'  # Replace with your actual serial port
    serial_port = serial.Serial(serial_port_path, 9600)
    data = {'x': [], 'y': [], 'z': [], 'Probability': [], 'Probability_time': [], 'total': [], 'time': [], 'temperature': [], 'temperature_time': []}

    fig = plt.figure(figsize=(18, 6))

    # Subplot 1: Line plots for x, y, z, and total acceleration over time
    global ax1
    ax1 = fig.add_subplot(3, 2, 1)
    lines = [
        ax1.plot([], [], lw=0.5, label='X Acceleration')[0],
        ax1.plot([], [], lw=0.5, label='Y Acceleration')[0],
        ax1.plot([], [], lw=0.5, label='Z Acceleration')[0],
        ax1.plot([], [], lw=1.5, label='Total Acceleration')[0],
    ]
    ax1.axhline(y=SAFE_TOTAL_ACCELERATION, color='r', linestyle='--', label='Safe Total Acceleration')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Acceleration (g)')
    ax1.set_title('Real-time Acceleration vs Time')
    ax1.legend()

    # Subplot 2: 3D scatter plot for x, y, z acceleration
    global ax2
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    scatters = ax2.scatter([], [], [], c=[], s=20)
    ax2.set_xlabel('X Acceleration')
    ax2.set_ylabel('Y Acceleration')
    ax2.set_zlabel('Z Acceleration')
    ax2.set_title('Real-time 3D Acceleration Data')
    ax2.legend(['Low (<1.25)', 'Medium (1.25-6)', 'High (>6)'])

    # Subplot 3: Temperature over time
    global ax3
    ax3 = fig.add_subplot(3, 2, 5)
    temp_line, = ax3.plot([], [], lw=2, label='Temperature')
    ax3.axhline(y=TEMPERATURE_SAFETY_UPPER_LIMIT, color='r', linestyle='--', label='Temperature Safety Upper Limit')
    ax3.axhline(y=TEMPERATURE_SAFETY_LOWER_LIMIT, color='k', linestyle='--', label='Temperature Safety Lower Limit') 
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Temperature (C)')
    ax3.set_title('Real-time Temperature vs Time')
    ax3.legend()
    
    # Subplot 4: Probability of falling over time
    global ax4
    ax4 = fig.add_subplot(3, 2, 3)
    probability_line, = ax4.plot([], [], lw=2, label='Probability of Falling')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Probability of Falling')
    ax4.set_title('Real-time Probability of Falling vs Time')
    
    # Create animation
    ani = animation.FuncAnimation(fig, update_plots, fargs=(serial_port, data, lines, scatters, temp_line, probability_line), interval=100, blit=False)
    plt.show()

if __name__ == "__main__":
    main()