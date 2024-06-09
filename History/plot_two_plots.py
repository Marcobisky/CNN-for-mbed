import serial
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# Define constants
SAFE_TOTAL_ACCELERATION = 1.25
MAX_TOTAL_ACCELERATION = 6

# Function to read serial data
def read_serial_data(serial_port):
    line = serial_port.readline().decode('utf-8').strip()
    try:
        acc_x, acc_y, acc_z = map(float, line.split(','))
        total_acc = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
        print(f"Read values: x={acc_x}, y={acc_y}, z={acc_z}, total={total_acc}")
    except ValueError:
        acc_x, acc_y, acc_z, total_acc = None, None, None, None
    return acc_x, acc_y, acc_z, total_acc

# Function to update plots
def update_plots(i, serial_port, data, lines, scatter):
    acc_x, acc_y, acc_z, total_acc = read_serial_data(serial_port)
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
        scatter._offsets3d = (data['x'], data['y'], data['z'])
        colors = []
        for t in data['total']:
            if t < SAFE_TOTAL_ACCELERATION:
                colors.append('green')
            elif SAFE_TOTAL_ACCELERATION <= t < MAX_TOTAL_ACCELERATION:
                colors.append('purple')
            else:
                colors.append('red')
        scatter.set_color(colors)

        # Update limits for scatter plot
        ax2.set_xlim(min(data['x']), max(data['x']))
        ax2.set_ylim(min(data['y']), max(data['y']))
        ax2.set_zlim(min(data['z']), max(data['z']))
    return lines + [scatter]

# Main function
def main():
    serial_port_path = '/dev/cu.usbmodem1103'  # Replace with your actual serial port
    serial_port = serial.Serial(serial_port_path, 9600)
    data = {'x': [], 'y': [], 'z': [], 'total': [], 'time': []}

    fig = plt.figure(figsize=(14, 4))

    # Left subplot: Line plots for x, y, z, and total acceleration over time
    global ax1
    ax1 = fig.add_subplot(121)
    lines = [
        ax1.plot([], [], lw=1, label='X Acceleration')[0],
        ax1.plot([], [], lw=1, label='Y Acceleration')[0],
        ax1.plot([], [], lw=1, label='Z Acceleration')[0],
        ax1.plot([], [], lw=1, label='Total Acceleration')[0]
    ]
    ax1.axhline(y=SAFE_TOTAL_ACCELERATION, color='r', linestyle='--', label='Safe Total Acceleration')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Acceleration (g)')
    ax1.set_title('Real-time Acceleration vs Time')
    ax1.legend()

    # Right subplot: 3D scatter plot for x, y, z acceleration
    global ax2
    ax2 = fig.add_subplot(122, projection='3d')
    scatter = ax2.scatter([], [], [], c=[], s=20)
    ax2.set_xlabel('X Acceleration')
    ax2.set_ylabel('Y Acceleration')
    ax2.set_zlabel('Z Acceleration')
    ax2.set_title('Real-time 3D Acceleration Data')
    ax2.legend(['Low (<1.25)', 'Medium (1.25-6)', 'High (>6)'])

    # Create animation
    ani = animation.FuncAnimation(fig, update_plots, fargs=(serial_port, data, lines, scatter), interval=200, blit=False)
    plt.show()

if __name__ == "__main__":
    main()