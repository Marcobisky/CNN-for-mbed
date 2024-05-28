import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

def read_serial_data(serial_port):
    line = serial_port.readline().decode('utf-8').strip()
    try:
        acc_x, acc_y, acc_z = map(float, line.split(','))
        print(f"Read values: x={acc_x}, y={acc_y}, z={acc_z}")  # Debug print to show the read values
    except ValueError:
        acc_x, acc_y, acc_z = None, None, None
    return acc_x, acc_y, acc_z

def update_plot(i, serial_port, data, scatter):
    acc_x, acc_y, acc_z = read_serial_data(serial_port)
    if acc_x is not None and acc_y is not None and acc_z is not None:
        data['x'].append(acc_x)
        data['y'].append(acc_y)
        data['z'].append(acc_z)

        scatter._offsets3d = (data['x'], data['y'], data['z'])

        # Dynamically adjust the axes limits to fill the space
        # ax = plt.gca()
        # ax.set_xlim(min(data['x']), max(data['x']))
        # ax.set_ylim(min(data['y']), max(data['y']))
        # ax.set_zlim(min(data['z']), max(data['z']))
        
    return scatter,

def main():
    serial_port_path = '/dev/cu.usbmodem1103'
    serial_port = serial.Serial(serial_port_path, 9600)
    data = {'x': [], 'y': [], 'z': []}

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter([], [], [])

    ax.set_xlabel('X Acceleration')
    ax.set_ylabel('Y Acceleration')
    ax.set_zlabel('Z Acceleration')
    ax.set_title('Real-time 3D Acceleration Data')
    
    # Set fixed axes limits to ensure the plot is not resized
    ax.set_xlim(-1, 14)
    ax.set_ylim(-1, 14)
    ax.set_zlim(-1, 14)

    ani = animation.FuncAnimation(fig, update_plot, fargs=(serial_port, data, scatter), interval=250)
    plt.show()

if __name__ == "__main__":
    main()