import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def read_serial_data(serial_port):
    line = serial_port.readline().decode('utf-8').strip()
    try:
        value = float(line)
    except ValueError:
        value = None
    return value

def update_plot(i, serial_port, data, line):
    value = read_serial_data(serial_port)
    if value is not None:
        data.append(value)
        line.set_ydata(data)
        line.set_xdata(range(len(data)))

        # Dynamically adjust x-axis limits to fill the space
        ax = plt.gca()
        ax.relim()
        ax.autoscale_view()
        
        # Ensure the x-axis limits are set to the range of data
        ax.set_xlim(0, len(data) - 1)

    return line,

def main():
    serial_port = serial.Serial('/dev/cu.usbmodem1103', 9600)  # Change the port name based on your setup
    data = []

    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=2)
    ax.set_ylim(0, 10)  # Adjust based on expected range of acceleration
    ax.set_xlim(0, 100)  # Adjust based on expected number of data points
    ax.set_xlabel('Time')
    ax.set_ylabel('Acceleration (g)')
    ax.set_title('Real-time Acceleration Data')

    ani = animation.FuncAnimation(fig, update_plot, fargs=(serial_port, data, line), interval=1000)
    plt.show()

if __name__ == "__main__":
    main()