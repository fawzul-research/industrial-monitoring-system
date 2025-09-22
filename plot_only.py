import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

def draw_pie_chart():
    # Data for the pie chart
    labels = ['A', 'B', 'C', 'D']
    sizes = [15, 30, 45, 10]

    # Calculate the size in inches based on pixels and DPI
    dpi = plt.rcParams['figure.dpi']
    width_px = 800  # Specify the width in pixels
    height_px = 600  # Specify the height in pixels
    width_inch = width_px / dpi
    height_inch = height_px / dpi

    # Create a figure with desired size (in inches)
    fig, ax = plt.subplots(figsize=(width_inch, height_inch))

    # Draw the pie chart
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

    # Display the pie chart in the Tkinter GUI
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack()

def exit_application():
    root.quit()  # Close the Tkinter window

# Create the Tkinter GUI window
root = tk.Tk()
root.title("Pie Chart in Tkinter")

# Set fixed window size
window_width = 800
window_height = 600
root.geometry(f"{window_width}x{window_height}")

# Button to draw the pie chart
draw_button = tk.Button(root, text="Draw Pie Chart", command=draw_pie_chart)
draw_button.pack()

# Button to exit the application
exit_button = tk.Button(root, text="Exit", command=exit_application)
exit_button.pack()

# Run the Tkinter event loop
root.mainloop()
