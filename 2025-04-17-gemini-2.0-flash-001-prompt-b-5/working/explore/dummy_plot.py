import matplotlib.pyplot as plt
import numpy as np

# Create some dummy data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create the plot
plt.figure(figsize=(6, 4))
plt.plot(x, y)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Dummy Plot")
plt.savefig("explore/dummy_plot.png")
plt.close()

print("Dummy plot created successfully in explore/dummy_plot.png")