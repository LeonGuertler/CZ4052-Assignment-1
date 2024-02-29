import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection

# Parameters
MAX_ITER = 1000
THRERESHOLD = 10

# User function class
class UserFunc:
    def __init__(self):
        self.param_1 = 1
        self.param_2 = 0.5

    def alpha_function(self, x:float) -> float:
        return x + self.param_1
    
    def beta_function(self, x:float) -> float:
        return x * self.param_2


# Instances
uf_1 = UserFunc()
uf_2 = UserFunc()

# Arrays for storing values
x1_values = np.zeros(MAX_ITER)
x2_values = np.zeros(MAX_ITER)
time_steps = np.arange(MAX_ITER)  # Time dimension
packages_lost = 0
percent_utilization = []

# Initial conditions
x1 = 1
x2 = 2

# Iteration process
for i in range(MAX_ITER):
    if (x1 + x2 <= THRERESHOLD):
        x1 = uf_1.alpha_function(x1)
        x2 = uf_2.alpha_function(x2)
    else:
        x1 = uf_1.beta_function(x1)
        x2 = uf_2.beta_function(x2)
        packages_lost += 2

    x1_values[i] = x1
    x2_values[i] = x2
    percent_utilization.append((x1 + x2) / THRERESHOLD)


# Display the final values
print("Final x1:", x1)
print("Final x2:", x2)
print("Packages Lost:", packages_lost)
print("Percent Utilization:", np.mean(percent_utilization))

# 3D Plot
fig = plt.figure(figsize=(14, 6))

points = np.array([x1_values, x2_values]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
norm = plt.Normalize(time_steps.min(), time_steps.max())
lc = LineCollection(segments, cmap='viridis', norm=norm, linewidth=2)
lc.set_array(time_steps)

# 3D plot with color coding
ax1 = fig.add_subplot(121, projection='3d')

# Create a color map based on the time steps for the 3D line
colors = plt.get_cmap('viridis')(norm(time_steps))

# Plot each segment with a color corresponding to its time step
for i in range(len(segments)):
    ax1.plot3D(segments[i][:, 0], segments[i][:, 1], [time_steps[i], time_steps[i+1]], color=colors[i])

ax1.set_title('3D Plot of x1 and x2 over Time with Color Gradient')
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.set_zlabel('Time')

# Second subplot for gradient line plot
ax2 = fig.add_subplot(122)
ax2.add_collection(lc)
ax2.autoscale()
fig.colorbar(lc, ax=ax2).set_label('Time Step')
ax2.set_title('Line Plot of x1 and x2 with Time Gradient')
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')

#plt.tight_layout()
plt.show()


