import DarkImu
import numpy as np
import matplotlib.pyplot as plt

# Input file
fname = "settings.yml"

# Initialize IMU
imu = DarkImu.DarkImu(fname)

## Trajectory
# Tabletop Sim: 1g and earth rate
w = np.array([ 0, 0, 7.29e-5])
sf = np.array([0, 0, 1])

# Freefall: Zero input
w = np.array([ 0, 0, 0])
sf = np.array([0, 0, 0])
dt = 1/100

# Timing
time = 4  # seconds
n = int(time / dt)
t = np.linspace(dt, 4, n)

# Container for results
deltaState = np.zeros([n, 6])


for i in range(n):
    deltaState[i, :] = imu.getImuOutput(w, sf, dt)

# Plot the results
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15,5))
plt.suptitle("IMU Signal")
ax1.plot(t, deltaState[:, 0], label='X')
ax1.plot(t, deltaState[:, 1], label='Y')
ax1.plot(t, deltaState[:, 2], label='Z')
ax1.set_ylabel('Delta V (m/s)')
ax1.legend()

ax2.plot(t, deltaState[:, 3])
ax2.plot(t, deltaState[:, 4])
ax2.plot(t, deltaState[:, 5])
ax2.set_xlabel('Time')
ax2.set_ylabel('Delta Theta (rad)')
plt.show()