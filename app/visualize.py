import matplotlib.pyplot as plt
import numpy as np
import os


motion = np.loadtxt("output/motion.txt", dtype=float)
smoothMotion = np.loadtxt("output/smoothMotion.txt", dtype=float)
smoothMotion2 = np.loadtxt("output/smoothMotion2.txt", dtype=float)

plt.plot(motion[:500, 0].tolist())
plt.plot(smoothMotion[:500, 0].tolist())
plt.plot(smoothMotion2[:500, 0].tolist())
plt.show()


plt.plot(motion[:500, 1].tolist())
plt.plot(smoothMotion[:500, 1].tolist())
plt.plot(smoothMotion2[:500, 1].tolist())
plt.show()
