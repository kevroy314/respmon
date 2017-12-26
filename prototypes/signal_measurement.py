import numpy as np
import matplotlib.pyplot as plt

measure_buffer_length = 128  # Hyperparameter (from main.py)

# filename = "timber.mp4.npy"
filename = "timber2.mp4.npy"
t, data = np.transpose(np.load(r"C:\Users\kevin\Desktop\Active Projects\Video Magnification Videos\\" + filename))

plt.plot(t, data)
plt.show()
