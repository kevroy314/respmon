import os
import cv2
import logging
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import scipy.signal as signal
from prototypes.parabolic import parabolic


# Set up logging
logging.basicConfig(format="%(asctime)s :: %(message)s", level=logging.INFO)

# Input params
verbose = True
video_filename = r"C:\Users\kevin\Desktop\timber.mp4"
frame_buffer_length = 64

# Dynamic params
# logging.info(('Loading file {0}, {1} pyramid levels, {2}Hz to {3}Hz frequency range with {4}x amplification '
#              '(frame buffer length is {5} and skip levels is {6})').format(video_filename, pyramid_levels,
#                                                                            freq_min, freq_max, amplification,
#                                                                            frame_buffer_length, skip_levels_at_top))
# Load a video into a numpy array
logging.info("Loading {0}".format(video_filename))
if not os.path.isfile(video_filename):
    raise Exception("File Not Found: %s" % video_filename)
# noinspection PyArgumentList
capture = cv2.VideoCapture(video_filename)
frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(capture.get(cv2.CAP_PROP_FPS))
logging.info("Loaded video with {0} frames, {1}x{2} @ {3} fps.".format(frame_count, width, height, fps))

# Artificially capture frames
x = 0
vid_gray = np.zeros((frame_count, height, width), dtype='uint8')
while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        break
    vid_gray[x] = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    x += 1
capture.release()
logging.info("Done loading. Converting to floats.")

x = 539
y = 243
w = 67
h = 51


def freq_from_fft(sig, fs):
    """
    Estimate frequency from peak of FFT
    """
    # Compute Fourier transform of windowed signal
    windowed = sig * signal.blackmanharris(len(sig))
    freq_windowed = np.fft.rfft(windowed)

    # Find the peak and interpolate to get a more accurate peak
    max_idx = np.argmax(abs(freq_windowed))  # Just use this for less-accurate, naive version
    true_i = parabolic(np.log(abs(freq_windowed)), max_idx)[0]

    # Convert to equivalent frequency
    return fs * true_i / len(windowed)


data = deque()
t = deque()
freq = deque()

f, (ax1, ax2, ax3) = plt.subplots(1, 3)

plt.ion()
for i in range(0, len(vid_gray)):
    crop_img = vid_gray[i][y: y + h, x: x + w]
    # If the buffer is full, remove the oldest elements
    if len(data) > frame_buffer_length:
        data.popleft()
        t.popleft()
        freq.popleft()
    avg = np.average(crop_img)
    data.append(avg)
    if len(t) == 0:
        t.append(0.)
    else:
        t.append(t[-1] + (1. / fps))
    if len(data) > 10:
        freq.append(freq_from_fft(data, 1./fps)*60*fps)  # TODO: this isn't working
    else:
        freq.append(np.nan)
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax1.plot(t, data, c='b')
    ax2.imshow(crop_img)
    ax3.plot(t, freq, c='r')

    if freq[-1] is np.nan:
        ax3.set_title('Initializing...')
    else:
        ax3.set_title('{0:#.4} Hz'.format(freq[-1]))

    plt.pause(1./fps)
