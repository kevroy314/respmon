import cv2
import time
from pyqtgraph.Qt import QtGui
import numpy as np
import pyqtgraph as pg
from scipy.signal import butter, lfilter, filtfilt
import copy

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # noinspection PyTupleAssignmentBalance
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

app = QtGui.QApplication([])

win = pg.GraphicsWindow(title="Basic plotting examples")
win.resize(1000, 600)
win.setWindowTitle('pyqtgraph example: Plotting')

# Enable antialiasing for prettier plots
pg.setConfigOptions(antialias=True)
plot = win.addPlot(title="Raw Signal")
motion_plot = win.addPlot(title="Motion Signal")
plot.showGrid(x=True, y=True)
plot.enableAutoRange('xy', True)
raw_signal = None
motion_signal = motion_plot.plot(pen='y')
motion_plot.enableAutoRange('xy', False)
motion_plot.setXRange(-2, 2, padding=0)
motion_plot.setYRange(-4, 4, padding=0)

directory = r"C:\Users\kevin\Desktop\Active Projects\Video Magnification Videos\\"
cap = cv2.VideoCapture(directory + "trooper.mp4.avi")
fps = int(cap.get(cv2.CAP_PROP_FPS))
# params for ShiTomasi corner detection
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)
# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0, 255, (100, 3))
# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
p00 = np.array([p[0] for p in p0])
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
data = []
motion_data = []
while 1:
    t0 = time.time()
    # noinspection PyRedeclaration
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]
    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        # frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
    img = cv2.add(frame, mask)
    cv2.imshow('frame', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)
    if raw_signal is None:
        raw_signal = []
        for point in good_new:
            data.append([])
            raw_signal.append(plot.plot(pen='y'))
    data = [i for s, i in zip(st, data) if s]
    raw_signal = [i for s, i in zip(st, raw_signal) if s]
    motion_data.append(list(np.mean(good_old - good_new, axis=0)))
    try:
        filtered_data = np.transpose([butter_lowpass_filter(x, 0.5, 5.0) for x in np.transpose(motion_data)])[:-1]
    except ValueError:
        filtered_data = copy.deepcopy(motion_data)
    motion_signal.setData(*np.transpose(filtered_data))
    for idx, point in enumerate(good_new):
        data[idx].append(point)
        raw_signal[idx].setData(*np.transpose(np.array(data[idx])))
    pg.QtGui.QApplication.processEvents()
    elapsed = time.time()-t0
    if elapsed < (1.0/fps):
        time.sleep((1.0/fps) - elapsed)

cv2.destroyAllWindows()
cap.release()
