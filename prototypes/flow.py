import cv2
import time
from pyqtgraph.Qt import QtGui
import numpy as np
import pyqtgraph as pg
from scipy.signal import butter, filtfilt
import copy


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # noinspection PyTupleAssignmentBalance
    _b, _a = butter(order, normal_cutoff, btype='low', analog=False)
    return _b, _a


def butter_lowpass_filter(_data, cutoff, fs, order=5):
    _b, _a = butter_lowpass(cutoff, fs, order=order)
    _y = filtfilt(_b, _a, _data)
    return _y


'''Plotting'''
app = QtGui.QApplication([])

win = pg.GraphicsWindow(title="Basic plotting examples")
win.resize(1000, 600)
win.setWindowTitle('pyqtgraph example: Plotting')

image_view = win.addViewBox()
image_view.setAspectLocked(True)
capture_image = pg.ImageItem(title='Capture Image', border='w')
image_view.addItem(capture_image)
pg.setConfigOptions(antialias=True)
plot = win.addPlot(title="Raw Signal")
motion_plot = win.addPlot(title="Motion Signal")
plot.showGrid(x=True, y=True)
plot.enableAutoRange('xy', True)
plot.setAspectLocked(True)
raw_signal = None
motion_signal = motion_plot.plot(pen='y')
motion_plot.enableAutoRange('xy', False)
motion_plot.setXRange(-2, 2, padding=0)
motion_plot.setYRange(-4, 4, padding=0)
motion_plot.setAspectLocked(True)
motion_scatter = motion_plot.plot([0], [0], pen=None, symbol='o')
eigen_line = motion_plot.plot(pen='r')

pca_plot = win.addPlot(title="Priciple Component of Motion")
motion_component = pca_plot.plot(pen='y')

# Create some random colors
color = np.random.randint(0, 255, (100, 3))

'''File I/O'''
directory = r"C:\Users\kevin\Desktop\Active Projects\Video Magnification Videos\\"
# noinspection PyArgumentList
cap = cv2.VideoCapture(directory + "trooper.mp4.avi")
fps = int(cap.get(cv2.CAP_PROP_FPS))

'''Motion Features'''
# params for ShiTomasi corner detection
feature_params = dict(maxCorners=100,  # Hyperparameter
                      qualityLevel=0.3,  # Hyperparameter
                      minDistance=7,  # Hyperparameter
                      blockSize=7)  # Hyperparameter
# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),  # Hyperparameter
                 maxLevel=2,  # Hyperparameter
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))  # Hyperparameter(s)
filter_band = (0.1, 1.0)  # Hyperparameter(s)

'''Get Features'''
# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
p00 = np.array([p[0] for p in p0])
'''Create Data Buffers'''
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
data = []
motion_data = []
timing = []
line_vec = None
filtered_data = None
while cap.isOpened():
    '''Acquire Image'''
    t0 = time.time()
    # noinspection PyRedeclaration
    ret, frame = cap.read()
    if frame is None or frame is False:
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    '''Optical Flow'''
    start_time = time.time()
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]
    '''Restructure Data if Feature Points Change'''
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
    if len(good_new) == 0 or len(good_old) == 0:
        break
    motion_data.append(list(np.mean(good_old - good_new, axis=0)))
    '''PCA
        1. Find the first eigenvector and transform the points along that, getting only the primary motion component.
        2. Transform the feature points into the first component dimension.
        3. Lowpass filter the motion 
    '''
    if len(motion_data) >= 2:
        x, y = np.transpose(motion_data)
        motion_scatter.setData(x, y)
        coords = np.vstack([x, y])
        cov_mat = np.cov(coords)
        eig_vals, eig_vecs = np.linalg.eig(cov_mat)
        # eig_vals, eig_vecs = zip(*list(sorted(zip(eig_vals, eig_vecs), key=lambda k: k[0], reverse=True)))
        sort_indices = np.argsort(eig_vals)[::-1]
        evec1, evec2 = eig_vecs[:, sort_indices]
        x_v1, y_v1 = evec1  # Eigenvector with largest eigenvalue
        x_v2, y_v2 = evec2
        line_vec = np.array([[0, 0], evec1])
        reduced_data = np.array(motion_data).dot(evec1)
        try:
            filtered_data = butter_lowpass_filter(reduced_data, *filter_band)
        except ValueError:
            filtered_data = copy.deepcopy(reduced_data)
    timing.append(time.time()-start_time)
    '''Visualize'''
    if line_vec is not None:
        eigen_line.setData(*np.transpose(line_vec))
    if filtered_data is not None:
        motion_component.setData(filtered_data)
    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        # frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
    img = cv2.add(frame, mask)
    img = np.transpose(img, axes=(1, 0, 2))
    capture_image.setImage(img)
    motion_signal.setData(*np.transpose(motion_data))
    for idx, point in enumerate(good_new):
        data[idx].append(point)
        raw_signal[idx].setData(*np.transpose(np.array(data[idx])))
    pg.QtGui.QApplication.processEvents()

    '''Dynamically Sleep to Maintain Frame Rate'''
    elapsed = time.time()-t0
    if elapsed < (1.0/fps):
        time.sleep((1.0/fps) - elapsed)

cv2.destroyAllWindows()
cap.release()
print('Average Compute Time = {0}s'.format(np.mean(timing)))
