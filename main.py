import os
import cv2
import time
import copy
import logging
import numpy as np
from tqdm import tqdm
from collections import deque
from base import eulerian_magnification_bandpass
from transforms import uint8_to_float, float_to_uint8
import scipy.signal as signal
from scipy.optimize import least_squares
import peakutils
import parabolic

# TODO: Large motion error detection (hard threshold?)
# TODO: Hyperparameter optimization for stability (pull data from youtube)


def freq_from_fft(sig, fs):
    """
    Estimate frequency from peak of FFT
    """
    # Compute Fourier transform of windowed signal
    windowed = sig * signal.blackmanharris(len(sig))
    f = np.fft.rfft(windowed)

    # Find the peak and interpolate to get a more accurate peak
    i = np.argmax(abs(f))  # Just use this for less-accurate, naive version
    true_i = parabolic.parabolic(np.log(abs(f)), i)[0]

    # Convert to equivalent frequency
    return fs * true_i / len(windowed)


def locate(calibration_video_data, fps,
           freq_min=0.1, freq_max=1.0, amplification=500,
           pyramid_levels=9, skip_levels_at_top=4, temporal_threshold=0.7,
           threshold=20, threshold_type=cv2.THRESH_BINARY,
           verbose=False, save_calibration_image=False):
    logging.info("Beginning processing calibration frames...")
    # Perform motion extraction
    op = eulerian_magnification_bandpass(calibration_video_data, fps, freq_min, freq_max, amplification,
                                         skip_levels_at_top=skip_levels_at_top, pyramid_levels=pyramid_levels,
                                         threshold=temporal_threshold,
                                         verbose=verbose)
    logging.info("Done processing calibration frames.")
    # Generate normed average frame (0-255 grayscale)
    logging.info("Finding peak region...")
    avg_frame = np.array(np.average(op, axis=0))
    avg_norm = ((avg_frame - avg_frame.min()) / (avg_frame.max() - avg_frame.min()))
    avg = float_to_uint8(avg_norm)
    # Find largest region
    ret, thresh = cv2.threshold(avg, threshold, 255, threshold_type)  # Threshold image
    thresh_copy = copy.deepcopy(thresh)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
    if len(contours) <= 0:
        return None
    c = max(contours, key=cv2.contourArea)  # Find max contours
    area = cv2.contourArea(c)
    logging.info("Found peak region.")
    # Get bounding box
    x, y, w, h = cv2.boundingRect(c)

    if save_calibration_image:
        logging.info('Creating calibration image.')
        total_avg = float_to_uint8(np.average(calibration_video_data, axis=0))
        contour_img = copy.deepcopy(total_avg)
        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 3)
        avg_copy = copy.deepcopy(avg)
        drawn = cv2.rectangle(total_avg + avg_copy, (x, y), (x + w, y + h), 255, 2)
        row0 = np.hstack((contour_img, avg))
        row1 = np.hstack((thresh_copy, drawn))
        calibration = np.vstack((row0, row1))
        i = 0
        while os.path.exists("calibration%s.png" % i):
            i += 1
        cv2.imwrite(r'calibration%s.png' % i, calibration)
        logging.info('Calibration image saved.')

    if verbose:
        print('contour area:{4} - x:{0}, y:{1}, w:{2}, h:{3}'.format(x, y, w, h, area))

    return x, y, w, h


def next_frame(capture_object):
    ret, frame = capture_object.read()
    if frame is not None and frame is not False:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return uint8_to_float(gray)
    else:
        return False


def detect_errors():
    return False


def reduce_bounding_box(x, y, w, h, maximum_area):
    start_area = w*h
    if start_area <= maximum_area:
        return x, y, w, h
    shrink_proportion = np.sqrt((float(maximum_area)/float(start_area)))
    new_w = w * shrink_proportion
    new_h = h * shrink_proportion
    new_x = x + ((w - new_w) / 2.)
    new_y = y + ((h - new_h) / 2.)
    return int(np.round(new_x)), int(np.round(new_y)), int(np.round(new_w)), int(np.round(new_h))


def fit_sine(_data, _t, min_least_squares_length=10):
    guess_mean = np.mean(_data)
    guess_std = 3 * np.std(_data) / (2 ** 0.5)
    guess_phase = 0
    guess_freq = 1

    cost = np.nan
    if len(_data) > min_least_squares_length:
        optimize_func = lambda x: x[0] * np.sin(x[3] * np.array(_t) + x[1]) + x[2] - np.array(_data)
        res = least_squares(optimize_func, x0=[guess_std, guess_phase, guess_mean, guess_freq],
                            bounds=([0, 0, 0, 0], [np.inf, 2 * np.pi, np.inf, np.inf]))
        est_std, est_phase, est_mean, est_freq = res.x
        cost = res.cost
    else:
        est_std = guess_std
        est_phase = guess_phase
        est_mean = guess_mean
        est_freq = guess_freq

    return est_std, est_phase, est_mean, est_freq, cost


def main(capture_target=0, save_calibration_image=False, visualize='pyqtgraph', fig_size=None):
    # Initialize Capture
    # noinspection PyArgumentList
    cap = cv2.VideoCapture(capture_target)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps == 0:
        fps = np.nan
    limit_fps = 10

    maximum_bounding_box_area = np.inf  # Adjust to speed up program if the bounding box areas are ending up too large

    # Calibration Variables
    calibration_buffer_target_length = 128
    calibration_buffer = np.zeros((calibration_buffer_target_length, height, width), dtype=float)
    calibration_buffer_idx = 0

    # Target Bounding Box
    # noinspection PyUnusedLocal
    x, y, w, h = None, None, None, None

    # Measurement Variables
    min_peak_index_distance = int(np.round(fps / 2.))  # Limits peak frequency to 2Hz
    measure_buffer_length = 128
    measure_initialization_length = 32
    max_peak_detection_threshold_proportion = 0.2
    data = deque()
    t = deque()
    freq = deque()

    if visualize == 'matplotlib':
        import matplotlib.pyplot as plt
        # UI Variables
        if fig_size is None:
            f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 8))
        else:
            f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=fig_size)
        f.set_figheight(13)
        f.set_figwidth(21)
        ax1_2 = ax1.twinx()

        subplots = [ax1, ax1_2, ax2, ax3]
        plt.ion()
        f.show()
    elif visualize == 'pyqtgraph':
        from pyqtgraph.Qt import QtGui
        import pyqtgraph as pg

        # noinspection PyUnusedLocal
        app = QtGui.QApplication([])

        win = pg.GraphicsWindow(title="Respiration Monitor")
        if fig_size is None:
            win.resize(1500, 900)
        else:
            win.resize(*fig_size)
        layout = pg.GraphicsLayout()
        win.addItem(layout)

        pg.setConfigOptions(antialias=True)

        p0 = layout.addPlot(title="Raw Signal")
        p0.showGrid(x=True, y=True)

        curve0 = p0.plot(pen='y')
        curve3 = p0.plot(pen=None, symbolBrush=(255, 0, 0), symbolPen=None)

        view = layout.addViewBox()
        view.setAspectLocked(True)
        img = pg.ImageItem(title='Capture Image', border='w')
        view.addItem(img)

        p2 = layout.addPlot(title="Frequency Plot (bpm)")
        p2.showGrid(x=True, y=True)
        curve2 = p2.plot()
        p2.enableAutoRange('xy', False)

        text = pg.TextItem(text='??? BPM', anchor=(-0.1, 1.2), color=(255, 255, 255, 255))
        font = pg.QtGui.QFont()
        font.setBold(True)
        font.setPointSize(24)
        text.setFont(font)
        p2.addItem(text)
        text.setPos(0, 0)

        curves = [curve0, img, curve2, curve3, text]

    # State Machine State
    # noinspection PyUnusedLocal
    state = 'calibration'

    '''Hack to avoid calibration for testing'''
    # state = 'measure'
    # x = 538
    # y = 243
    # w = 70
    # h = 51
    ''''''''''''''''''''''''''''''''''''''''''''

    logging.info("Capturing {0} calibration frames.".format(calibration_buffer_target_length))
    t0 = time.time()
    # Begin progress bar for calibration
    calibration_pbar = tqdm(total=calibration_buffer_target_length)
    while cap.isOpened():
        loop_start_time = time.time()

        if visualize == 'matplotlib':
            # Clear the UI
            [ax.clear() for ax in subplots]

        # Capture the frame (quit if the frame is a bool, meaning end of stream)
        frame = next_frame(cap)
        if isinstance(frame, bool):
            break

        # Calibration phase
        if state == 'calibration':
            # The beginning of the calibration phase is just acquiring enough images to calibrate
            if calibration_buffer_idx < calibration_buffer_target_length:
                # Draw progress and frame
                y = [1] * calibration_buffer_idx + [0] * (calibration_buffer_target_length - calibration_buffer_idx)
                x = list(range(0, calibration_buffer_target_length))

                if visualize == 'matplotlib':  # Update plot for calibration
                    plt.suptitle('Capturing calibration frames...')
                    ax1.fill_between(x, 0, y)
                    ax2.imshow(frame)
                    ax3.fill_between(x, 0, list(reversed(y)))
                elif visualize == 'pyqtgraph':
                    # noinspection PyUnboundLocalVariable
                    win.setWindowTitle(
                        'Capturing calibration frames... {0}/{1}'.format(calibration_buffer_idx,
                                                                         calibration_buffer_target_length))
                    curves[1].setImage(frame)

                # Fill frame buffer
                calibration_buffer[calibration_buffer_idx][:] = frame
                calibration_buffer_idx += 1
                # Update the progress bar
                calibration_pbar.update(1)
            # Once enough images have been acquired, the locate function is run to find the ROI
            else:
                if visualize == 'matplotlib':  # Update Status
                    plt.suptitle('Calibrating (may take several seconds)...')
                elif visualize == 'pyqtgraph':
                    # noinspection PyUnboundLocalVariable
                    win.setWindowTitle('Calibrating (may take several seconds)...')
                logging.info("Finished capturing calibration frames. Beginning calibration...")
                t1 = time.time()
                if fps == 0:
                    fps = calibration_buffer_target_length / (t1 - t0)
                    logging.info("Computer FPS as {0}.".format(fps))
                if fps > limit_fps:
                    fps = limit_fps
                location = locate(calibration_buffer, fps, save_calibration_image=save_calibration_image)
                if location is None:
                    logging.info("Failed finding ROI during calibration. Retrying...")
                    calibration_buffer_idx = 0
                    continue
                x, y, w, h = location
                x, y, w, h = reduce_bounding_box(x, y, w, h, maximum_bounding_box_area)
                logging.info("Finished calibration.")
                logging.info("Beginning measuring...")
                calibration_pbar.update(1)
                calibration_pbar.close()
                if visualize == 'matplotlib':  # Update Status
                    ax1.set_title('Intensity Measurement')
                    ax3.set_title('Frequency Measurement (BPM)')
                elif visualize == 'pyqtgraph':
                    # noinspection PyUnboundLocalVariable
                    p2.enableAutoRange('xy', True)
                    win.setWindowTitle('Measuring...')
                state = 'measure'
        elif state == 'measure':
            # Crop to the bounding box
            crop_img = frame[y: y + h, x: x + w]
            # Check for full buffer and popleft
            if len(data) > measure_buffer_length:
                data.popleft()
                t.popleft()
                freq.popleft()
            # Average the cropped image pixels
            avg = np.average(crop_img)

            # Fill the measurememt buffers
            data.append(avg)
            if len(t) == 0:
                t.append(0.)
            else:
                t.append(t[-1] + (1. / fps))

            if len(data) > measure_initialization_length:
                # Sine fitting
                # est_std, est_phase, est_mean, est_freq, cost = fit_sine(data, t) # TODO: This is too slow/inaccurate
                # recreate the fitted curve using the optimized parameters
                # data_fit = est_std * np.sin(est_freq * np.array(t) + est_phase) + est_mean

                # FFT frequency analysis
                # est_freq = freq_from_fft(data, 1. / fps) * 60 * fps # TODO: This isn't working at all...

                # Peak detection
                indices = peakutils.indexes(np.array(data),
                                            thres=max_peak_detection_threshold_proportion / max(data),
                                            min_dist=min_peak_index_distance)
                peak_times = np.take(t, indices)
                diffs = [a - b for b, a in zip(peak_times, peak_times[1:])]
                if len(diffs) == 0:
                    interval = np.nan
                else:
                    interval = np.mean(diffs)
                est_freq = 1.0/interval * 60.0  # 60 seconds in a minute

                if visualize == 'matplotlib':  # Visualize peaks
                    ax1.scatter(x=peak_times, y=np.take(data, indices), c='r', marker='x')
                if visualize == 'pyqtgraph':
                    curves[3].setData(peak_times, np.take(data, indices))

                freq.append(est_freq)
            else:
                freq.append(np.nan)

            if visualize == 'matplotlib':  # Visualize the measurement
                # Fill the UI
                ax1.plot(t, data, c='b')
                # ax1_2.plot(t, data_fit, c='g')
                ax2.imshow(crop_img)
                ax3.plot(t, freq, c='r')

                if freq[-1] is np.nan:
                    plt.suptitle('Initializing...')
                else:
                    plt.suptitle('{0:#.4} BPM'.format(freq[-1]))
            elif visualize == 'pyqtgraph':
                if len(data) >= 2 and len(t) >= 2:
                    curves[0].setData(t, data)
                curves[1].setImage(crop_img)
                freq_no_nan = [x for x in freq if x is not np.nan]
                if len(freq_no_nan) >= 2 and len(t) >= 2:
                    curves[2].setData(np.array(t)[-len(freq_no_nan):], freq_no_nan)
                curves[4].setText('{0:#.4} BPM'.format(freq[-1]))

            if detect_errors():  # TODO: Finish error handler
                state = 'calibration'

        if visualize == 'matplotlib':  # Prevent always on top
            # noinspection PyUnboundLocalVariable
            f.canvas.flush_events()
        elif visualize == 'pyqtgraph':
            # noinspection PyUnboundLocalVariable
            pg.QtGui.QApplication.processEvents()

        fps_x = fps
        if fps == np.nan:
            fps_x = limit_fps
        sleep_time = (1.0 / fps_x) - (time.time() - loop_start_time)
        if sleep_time > 0:
            time.sleep(sleep_time)

    logging.info("Capture closed.")


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s :: %(message)s", level=logging.INFO)
    # noinspection PyTypeChecker
    main(capture_target=r"C:\Users\kevin\Desktop\Video Magnification Videos\timber.mp4", save_calibration_image=True)
    # main(capture_target=0, save_calibration_image=True)
