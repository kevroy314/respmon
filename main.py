import os
import cv2
import pywt
import time
import copy
import logging
import numpy as np
from tqdm import tqdm
from collections import deque
from base import eulerian_magnification_bandpass
from transforms import uint8_to_float, float_to_uint8
import scipy.signal as signal
import scipy.stats as stats
from scipy.optimize import least_squares
import peakutils
import parabolic

# TODO: Serious peak detection issue causes constant estimation of respiration rates within the same freq band
# TODO: Hyperparameter optimization for stability (pull data from web cams)


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


def detect_errors(data, confidence, fit_error, fit_threshold):
    if len(data) >= 1 and len(confidence) >= 2:
        low, high = sorted(confidence[-2])
        if data[-1] < low or data[-1] > high:
            return True

    if np.abs(fit_error[-1]) > fit_threshold:
        return True

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
    guess_std = max(_data) - min(_data)
    guess_freq = est_freq = 1

    cost = np.nan
    if len(_data) > min_least_squares_length:
        optimize_func = lambda x: guess_std * np.cos(x[0] * np.array(_t)) + guess_mean - np.array(_data)
        res = least_squares(optimize_func, x0=[guess_freq],
                            bounds=([0], [np.inf]))
        est_freq = res.x
        cost = res.cost

    return guess_std, np.pi/2.0, guess_mean, est_freq, cost


def get_confidence_intervals(sample, value, ci):
    std = np.mean(sample)
    z_critical = stats.norm.ppf(q=1.0-((1.0-ci)/2.0))
    margin_of_error = z_critical * (std / np.sqrt(len(sample)))
    return value - margin_of_error, value + margin_of_error


def nomrmalize(data):
    return (data - min(data))/(max(data)-min(data))


def wavelet_analysis(data):
    from wavelets import plot_signal_decomp
    import matplotlib.pyplot as plt
    plot_signal_decomp(data, 'db4', "db4 Wavelet Decomposition")
    plt.show()


def wavelet_filter(data, w='db4', iterations=5):
    w = pywt.Wavelet(w)
    a = data
    ca = []
    cd = []
    for i in range(iterations):
        (a, d) = pywt.dwt(a, w, pywt.Modes.smooth)
        ca.append(a)
        cd.append(d)

    rec_a = []

    for i, coeff in enumerate(ca):
        coeff_list = [coeff, None] + [None] * i
        rec_a.append(pywt.waverec(coeff_list, w))
    return rec_a[-1]


def main(capture_target=0, save_calibration_image=False, visualize='pyqtgraph', fig_size=None, fps_limit=10,
         error_reset_delay=10.0, show_error_signal=False):
    assert isinstance(fps_limit, (int, float)), "fps_limit must be int or float"
    assert isinstance(save_calibration_image, bool), "save_calibration_image must be bool"
    assert visualize == 'pyqtgraph' or visualize is None, \
        "visualize must be 'pyqtgraph' or None"
    assert fig_size is None or (isinstance(fig_size, (tuple, list)) and len(fig_size) == 2), \
        "fig_size should be None or length 2 tuple or list"
    assert isinstance(error_reset_delay, (int, float)) and error_reset_delay >= 0, \
        "error_reset_delay must be a positive int or float"

    # Initialize Capture
    # noinspection PyArgumentList
    cap = cv2.VideoCapture(capture_target)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps == 0:
        fps = np.nan

    maximum_bounding_box_area = np.inf  # Adjust to speed up program if the bounding box areas are ending up too large
    min_peak_index_frequency = 0.5  # Limits peak frequency to 2Hz # Hyperparameter

    # Calibration Variables
    calibration_buffer_target_length = 128  # Hyperparameter
    calibration_buffer = np.zeros((calibration_buffer_target_length, height, width), dtype=float)
    calibration_buffer_idx = 0
    freq_min = 0.1  # Hyperparameter
    freq_max = 1.0  # Hyperparameter
    temporal_threshold = 0.7  # Hyperparameter
    threshold = 20  # Hyperparameter

    # Target Bounding Box
    # noinspection PyUnusedLocal
    x, y, w, h = None, None, None, None

    # Measurement Variables
    measure_buffer_length = 128  # Hyperparameter
    measure_initialization_length = 32  # Hyperparameter
    max_peak_detection_threshold_proportion = 0.2  # Hyperparameter
    confidence_interval = 0.95  # Hyperparameter
    fit_error_threshold = 5.0  # Hyperparameter

    data = deque()
    t = deque()
    freq = deque()
    confidence = deque()
    fit_error = deque()
    fit_error_mean = deque()
    fitted_data = []

    if visualize == 'pyqtgraph':
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
        curve4 = p0.plot(pen='w')
        curve5 = p0.plot(pen='w')
        curve6 = pg.FillBetweenItem(curve4, curve5, (255, 0, 0, 100))
        p0.addItem(curve6)

        curve7 = p0.plot(pen='g')

        curve8 = p0.plot(pen='r')

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

        curves = {"raw_signal": curve0, "capture_image": img, "frequency_plot": curve2, "peak_plot": curve3,
                  "bpm_text": text, "top_confidence_interval": curve4, "bottom_confidence_interval": curve5,
                  "fill_confidence_interval": curve6, "fitted_plot": curve7, "error_plot": curve8}

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
    calibration_progress_bar = tqdm(total=calibration_buffer_target_length)
    while cap.isOpened():
        loop_start_time = time.time()

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

                if visualize == 'pyqtgraph':
                    # noinspection PyUnboundLocalVariable
                    win.setWindowTitle(
                        'Capturing calibration frames... {0}/{1}'.format(calibration_buffer_idx,
                                                                         calibration_buffer_target_length))
                    curves["capture_image"].setImage(frame)

                # Fill frame buffer
                calibration_buffer[calibration_buffer_idx][:] = frame
                calibration_buffer_idx += 1
                # Update the progress bar
                calibration_progress_bar.update(1)
            # Once enough images have been acquired, the locate function is run to find the ROI
            else:
                if visualize == 'pyqtgraph':
                    # noinspection PyUnboundLocalVariable
                    win.setWindowTitle('Calibrating (may take several seconds)...')
                logging.info("Finished capturing calibration frames. Beginning calibration...")
                t1 = time.time()
                if fps == 0 or fps is np.nan:
                    fps = calibration_buffer_target_length / (t1 - t0)
                    logging.info("Computer FPS as {0}.".format(fps))
                if fps > fps_limit:
                    fps = fps_limit
                min_peak_index_distance = int(np.round(fps / min_peak_index_frequency))  # Limits peak frequency to 2Hz
                location = locate(calibration_buffer, fps, save_calibration_image=save_calibration_image,
                                  freq_min=freq_min, freq_max=freq_max,
                                  temporal_threshold=temporal_threshold, threshold=threshold)
                if location is None:
                    logging.info("Failed finding ROI during calibration. Retrying...")
                    calibration_buffer_idx = 0
                    continue
                x, y, w, h = location
                x, y, w, h = reduce_bounding_box(x, y, w, h, maximum_bounding_box_area)
                logging.info("Finished calibration.")
                logging.info("Beginning measuring...")
                calibration_progress_bar.update(1)
                calibration_progress_bar.close()
                if visualize == 'pyqtgraph':
                    win.setWindowTitle('Measuring...')
                state = 'measure'
        elif state == 'measure':
            # Crop to the bounding box
            crop_img = frame[y: y + h, x: x + w]
            # Check for full buffer and popleft
            buffers = [data, confidence, t, freq, fit_error, fit_error_mean]
            for b in buffers:
                if len(b) > measure_buffer_length:
                    b.popleft()
            # Average the cropped image pixels
            avg = np.average(crop_img)

            # Fill the measurement buffers
            data.append(avg)
            if len(t) == 0:
                t.append(0.)
            else:
                t.append(t[-1] + (1. / fps))

            if len(data) > measure_initialization_length:
                # noinspection PyUnboundLocalVariable
                p2.enableAutoRange('xy', True)
                # noinspection PyUnboundLocalVariable
                p0.enableAutoRange('xy', True)

                # Generate confidence intervals
                confidence.append(get_confidence_intervals(data, data[-1], confidence_interval))

                # Peak detection
                # noinspection PyUnboundLocalVariable
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

                if visualize == 'pyqtgraph':
                    if len(peak_times) > 0:
                        curves["peak_plot"].setData(peak_times, np.take(data, indices))

                freq.append(est_freq)
                if len(indices) > 0:
                    start_idx = np.clip(indices[-1] - int(measure_initialization_length/2.), 0, len(data))
                    end_idx = np.clip(indices[-1] + int(measure_initialization_length/2.), 0, len(data))
                    fitted_t = np.array(t)[start_idx:end_idx]
                    data_segment = np.array(data)[start_idx:end_idx]
                    est_std, est_phase, est_mean, est_freq, cost = fit_sine(data_segment, fitted_t)
                    fitted_data = est_std * np.sin(fitted_t * est_freq + est_phase) + est_mean

                    ssr = np.sum(np.power(np.subtract(data_segment, fitted_data), 2.0))
                    sst = np.sum(np.power(np.subtract(data_segment, np.mean(data_segment)), 2))
                    r2 = 1 - (ssr / sst)
                    fit_error.append(r2)
                    fit_error_mean.append(np.mean(np.array(fit_error)))

                if detect_errors(data, confidence, fit_error_mean, fit_error_threshold):
                    state = 'calibration'
                    calibration_buffer_idx = 0
                    data.clear()
                    confidence.clear()
                    freq.clear()
                    t.clear()
                    fit_error.clear()
                    fit_error_mean.clear()
                    for key in curves:
                        if callable(getattr(curves[key], "clear", None)):
                            curves[key].clear()
                    win.setWindowTitle('Error detected, waiting {0}s before recalibrating...'.format(error_reset_delay))
                    time.sleep(error_reset_delay)

            if visualize == 'pyqtgraph':
                if len(data) >= 2 and len(t) >= 2:
                    curves["raw_signal"].setData(t, data)
                if len(fitted_data) >= 2 and len(fitted_t) >= 2:
                    # noinspection PyUnboundLocalVariable
                    curves["fitted_plot"].setData(fitted_t, fitted_data)
                fit_error_t = np.array(t)[-len(fit_error):]
                if show_error_signal and len(fit_error) >= 2 and len(fit_error_t) >= 2:
                    curves["error_plot"].setData(fit_error_t, np.abs(np.array(fit_error_mean)/fit_error_threshold))
                curves["capture_image"].setImage(crop_img)
                if len(freq) >= 2 and len(t) >= 2:
                    curves["frequency_plot"].setData(np.array(t)[-len(freq):], freq)
                    curves["bpm_text"].setText('{0:#.4} BPM'.format(freq[-1]))
                if len(confidence) >= 2:
                    ci_top, ci_bottom = np.transpose(confidence)
                    ci_t = np.array(t)[-len(ci_top):]
                    curves["top_confidence_interval"].setData(ci_t, ci_top)
                    curves["bottom_confidence_interval"].setData(ci_t, ci_bottom)

        if visualize == 'pyqtgraph':
            # noinspection PyUnboundLocalVariable
            pg.QtGui.QApplication.processEvents()

        fps_x = fps
        if fps is np.nan:
            fps_x = fps_limit
        sleep_time = (1.0 / fps_x) - (time.time() - loop_start_time)
        if sleep_time > 0:
            time.sleep(sleep_time)

    logging.info("Capture closed.")


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s :: %(message)s", level=logging.INFO)
    # noinspection PyTypeChecker
    # main(capture_target=r"C:\Users\kevin\Desktop\Video Magnification Videos\timber1.mp4", save_calibration_image=True)
    # main(capture_target=r"C:\Users\kevin\Desktop\Video Magnification Videos\timber2.mp4",
    #      save_calibration_image=True, fps_limit=100)
    main(capture_target=0, save_calibration_image=True)
