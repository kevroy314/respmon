import os
import cv2
import time
import copy
import logging
import numpy as np
import pyqtgraph as pg
from tqdm import tqdm
import scipy.stats as stats
from collections import deque
from pyqtgraph.Qt import QtGui
from prototypes.detect_peaks import detect_peaks
from prototypes.signal_measurement import find_peaks_simplified
from scipy.optimize import least_squares
from transforms import uint8_to_float, float_to_uint8, temporal_bandpass_filter_fft
from pyramid import create_laplacian_video_pyramid, collapse_laplacian_video_pyramid


def eulerian_magnification_bandpass(vid_data, fps, freq_min, freq_max, amplification,
                                    pyramid_levels=4, skip_levels_at_top=2, verbose=False,
                                    temporal_filter_function=temporal_bandpass_filter_fft, threshold=0.7):
    t0 = time.time()
    vid_pyramid = create_laplacian_video_pyramid(vid_data, pyramid_levels=pyramid_levels)
    t1 = time.time()
    bandpassed_pyramid = []
    for i0 in range(0, len(vid_pyramid)):
        bandpassed_pyramid.append(np.zeros(vid_pyramid[i0].shape))
    if verbose:
        print("{2} (t={0}, dt={1})".format(t0, t1-t0, "create_laplacian_video_pyramid"))
        print("{2} (t={0}, dt={1})".format('n/a', (t1-t0)/float(len(vid_data)), "Frame Average"))
    for i, vid in enumerate(vid_pyramid):
        if i < skip_levels_at_top or i >= len(vid_pyramid) - 1:
            # ignore the top and bottom of the pyramid. One end has too much noise and the other end is the
            # gaussian representation
            continue
        t0 = time.time()
        bandpassed = temporal_filter_function(vid, fps, freq_min=freq_min, freq_max=freq_max,
                                              amplification_factor=amplification,
                                              debug='{0},{1}:'.format('n/a', i), verbose=verbose)
        t1 = time.time()
        if verbose:
            print("{2} (t={0}, dt={1})".format(t0, t1-t0, "temporal_bandpass_filter"))
            print("{2} (t={0}, dt={1})".format('n/a', (t1-t0)/float(len(vid_data)), "Frame Average"))
        # play_vid_data(bandpassed)
        bandpassed_pyramid[i] += bandpassed
        vid_pyramid[i] += bandpassed
        # play_vid_data(vid_pyramid[i])
    t0 = time.time()
    vid_data = collapse_laplacian_video_pyramid(vid_pyramid)
    bandpassed_data = collapse_laplacian_video_pyramid(bandpassed_pyramid)

    window_proportional_width = threshold
    min_val = bandpassed_data.min()
    replace_value = min_val
    max_val = bandpassed_data.max()
    intensity_filter_width = (max_val - min_val) * window_proportional_width
    top = max_val - intensity_filter_width
    # bottom = min_val + intensity_filter_width
    # mask = np.logical_or(bandpassed_data >= top, bandpassed_data <= bottom)
    mask = bandpassed_data >= top
    bandpassed_data[mask] = replace_value
    t1 = time.time()
    if verbose:
        print('min={0}, max={1}'.format(np.array(vid_data).min(), np.array(vid_data).max()))
        print("{2} (t={0}, dt={1})".format(t0, t1-t0, "collapse_laplacian_video_pyramid"))
        print("{2} (t={0}, dt={1})".format('n/a', (t1-t0)/float(len(vid_data)), "Frame Average"))
    return bandpassed_data


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


# TODO: Serious peak detection issue causes constant estimation of respiration rates within the same freq band
# TODO: Hyperparameter optimization for stability (pull data from web cams)

class RespiratoryMonitor:
    def __init__(self, capture_target=0, save_calibration_image=False, visualize='pyqtgraph', fig_size=None,
                 fps_limit=10, error_reset_delay=10.0, show_error_signal=False, save_all_data=True):
        self.data = []
        assert isinstance(fps_limit, (int, float)) and fps_limit > 0, "fps_limit must be a positive int or float"
        assert isinstance(save_calibration_image, bool), "save_calibration_image must be bool"
        assert visualize == 'pyqtgraph' or visualize is None, \
            "visualize must be 'pyqtgraph' or None"
        assert fig_size is None or (isinstance(fig_size, (tuple, list)) and len(fig_size) == 2), \
            "fig_size should be None or length 2 tuple or list"
        assert isinstance(error_reset_delay, (int, float)) and error_reset_delay >= 0, \
            "error_reset_delay must be a positive int or float"
        assert isinstance(show_error_signal, bool), "show_error_signal should be bool"
        assert isinstance(save_all_data, bool), "save_all_data should be bool"

        self.error_reset_delay = error_reset_delay
        self.show_error_signal = show_error_signal
        self.save_all_data = save_all_data
        self.fig_size = fig_size
        self.save_calibration_image = save_calibration_image
        self.capture_target = capture_target
        self.visualize = visualize

        # Initialize Capture
        # noinspection PyArgumentList
        self.cap = cv2.VideoCapture(capture_target)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.maximum_bounding_box_area = np.inf  # Hyperparameter; If the bounding box is too large
        self.min_peak_index_frequency = 0.1  # Hyperparameter; Limits peak frequency to 2Hz

        # Calibration Variables
        self.calibration_buffer_target_length = 128  # Hyperparameter; The number of frames to use for calibration
        self.freq_min = 0.1  # Hyperparameter; Minimum frequency to look for during calibration
        self.freq_max = 1.0  # Hyperparameter; Maximum frequency to look for during calibration
        self.temporal_threshold = 0.7  # Hyperparameter; The strength of temporal information to eliminate as noise
        self.threshold = 20  # Hyperparameter; The threshold of the temporal locator image to use for isolating a box

        # Measurement Variables
        self.measure_buffer_length = 128  # Hyperparameter; The buffer length for the measurement stream
        self.measure_initialization_length = 32  # Hyperparameter; The number of frames to wait for to being measuring
        self.max_peak_detection_threshold_proportion = 0.  # Hyperparameter; The maximum peak height threshold
        self.confidence_interval = 0.99  # Hyperparameter; The confidence interval for large motion
        self.fit_error_threshold = 5.0  # Hyperparameter; The fit threshold for shape errors

        # If no FPS is provided, set it to NaN to inform downstream checks that it isn't a valid FPS
        if self.fps == 0:
            self.fps = np.nan
            self.min_peak_index_distance = np.nan
        else:
            self.min_peak_index_distance = int(np.round(self.fps / self.min_peak_index_frequency))
        self.fps_limit = fps_limit

        # Target Bounding Box
        self.x, self.y, self.w, self.h = None, None, None, None

        self.enable_error_detection = False

        self.calibration_buffer_idx = 0
        self.calibration_buffer = np.zeros((self.calibration_buffer_target_length, self.height, self.width),
                                           dtype=float)
        self.all_data = []
        self.data = deque()
        self.t = deque()
        self.freq = deque()
        self.confidence = deque()
        self.num_peaks = deque()
        self.num_peaks_mean = deque()
        self.fitted_data = []
        self.fitted_t = []

        self.peak_indices = []
        self.peak_times = []

        self.current_frame = None
        self.cropped_image = None

        self.buffers = [self.data, self.confidence, self.t, self.freq, self.num_peaks, self.num_peaks_mean]

        if visualize == 'pyqtgraph':
            self.ui = self.initialize_pyqtgraph_visualization()

        # State Machine State
        self.state = 'initialize'

        '''Hack to avoid calibration for testing'''
        '''self.skip_calibration(538, 243, 70, 51)'''
        ''''''''''''''''''''''''''''''''''''''''''''

        logging.info("Capturing {0} calibration frames.".format(self.calibration_buffer_target_length))
        self.calibration_start_time = np.nan
        self.loop_start_time = np.nan
        self.reset_start_time = np.nan

        # Begin progress bar for calibration
        self.calibration_progress_bar = tqdm(total=self.calibration_buffer_target_length)

        self.run()

    def skip_calibration(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.state = 'measure'

    def initialize_pyqtgraph_visualization(self):
        app = QtGui.QApplication([])

        win = pg.GraphicsWindow(title="Respiration Monitor")
        if self.fig_size is None:
            win.resize(1500, 900)
        else:
            win.resize(*self.fig_size)
        layout = pg.GraphicsLayout()
        win.addItem(layout)

        pg.setConfigOptions(antialias=True)

        left_plot = layout.addPlot(title="Raw Signal")
        left_plot.showGrid(x=True, y=True)
        left_plot.enableAutoRange('xy', False)

        raw_signal = left_plot.plot(pen='y')
        peak_plot = left_plot.plot(pen=None, symbolBrush=(255, 0, 0), symbolPen=None)
        top_confidence_interval = left_plot.plot(pen='w')
        bottom_confidence_interval = left_plot.plot(pen='w')
        fill_confidence_interval = pg.FillBetweenItem(top_confidence_interval, bottom_confidence_interval,
                                                      (255, 0, 0, 100))
        left_plot.addItem(fill_confidence_interval)

        fitted_plot = left_plot.plot(pen='g')

        error_plot = left_plot.plot(pen='r')

        image_view = layout.addViewBox()
        image_view.setAspectLocked(True)
        capture_image = pg.ImageItem(title='Capture Image', border='w')
        image_view.addItem(capture_image)

        right_plot = layout.addPlot(title="Frequency Plot (bpm)")
        right_plot.showGrid(x=True, y=True)
        right_plot.enableAutoRange('xy', False)

        frequency_plot = right_plot.plot()

        bpm_text = pg.TextItem(text='??? BPM', anchor=(-0.1, 1.2), color=(255, 255, 255, 255))
        bpm_text_font = pg.QtGui.QFont()
        bpm_text_font.setBold(True)
        bpm_text_font.setPointSize(24)
        bpm_text.setFont(bpm_text_font)
        right_plot.addItem(bpm_text)
        bpm_text.setPos(0, 0)

        return {"raw_signal": raw_signal, "capture_image": capture_image, "frequency_plot": frequency_plot,
                "peak_plot": peak_plot, "bpm_text": bpm_text, "top_confidence_interval": top_confidence_interval,
                "bottom_confidence_interval": bottom_confidence_interval,
                "fill_confidence_interval": fill_confidence_interval, "fitted_plot": fitted_plot,
                "error_plot": error_plot, "window": win, "plots": [left_plot, right_plot], "application": app}

    def next_frame(self):
        ret, frame = self.cap.read()
        if frame is not None and frame is not False:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return uint8_to_float(gray)
        else:
            return False

    def set_window_title(self, title):
        self.ui["window"].setWindowTitle(title)

    def set_image(self, img):
        self.ui["capture_image"].setImage(img)

    def set_plot_autoscale(self, autoscale_enabled, axes='xy'):
        for plot in self.ui["plots"]:
            plot.enableAutoRange(axes, autoscale_enabled)

    def update_ui(self):
        if self.visualize == 'pyqtgraph':
            if self.state == "calibration":
                if self.calibration_buffer_idx < self.calibration_buffer_target_length:
                    self.set_window_title(
                        'Capturing calibration frames... {0}/{1}'.format(self.calibration_buffer_idx,
                                                                         self.calibration_buffer_target_length))
                    self.set_image(self.current_frame)
                else:
                    self.set_window_title('Measuring...')
                    self.set_plot_autoscale(True)
            elif self.state == "measure":
                self.set_window_title('Building Measurement Buffer.'+'.'.join(['' for _ in range(0, len(self.data) % 4)]))
                if len(self.peak_times) > 0:
                    self.ui["peak_plot"].setData(self.peak_times, np.take(self.data, self.peak_indices))
                self.set_window_title(
                    'Measuring.' + '.'.join(['' for _ in range(0, len(self.data) % 4)]))
                if len(self.data) >= 2 and len(self.t) >= 2:
                    self.ui["raw_signal"].setData(self.t, self.data)
                if len(self.fitted_data) >= 2 and len(self.fitted_t) >= 2:
                    self.ui["fitted_plot"].setData(self.fitted_t, self.fitted_data)
                fit_error_t = np.array(self.t)[-len(self.num_peaks):]
                if self.show_error_signal and len(self.num_peaks) >= 2 and len(fit_error_t) >= 2:
                    self.ui["error_plot"].setData(fit_error_t, np.abs(
                        np.array(self.num_peaks_mean) / self.fit_error_threshold))
                self.ui["capture_image"].setImage(self.cropped_image)
                if len(self.freq) >= 2 and len(self.t) >= 2:
                    self.ui["frequency_plot"].setData(np.array(self.t)[-len(self.freq):], self.freq)
                    self.ui["bpm_text"].setText('{0:#.4} BPM'.format(self.freq[-1]))
                if len(self.confidence) >= 2:
                    ci_top, ci_bottom = np.transpose(self.confidence)
                    ci_t = np.array(self.t)[-len(ci_top):]
                    self.ui["top_confidence_interval"].setData(ci_t, ci_top)
                    self.ui["bottom_confidence_interval"].setData(ci_t, ci_bottom)
            elif self.state == "error":
                self.set_window_title(
                    "Error: Recalibrating due to poor signal in {0}s.".format(
                        self.error_reset_delay - (time.time()-self.reset_start_time)))

            pg.QtGui.QApplication.processEvents()

    def run(self):
        while self.cap.isOpened():
            self.loop_start_time = time.time()

            # Capture the frame (quit if the frame is a bool, meaning end of stream)
            self.current_frame = self.next_frame()
            if isinstance(self.current_frame, bool):
                break

            if self.state == 'initialize':
                self.calibration_start_time = time.time()
                self.calibration_buffer_idx = 0
                self.state = 'calibration'
            # Calibration phase
            elif self.state == 'calibration':
                # The beginning of the calibration phase is just acquiring enough images to calibrate
                if self.calibration_buffer_idx < self.calibration_buffer_target_length:
                    # Fill frame buffer
                    self.calibration_buffer[self.calibration_buffer_idx][:] = self.current_frame
                    self.calibration_buffer_idx += 1
                    # Update the progress bar
                    self.calibration_progress_bar.update(1)
                # Once enough images have been acquired, the locate function is run to find the ROI
                else:
                    logging.info("Finished capturing calibration frames. Beginning calibration...")
                    if self.fps == 0 or self.fps is np.nan:
                        self.fps = self.calibration_buffer_target_length / (time.time() - self.calibration_start_time)
                        logging.info("Computer FPS as {0}.".format(self.fps))
                    if self.fps > self.fps_limit:
                        self.fps = self.fps_limit
                    # Limits peak frequency to 2Hz
                    self.min_peak_index_distance = int(np.round(self.fps / self.min_peak_index_frequency))
                    location = self.locate(self.calibration_buffer, self.fps,
                                           save_calibration_image=self.save_calibration_image,
                                           freq_min=self.freq_min, freq_max=self.freq_max,
                                           temporal_threshold=self.temporal_threshold, threshold=self.threshold)
                    if location is None:
                        logging.info("Failed finding ROI during calibration. Retrying...")
                        self.calibration_buffer_idx = 0
                        continue
                    self.x, self.y, self.w, self.h = location
                    self.x, self.y, self.w, self.h = reduce_bounding_box(self.x, self.y, self.w, self.h,
                                                                         self.maximum_bounding_box_area)
                    logging.info("Finished calibration.")
                    logging.info("Beginning measuring...")
                    self.calibration_progress_bar.update(1)
                    self.calibration_progress_bar.close()

                    self.state = 'measure'
            elif self.state == 'measure':
                # Crop to the bounding box
                self.cropped_image = self.current_frame[self.y: self.y + self.h, self.x: self.x + self.w]
                # Check for full buffer and popleft
                for b in self.buffers:
                    if len(b) > self.measure_buffer_length:
                        b.popleft()
                # Average the cropped image pixels
                avg = np.average(self.cropped_image)
                # Fill the measurement buffers
                self.data.append(avg)
                if len(self.t) == 0:
                    self.t.append(0.)
                else:
                    self.t.append(self.t[-1] + (1. / self.fps))
                if self.save_all_data:
                    self.all_data.append((self.t[-1], avg))
                if len(self.data) > self.measure_initialization_length:
                    # Generate confidence intervals
                    self.confidence.append(get_confidence_intervals(self.data, self.data[-1], self.confidence_interval))
                    # Peak detection
                    self.peak_indices, fits = find_peaks_simplified(self.data, self.t, fs=self.fps)

                    self.peak_times = np.take(self.t, self.peak_indices)
                    diffs = [a - b for b, a in zip(self.peak_indices, self.peak_indices[1:])]
                    if len(diffs) == 0:
                        interval = np.nan
                    else:
                        interval = np.mean(diffs)
                    est_freq = 1.0 / interval * 60.0  # 60 seconds in a minute

                    self.freq.append(est_freq)
                    if len(self.peak_indices) > 0:
                        self.num_peaks.append(len(self.peak_indices))
                        self.num_peaks_mean.append(np.std(np.array(self.num_peaks)))

                    if self.enable_error_detection and len(self.data) > 0 and len(self.confidence) > 0 and \
                            len(self.num_peaks_mean) > 0 and self.detect_errors():
                        self.state = 'error'
                        self.reset_start_time = time.time()
            elif self.state == 'error':
                if time.time() - self.reset_start_time >= self.error_reset_delay:
                    self.reset()
                    self.state = 'calibration'

            # Update the UI once the internal state has been set (will do nothing if visualize is None)
            self.update_ui()

            self.sync_to_fps()

        logging.info("Capture closed.")

        if self.save_all_data:
            np.save(str(self.capture_target) + '.npy', self.all_data)

    def reset(self):
        self.state = 'initialize'
        self.data.clear()
        self.confidence.clear()
        self.freq.clear()
        self.t.clear()
        self.num_peaks.clear()
        self.num_peaks_mean.clear()
        for key in self.ui:
            if callable(getattr(self.ui[key], "clear", None)):
                self.ui[key].clear()

    def sync_to_fps(self):
        fps_x = self.fps
        if self.fps is np.nan:
            fps_x = self.fps_limit
        sleep_time = (1.0 / fps_x) - (time.time() - self.loop_start_time)
        if sleep_time > 0:
            time.sleep(sleep_time)

    def detect_errors(self):
        if len(self.data) >= 1 and len(self.confidence) >= 2:
            low, high = sorted(self.confidence[-2])
            if self.data[-1] < low or self.data[-1] > high:
                return True

        if np.abs(self.num_peaks[-1]) > self.fit_error_threshold:
            return True

        return False

    @staticmethod
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
