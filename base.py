import os
import cv2
import time
import copy
import logging
import peakutils
import numpy as np
import pyqtgraph as pg
from tqdm import tqdm
from collections import deque
from pyqtgraph.Qt import QtGui
from tools import reduce_bounding_box
from transforms import uint8_to_float, float_to_uint8, eulerian_magnification_bandpass, butter_lowpass_filter


# TODO: Fix error tracking (large motion is handled by point tracking, but no motion is an issue now)
# TODO: Peak detection has double-peak issue?
# TODO: Hyperparameter optimization for stability (pull data from web cams)

class RespiratoryMonitor:
    def __init__(self, capture_target=0, save_calibration_image=False, visualize='pyqtgraph', fig_size=None,
                 fps_limit=10, error_reset_delay=10.0, save_all_data=True,
                 motion_extraction_method='average'):
        assert isinstance(fps_limit, (int, float)) and fps_limit > 0, "fps_limit must be a positive int or float"
        assert isinstance(save_calibration_image, bool), "save_calibration_image must be bool"
        assert visualize == 'pyqtgraph' or visualize is None, \
            "visualize must be 'pyqtgraph' or None"
        assert fig_size is None or (isinstance(fig_size, (tuple, list)) and len(fig_size) == 2), \
            "fig_size should be None or length 2 tuple or list"
        assert isinstance(error_reset_delay, (int, float)) and error_reset_delay >= 0, \
            "error_reset_delay must be a positive int or float"
        assert isinstance(save_all_data, bool), "save_all_data should be bool"
        assert motion_extraction_method == "average" or motion_extraction_method == "flow", \
            "motion_extraction_method must be 'average' or 'flow'"

        self.error_reset_delay = error_reset_delay
        self.save_all_data = save_all_data
        self.fig_size = fig_size
        self.save_calibration_image = save_calibration_image
        self.capture_target = capture_target
        self.visualize = visualize
        self.motion_extraction_method = motion_extraction_method

        # Initialize Capture
        # noinspection PyArgumentList
        self.cap = cv2.VideoCapture(capture_target)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Calibration Variables
        '''
        Low Frequency Parameter, FPS, and Buffer Lengths
            For low frequency, len(self.calibration_buffer_target_length) / self.fps > 2 / self.freq_min for at least 2
            peaks to be present at all times.
        High Frequency Parameter and FPS
            For high frequency, 1 / self.freq_max > 2 / self.fps in order for the sample rate to fulfill the nyquist
            criteria.
        Buffer Length, Measurement Accuracy due to Drift, and Performance
            Excessive buffer lengths should be avoided for two reasons:
                1. Increasing the buffer length means variations in peak-to-peak intervals will be averaged
                substantially more. At higher frequencies, this can result in a very inaccurate reading. The ideal
                balance depends on the maximum rate of change of peak-to-peak interval which, at this point, is
                not known.
                2. A large buffer means each update step takes substantially longer, which can severely degrade 
                performance.
        Parameters Needing Investigation
            temporal_threshold - This threshold determines how well the localized region must match the temporal
            properties in order to be passed along to the next step (it should be very high as the threshold is
            based on a min-max range).
            threshold - This threshold determines the intensity of the previous threshold step which should be counted
            as a contour for the purpose of final localization (it should be very low as the threshold is meant to
            max out all values which aren't near 0).  
            gaussian_cutoff - 
            peak_variation_error_threshold - This method of signal integrity is probably not good, but it should be
            tested either way.
        '''
        self.maximum_bounding_box_area = np.inf  # Hyperparameter; If the bounding box is too large
        self.calibration_buffer_target_length = 128  # Hyperparameter; The number of frames to use for calibration
        self.freq_min = 0.1  # Hyperparameter; Minimum frequency to look for during calibration
        self.freq_max = 1.0  # Hyperparameter; Maximum frequency to look for during calibration
        self.temporal_threshold = 0.7  # Hyperparameter; The strength of temporal information to eliminate as noise
        self.threshold = 0.08  # Hyperparameter; The threshold of the temporal locator image to use for isolating a box

        # Measurement Variables
        self.measure_buffer_length = 128  # Hyperparameter; The buffer length for the measurement stream
        self.confidence_interval = 0.95  # Hyperparameter; The confidence interval for large motion

        self.feature_params = dict(maxCorners=100,  # Hyperparameter
                                   qualityLevel=0.3,  # Hyperparameter
                                   minDistance=7,  # Hyperparameter
                                   blockSize=7)  # Hyperparameter
        # Parameters for lucas kanade optical flow
        self.lk_params = dict(winSize=(15, 15),  # Hyperparameter
                              maxLevel=2,  # Hyperparameter
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))  # Hyperparameter(s)

        self.gaussian_cutoff = 10.0  # Hyperparameter; Gaussian curvature parameter; Not sure best value
        self.filter_order = 3  # Hyperparameter; The filter order for the raw data filtering

        # This parameter depends on the FPS and is auto-set once the FPS is fixed
        self.peak_minimum_sample_distance = 0
        # The number of frames to wait for to being measuring, 12 is the smallest possible value
        self.measure_initialization_length = 12

        # If no FPS is provided, set it to NaN to inform downstream checks that it isn't a valid FPS
        if self.fps == 0:
            self.fps = np.nan
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
        self.motion_data = deque()
        self.filtered_data = []

        self.peak_indices = []
        self.peak_times = []

        self.current_frame = None
        self.cropped_image = None
        self.previous_cropped_image = None
        self.display_frame = None
        self.motion_key_points = None
        self.video_out = None

        self.buffers = [self.data, self.confidence, self.t, self.freq, self.num_peaks, self.num_peaks_mean,
                        self.motion_data]

        if visualize == 'pyqtgraph':
            self.ui = self.initialize_pyqtgraph_visualization()

        # State Machine State
        self.state = 'initialize'

        '''Hack to avoid calibration for testing'''
        # self.skip_calibration(538, 243, 70, 51)
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
        self.peak_minimum_sample_distance = int(np.floor(self.fps / self.freq_max))
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

        image_view = layout.addViewBox()
        image_view.setAspectLocked(True)
        capture_image = pg.ImageItem(title='Capture Image', border='w')
        image_view.addItem(capture_image)

        right_plot = layout.addPlot(title="Frequency Plot (bpm)")
        right_plot.showGrid(x=True, y=True)
        right_plot.enableAutoRange('xy', False)

        frequency_plot = right_plot.plot()

        bpm_text = pg.TextItem(text='??? BPM', anchor=(-0.1, 1.2), color=(255, 255, 255, 255),
                               border=(0, 0, 0, 255), fill=(0, 0, 0, 127))
        bpm_text_font = pg.QtGui.QFont()
        bpm_text_font.setBold(True)
        bpm_text_font.setPointSize(24)
        bpm_text.setFont(bpm_text_font)
        image_view.addItem(bpm_text)
        bpm_text.setPos(0, 0)

        return {"raw_signal": raw_signal, "capture_image": capture_image, "frequency_plot": frequency_plot,
                "peak_plot": peak_plot, "bpm_text": bpm_text, "top_confidence_interval": top_confidence_interval,
                "bottom_confidence_interval": bottom_confidence_interval,
                "fill_confidence_interval": fill_confidence_interval, "fitted_plot": fitted_plot,
                "window": win, "plots": [left_plot, right_plot], "application": app}

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

    def set_plot_x_range(self, low, high):
        for plot in self.ui["plots"]:
            plot.setXRange(low, high, padding=0)

    def update_ui(self):
        if self.visualize == 'pyqtgraph':
            if self.state == "calibration":
                if self.calibration_buffer_idx < self.calibration_buffer_target_length:
                    self.set_window_title(
                        'Capturing calibration frames... {0}/{1}'.format(self.calibration_buffer_idx,
                                                                         self.calibration_buffer_target_length))
                    self.display_frame = self.current_frame
                    self.set_image(self.display_frame)
                else:
                    self.set_window_title('Measuring...')
            elif self.state == "measure":
                # First iteration
                if self.cropped_image is None:
                    self.set_plot_autoscale(True)
                    return
                self.display_frame = float_to_uint8(self.cropped_image)
                if self.motion_extraction_method == 'flow':
                    mask = np.zeros_like(self.display_frame)
                    for i, new in enumerate(self.motion_key_points):
                        a, b = new.ravel()
                        mask = cv2.circle(mask, (a, b), 2, (255, 255, 255), -1)
                        self.display_frame = cv2.add(self.display_frame, mask)
                self.set_window_title('Building Measurement Buffer.'+'.'.join(
                    ['' for _ in range(0, len(self.filtered_data) % 4)]))
                if len(self.peak_times) > 0:
                    self.ui["peak_plot"].setData(self.peak_times, np.take(self.filtered_data, self.peak_indices))
                self.set_window_title(
                    'Measuring.' + '.'.join(['' for _ in range(0, len(self.filtered_data) % 4)]))
                if len(self.filtered_data) >= 2 and len(self.t) >= 2:
                    self.set_plot_x_range(min(self.t), max(self.t))
                    self.ui["raw_signal"].setData(self.t, self.filtered_data)
                self.ui["capture_image"].setImage(self.display_frame)
                if len(self.freq) >= 2 and len(self.t) >= 2:
                    self.ui["frequency_plot"].setData(np.array(self.t)[-len(self.freq):], self.freq)
                    self.ui["bpm_text"].setText('{0:#.4} BPM'.format(self.freq[-1]))
                # if len(self.confidence) >= 2:
                #     ci_top, ci_bottom = np.transpose(self.confidence)
                #     ci_t = np.array(self.t)[-len(ci_top):]
                #     self.ui["top_confidence_interval"].setData(ci_t, ci_top)
                #     self.ui["bottom_confidence_interval"].setData(ci_t, ci_bottom)
            elif self.state == "error":
                self.set_window_title(
                    "Error: Recalibrating due to poor signal in {0}s.".format(
                        self.error_reset_delay - (time.time()-self.reset_start_time)))

            pg.QtGui.QApplication.processEvents()

    def initialize(self):
        self.calibration_start_time = time.time()
        self.calibration_buffer_idx = 0

    def detect_fps(self):
        if self.fps == 0 or self.fps is np.nan:
            self.fps = self.calibration_buffer_target_length / (time.time() - self.calibration_start_time)
            logging.info("Computer FPS as {0}.".format(self.fps))
        if self.fps > self.fps_limit:
            logging.info("FPS Limited to {0}.".format(self.fps))
            self.fps = self.fps_limit
        logging.info("Final FPS is {0}.".format(self.fps))

    def find_peaks(self):
        width = self.peak_minimum_sample_distance
        indices = peakutils.indexes(self.filtered_data, min_dist=width)

        final_idxs = []
        fits = []
        for idx in indices:
            w = width
            if idx - width < 0:
                w = idx
            if idx + w > len(self.t):
                w = len(self.t) - idx
            ti = np.array(self.t)[idx - w:idx + w]
            datai = np.array(self.filtered_data)[idx - w:idx + w]
            try:
                params = peakutils.gaussian_fit(ti, datai, center_only=False)
                y = [peakutils.gaussian(x, *params) for x in ti]
                # ax.plot(ti, y)
                ssr = np.sum(np.power(np.subtract(y, datai), 2.0))
                sst = np.sum(np.power(np.subtract(y, datai), 2.0))
                r2 = 1 - (ssr / sst)
                fits.append(r2)
                if params[2] < self.gaussian_cutoff:
                    final_idxs.append(idx)
            except RuntimeError:
                pass
        return final_idxs, fits

    def measure(self):
        # Filter data
        self.filtered_data = np.array(butter_lowpass_filter(self.data, self.freq_max*0.5, self.fps, self.filter_order))

        # Peak detection
        self.peak_indices, fits = self.find_peaks()

        self.peak_times = np.take(self.t, self.peak_indices)
        diffs = [a - b for b, a in zip(self.peak_times, self.peak_times[1:])]
        if len(diffs) > 0:
            interval = np.mean(diffs)
            est_freq = 60.0 / interval  # 60 seconds in a minute
            self.freq.append(est_freq)

    def extract_motion(self):
        if self.motion_extraction_method == "average":
            # Average the cropped image pixels
            avg = np.average(self.cropped_image)
            return avg

        elif self.motion_extraction_method == "flow":
            # On the first iteration, no motion can be detected because it requires two points, however,
            # we can detect the key points
            if self.previous_cropped_image is None:
                self.previous_cropped_image = float_to_uint8(self.cropped_image.copy())
                self.motion_key_points = cv2.goodFeaturesToTrack(self.previous_cropped_image,
                                                                 mask=None, **self.feature_params)
                return 0.0

            p1, st, err = cv2.calcOpticalFlowPyrLK(self.previous_cropped_image, float_to_uint8(self.cropped_image),
                                                   self.motion_key_points, None, **self.lk_params)
            # TODO: Check for no points
            # Select good points
            good_new = p1[st == 1]
            good_old = self.motion_key_points[st == 1]
            '''Restructure Data if Feature Points Change'''
            # Now update the previous frame and previous points
            self.previous_cropped_image = float_to_uint8(self.cropped_image.copy())
            self.motion_key_points = good_new.reshape(-1, 1, 2)

            # Failed to find keypoints, tracking lost
            if len(good_new) == 0 or len(good_old) == 0:
                return np.nan

            raw_motion_estimation = list(np.mean(good_old - good_new, axis=0))
            self.motion_data.append(raw_motion_estimation)
            '''PCA
                1. Find the first eigenvector and transform the points along that, 
                    getting only the primary motion component.
                2. Transform the feature points into the first component dimension.
                3. Lowpass filter the motion 
            '''
            if len(self.motion_data) >= 2:
                x, y = np.transpose(self.motion_data)
                coords = np.vstack([x, y])
                cov_mat = np.cov(coords)
                eig_vals, eig_vecs = np.linalg.eig(cov_mat)
                sort_indices = np.argsort(eig_vals)[::-1]
                evec1, evec2 = eig_vecs[:, sort_indices]
                reduced_data = np.array(self.motion_data).dot(evec1)
                motion_estimation = reduced_data[-1]
                return motion_estimation
            else:
                return 0.0

    def run(self):
        while self.cap.isOpened():
            self.loop_start_time = time.time()

            # Capture the frame (quit if the frame is a bool, meaning end of stream)
            self.current_frame = self.next_frame()
            if isinstance(self.current_frame, bool):
                break

            if self.state == 'initialize':
                self.initialize()
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
                    # Detect the FPS if needed
                    self.detect_fps()
                    # Fill FPS dependent variables
                    self.peak_minimum_sample_distance = int(np.floor(self.fps / self.freq_max))
                    # Run the localizer
                    location = self.locate(self.calibration_buffer, self.fps,
                                           save_calibration_image=self.save_calibration_image,
                                           freq_min=self.freq_min, freq_max=self.freq_max,
                                           temporal_threshold=self.temporal_threshold,
                                           threshold=int(np.round(self.threshold*255)))
                    # If the localizer fails, try again
                    if location is None:
                        logging.info("Failed finding ROI during calibration. Retrying...")
                        self.calibration_buffer_idx = 0
                        continue
                    # If the localizer didn't fail, save the values and reduce the bounding box as requested
                    self.x, self.y, self.w, self.h = location
                    self.x, self.y, self.w, self.h = reduce_bounding_box(self.x, self.y, self.w, self.h,
                                                                         self.maximum_bounding_box_area)
                    logging.info("Finished calibration.")
                    logging.info("Beginning measuring...")
                    self.calibration_progress_bar.close()

                    self.state = 'measure'
            elif self.state == 'measure':
                if self.save_all_data and self.video_out is None:
                    self.video_out = cv2.VideoWriter(str(self.capture_target) + '.avi',
                                                     cv2.VideoWriter_fourcc(*'MSVC'),
                                                     self.fps, (self.w, self.h))
                # Crop to the bounding box
                self.cropped_image = self.current_frame[self.y: self.y + self.h, self.x: self.x + self.w]
                # Check for full buffer and popleft
                for b in self.buffers:
                    if len(b) >= self.measure_buffer_length:
                        b.popleft()

                current_motion_value = self.extract_motion()
                self.data.append(current_motion_value)

                # Append to the temporal domain
                if len(self.t) == 0:
                    self.t.append(0.)
                else:
                    self.t.append(self.t[-1] + (1. / self.fps))
                # If the raw data is to be saved, add it to the dedicated list
                if self.save_all_data:
                    self.video_out.write(float_to_uint8(self.cropped_image))
                    self.all_data.append((self.t[-1], current_motion_value))
                if len(self.data) > self.measure_initialization_length:
                    # Perform the measurement
                    self.measure()
                    # Look for errors
                    if self.enable_error_detection and self.detect_errors():
                        self.state = 'error'
                        self.reset_start_time = time.time()
            elif self.state == 'error':
                if time.time() - self.reset_start_time >= self.error_reset_delay:
                    self.reset()
                    self.state = 'calibration'

            # Update the UI once the internal state has been set (will do nothing if visualize is None)
            self.update_ui()
            # Sleep the loop as needed to sync to the desired FPS
            self.sync_to_fps()

        logging.info("Capture closed.")

        self.cap.release()

        if self.save_all_data:
            self.video_out.release()
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
        raise NotImplementedError()

    @staticmethod
    def locate(calibration_video_data, fps,
               freq_min=0.1, freq_max=1.0, amplification=500,
               pyramid_levels=9, skip_levels_at_top=4, temporal_threshold=0.7,
               threshold=20, threshold_type=cv2.THRESH_BINARY,
               verbose=False, save_calibration_image=False):
        logging.info("Beginning processing calibration frames...")
        # Perform motion extraction
        op, raw = eulerian_magnification_bandpass(calibration_video_data, fps, freq_min, freq_max, amplification,
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

            avg_raw_frame = np.array(np.average(raw, axis=0))
            avg_raw_norm = ((avg_raw_frame - avg_raw_frame.min()) / (avg_raw_frame.max() - avg_raw_frame.min()))
            avg_raw = float_to_uint8(avg_raw_norm)
            avg_original = float_to_uint8(np.average(calibration_video_data, axis=0))
            row0 = np.hstack((avg_original, avg_raw, avg))
            row1 = np.hstack((thresh_copy, contour_img,  drawn))
            calibration = np.vstack((row0, row1))
            i = 0
            while os.path.exists("calibration%s.png" % i):
                i += 1
            cv2.imwrite(r'calibration%s.png' % i, calibration)
            logging.info('Calibration image saved.')

        if verbose:
            print('contour area:{4} - x:{0}, y:{1}, w:{2}, h:{3}'.format(x, y, w, h, area))

        return x, y, w, h
