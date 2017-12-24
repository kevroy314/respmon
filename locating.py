import os
import cv2
import logging
import numpy as np
from collections import deque
from transforms import uint8_to_float, float_to_uint8, temporal_bandpass_filter
from pyramid import create_laplacian_image_pyramid, collapse_laplacian_pyramid
import copy


# Set up logging
logging.basicConfig(format="%(asctime)s :: %(message)s", level=logging.INFO)

# Input params
run_original = True
verbose = True
temporal_filter_function = temporal_bandpass_filter
video_filename = r"C:\Users\kevin\Desktop\timber.mp4"
output_filename = r'C:\Users\kevin\Desktop\output0.avi'
pyramid_levels = 9
frame_buffer_length = 64
skip_levels_at_top = 4
# freq_range = (0.4, 0.5)  # fft timber heart rate params (500 amplification, lvl 4, skip 1, buff 64)
freq_range = (0.1, 1.0)
amplification = 500
codec = cv2.VideoWriter_fourcc(*'MSVC')  # -1 prompts with available

# Dynamic params
freq_min, freq_max = freq_range
logging.info(('Loading file {0}, {1} pyramid levels, {2}Hz to {3}Hz frequency range with {4}x amplification '
             '(frame buffer length is {5} and skip levels is {6})').format(video_filename, pyramid_levels,
                                                                           freq_min, freq_max, amplification,
                                                                           frame_buffer_length, skip_levels_at_top))
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
vid_gray = uint8_to_float(vid_gray)

if run_original:
    from base import eulerian_magnification_bandpass
    logging.info("Beginning processing frames...")
    op = eulerian_magnification_bandpass(vid_gray, fps, freq_min, freq_max, amplification,
                                         skip_levels_at_top=skip_levels_at_top, pyramid_levels=pyramid_levels,
                                         verbose=verbose)
    avg = np.array(np.average(op, axis=0))
    avg = ((avg - avg.min())/(avg.max() - avg.min()))

    # Find largest region
    avg = float_to_uint8(avg)
    cv2.imwrite(r'C:\Users\kevin\Desktop\average.png', avg)
    ret, thresh = cv2.threshold(avg, 20, 255, cv2.THRESH_BINARY)
    cv2.imwrite(r'C:\Users\kevin\Desktop\thresh.png', thresh)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    print(cv2.contourArea(c))
    # Draw it
    x, y, w, h = cv2.boundingRect(c)
    print('x:{0}, y:{1}, w:{2}, h:{3}'.format(x, y, w, h))
    drawn = cv2.rectangle(avg, (x, y), (x + w, y + h), 255, 2)
    cv2.imwrite(r'C:\Users\kevin\Desktop\contour.png', drawn)

    logging.info("Done processing frames.")
    logging.info("Writing output file...")
    writer = cv2.VideoWriter(output_filename, codec, fps, (width, height))

    for i in range(0, len(op)):
        output = cv2.rectangle(float_to_uint8(vid_gray[i] + drawn), (x, y), (x + w, y + h), 255, 2)
        writer.write(output)
    writer.release()
    logging.info("Done writing output file. Quitting.")
    exit()


# Build real-time buffers
pyramid_buffer = [deque() for _ in range(0, pyramid_levels)]

logging.info("Beginning capture loop.")
logging.info("Outputting result to {0}.".format(output_filename))
writer = cv2.VideoWriter(output_filename, codec, fps, (width, height))
prev_frame = None
diff_frame = np.zeros((height, width), dtype=np.float)
tmp = []
# Capture Loop
for i in range(0, len(vid_gray)):
    # If the frame buffer is full, remove the oldest elements
    if len(pyramid_buffer[0]) > frame_buffer_length:
        for buffer in pyramid_buffer:
            buffer.popleft()
    # Add the new pyramid to the pyramid buffer (expensive)
    lip = copy.deepcopy(create_laplacian_image_pyramid(vid_gray[i], pyramid_levels))

    # Add level-by-level for structure consistency
    for level in range(0, pyramid_levels):
        pyramid_buffer[level].append(lip[level])

    output_lip = copy.deepcopy(lip)
    # Iterate through levels for temporal filtering
    for j, buffer_layer in enumerate(pyramid_buffer):
        # Ignore the top and bottom of the pyramid
        if j < skip_levels_at_top or j >= len(pyramid_buffer) - 1:
            continue
        # Temporal filter each pyramid layer (expensive)
        bandpassed = temporal_filter_function(buffer_layer, fps, freq_min=freq_min, freq_max=freq_max,
                                              amplification_factor=amplification,
                                              debug='{0},{1}:'.format(i, j), verbose=verbose)
        output_lip[j] += bandpassed[-1]

    # Save the output frame
    frame = collapse_laplacian_pyramid([np.array(output_lip[level]) for level in range(0, pyramid_levels)])
    '''
    if prev_frame is None:
        prev_frame = frame
    else:
        # noinspection PyArgumentList
        mask = np.logical_not(np.logical_or.reduce((frame <= 0.0, prev_frame <= 0.0, frame >= 1.0, prev_frame >= 1.0)))
        diff_frame = np.subtract(np.array(frame), np.array(prev_frame),
                                 out=np.zeros(frame.shape),
                                 where=np.array(mask, dtype=bool))
        prev_frame = copy.deepcopy(frame)
        writer.write(float_to_uint8(diff_frame))
    '''
    tmp.append((np.array(frame).min(), np.array(frame).max()))
    writer.write(float_to_uint8(frame))
    logging.info("Finished frame {0}.".format(i))

print('min={0}, max={1}'.format(np.array(tmp).min(), np.array(tmp).max()))
writer.release()
