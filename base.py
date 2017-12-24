import numpy as np
import scipy.fftpack
import scipy.signal
import time
from matplotlib import pyplot
from eulerian_magnification.pyramid import create_laplacian_video_pyramid, collapse_laplacian_video_pyramid
from eulerian_magnification.transforms import temporal_bandpass_filter_fft


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


def show_frequencies(vid_data, fps, bounds=None):
    """Graph the average value of the video as well as the frequency strength"""
    averages = []

    if bounds:
        for x in range(1, vid_data.shape[0] - 1):
            averages.append(vid_data[x, bounds[2]:bounds[3], bounds[0]:bounds[1], :].sum())
    else:
        for x in range(1, vid_data.shape[0] - 1):
            averages.append(vid_data[x, :, :, :].sum())

    averages = np.array(averages) - np.array(min(averages))

    charts_x = 1
    charts_y = 2
    pyplot.figure(figsize=(20, 10))
    pyplot.subplots_adjust(hspace=.7)

    pyplot.subplot(charts_y, charts_x, 1)
    pyplot.title("Pixel Average")
    pyplot.xlabel("Time")
    pyplot.ylabel("Brightness")
    pyplot.plot(averages)

    freqs = scipy.fftpack.fftfreq(len(averages), d=1.0 / fps)
    fft = abs(scipy.fftpack.fft(averages))
    idx = np.argsort(freqs)

    pyplot.subplot(charts_y, charts_x, 2)
    pyplot.title("FFT")
    pyplot.xlabel("Freq (Hz)")
    freqs = freqs[idx]
    fft = fft[idx]

    freqs = freqs[int(len(freqs) / 2. + 1.):]
    fft = fft[int(len(fft) / 2. + 1.):]
    pyplot.plot(freqs, abs(fft))

    pyplot.show()
