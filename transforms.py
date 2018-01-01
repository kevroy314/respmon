import cv2
import time
import copy
import pywt
import scipy
from prototypes import parabolic
import numpy as np
import scipy.fftpack
import scipy.signal as signal
import matplotlib.pyplot as plt
from prototypes.wavelets import plot_signal_decomp
from scipy.signal import butter, filtfilt
from pyramid import create_laplacian_video_pyramid, collapse_laplacian_video_pyramid


def nomrmalize(data):
    return (data - min(data))/(max(data)-min(data))


def uint8_to_float(img):
    result = np.ndarray(shape=img.shape, dtype='float')
    result[:] = img * (1. / 255)
    return result


def float_to_uint8(img):
    result = np.ndarray(shape=img.shape, dtype='uint8')
    result[:] = img * 255
    return result


def float_to_int8(img):
    result = np.ndarray(shape=img.shape, dtype='uint8')
    result[:] = (img * 255) - 127
    return result


def butter_bandpass(_lowcut, _highcut, _fs, order=5):
    _nyq = 0.5 * _fs
    _low = _lowcut / _nyq
    _high = _highcut / _nyq
    # noinspection PyTupleAssignmentBalance
    _b, _a = scipy.signal.butter(order, [_low, _high], btype='band', output='ba')
    return _b, _a


def butter_bandpass_filter(_data, _lowcut, _highcut, _fs, order=5):
    _b, _a = butter_bandpass(_lowcut, _highcut, _fs, order=order)
    _y = scipy.signal.lfilter(_b, _a, _data)
    return _y


def butter_bandpass_filter_fast(_data, _b, _a, axis=0):
    _y = scipy.signal.lfilter(_b, _a, _data, axis=axis)
    return _y


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


def temporal_bandpass_filter(data, fps, freq_min=0.833, freq_max=1, axis=0,
                             amplification_factor=50, verbose=False, debug=''):
    b, a = butter_bandpass(freq_min, freq_max, fps, order=6)
    result = butter_bandpass_filter_fast(data, b, a, axis=axis)
    result *= amplification_factor
    if verbose:
        print('{0}{1},{2}'.format(debug, result.min(), result.max()))
    return result


def temporal_bandpass_filter_fft(data, fps, freq_min=0.833, freq_max=1, axis=0,
                                 amplification_factor=50, verbose=False, debug=''):
    data_shape = (len(data), data[0].shape[0], data[0].shape[1])
    # noinspection PyUnresolvedReferences
    fft = scipy.fftpack.rfft(data, axis=axis)
    # noinspection PyUnresolvedReferences
    frequencies = scipy.fftpack.fftfreq(data_shape[0], d=1.0 / fps)
    bound_low = (np.abs(frequencies - freq_min)).argmin()
    bound_high = (np.abs(frequencies - freq_max)).argmin()
    fft[bound_high:-bound_high] = 0
    if bound_low != 0:
        fft[:bound_low] = 0
        fft[-bound_low:] = 0

    result = np.ndarray(shape=data_shape, dtype='float')
    # noinspection PyUnresolvedReferences
    result[:] = np.real(scipy.fftpack.ifft(fft, axis=0))
    result *= amplification_factor
    if verbose:
        print('{0}{1},{2}'.format(debug, result.min(), result.max()))
    return result


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


def wavelet_analysis(data):
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
        bandpassed_pyramid[i] += bandpassed
        vid_pyramid[i] += bandpassed
    t0 = time.time()

    # tmp = bandpassed_pyramid
    # img = np.hstack(tuple([float_to_uint8(cv2.copyMakeBorder(tmp[i][0],
    #                                                          tmp[0][0].shape[0] - tmp[i][0].shape[0],
    #                                                          0, 0, 0,
    #                                                          # tmp[0][0].shape[1] - tmp[i][0].shape[1],
    #                                                          cv2.BORDER_CONSTANT, value=(0, 0, 0))) for i in
    #                        range(0, len(tmp))]))
    # cv2.imwrite('pyramid.png', img)
    # vid_data = collapse_laplacian_video_pyramid(vid_pyramid)
    raw_bandpassed_data = collapse_laplacian_video_pyramid(bandpassed_pyramid)

    window_proportional_width = threshold
    min_val = raw_bandpassed_data.min()
    replace_value = min_val
    max_val = raw_bandpassed_data.max()
    intensity_filter_width = (max_val - min_val) * window_proportional_width
    top = max_val - intensity_filter_width
    mask = raw_bandpassed_data >= top
    bandpassed_data = copy.deepcopy(raw_bandpassed_data)
    bandpassed_data[mask] = replace_value
    t1 = time.time()
    if verbose:
        # print('min={0}, max={1}'.format(np.array(vid_data).min(), np.array(vid_data).max()))
        print("{2} (t={0}, dt={1})".format(t0, t1-t0, "collapse_laplacian_video_pyramid"))
        print("{2} (t={0}, dt={1})".format('n/a', (t1-t0)/float(len(vid_data)), "Frame Average"))
    return bandpassed_data, raw_bandpassed_data
