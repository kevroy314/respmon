import pywt
import scipy
from prototypes import parabolic
import numpy as np
import scipy.fftpack
import scipy.signal as signal
import matplotlib.pyplot as plt
from prototypes.wavelets import plot_signal_decomp


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
    plt.figure(figsize=(20, 10))
    plt.subplots_adjust(hspace=.7)

    plt.subplot(charts_y, charts_x, 1)
    plt.title("Pixel Average")
    plt.xlabel("Time")
    plt.ylabel("Brightness")
    plt.plot(averages)

    freqs = scipy.fftpack.fftfreq(len(averages), d=1.0 / fps)
    fft = abs(scipy.fftpack.fft(averages))
    idx = np.argsort(freqs)

    plt.subplot(charts_y, charts_x, 2)
    plt.title("FFT")
    plt.xlabel("Freq (Hz)")
    freqs = freqs[idx]
    fft = fft[idx]

    freqs = freqs[int(len(freqs) / 2. + 1.):]
    fft = fft[int(len(fft) / 2. + 1.):]
    plt.plot(freqs, abs(fft))

    plt.show()


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
