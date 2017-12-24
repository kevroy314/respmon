import numpy as np
import scipy
from scipy.signal import butter, lfilter


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
    _b, _a = butter(order, [_low, _high], btype='band', output='ba')
    return _b, _a


def butter_bandpass_filter(_data, _lowcut, _highcut, _fs, order=5):
    _b, _a = butter_bandpass(_lowcut, _highcut, _fs, order=order)
    _y = lfilter(_b, _a, _data)
    return _y


def butter_bandpass_filter_fast(_data, _b, _a, axis=0):
    _y = lfilter(_b, _a, _data, axis=axis)
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
